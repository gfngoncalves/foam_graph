from pathlib import Path

import torch

import torch_geometric.transforms as T
from torch_geometric.data import download_url, extract_tar

from foam_graph.utils.graph_from_foam import read_foam
from foam_graph.transforms import Make2D, NormalizeZScore

from torch_geometric_temporal.signal import (
    StaticGraphTemporalSignal,
    temporal_signal_split,
)

import numpy as np

from itertools import tee

try:
    from pytorch_lightning import LightningDataModule as PLLightningDataModule

    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    PLLightningDataModule = object
    no_pytorch_lightning = True


def _pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def _running_mean_and_std(data, axis=None):
    n = 0
    mean = 0.0
    m2 = 0.0

    for x in data:
        n_old = n

        n_i = np.size(x, axis)
        mean_i = np.mean(x, axis)
        var_i = np.var(x, axis)

        n += n_i
        delta = mean_i - mean
        mean += float(n_i) / n * delta
        m2 += var_i * n_i + delta ** 2 * float(n_old * n_i) / n

    if n:
        return (mean, np.sqrt(m2 / n))
    else:
        return (np.nan, np.nan)


class IcoDataModule(PLLightningDataModule):
    def __init__(self, case_name, data_url=None, train_ratio=0.8, noise=None):
        super().__init__()

        if no_pytorch_lightning:
            raise ModuleNotFoundError(
                "No module named 'pytorch_lightning' found on this machine. "
                "Run 'pip install pytorch_lightning' to install the library."
            )

        self.case_name = case_name
        self.data_url = data_url
        self.train_ratio = train_ratio
        self.noise = {} if noise is None else noise

        self.edge_transform = T.Compose(
            [Make2D("xy"), T.Cartesian(norm=False), T.Distance(norm=False),]
        )
        self.edge_normalize = NormalizeZScore("edge_attr")
        self.feature_normalize = NormalizeZScore("x")
        self.target_normalize = NormalizeZScore("y")
        self.has_setup_fit = False

    def prepare_data(self):
        if self.data_url is not None:
            if Path(self.case_name).is_dir():
                print(f"Directory {self.case_name} already exists, skipping download")
                return
            path_tar = download_url(self.data_url, ".")
            extract_tar(path_tar, ".", mode="r:xz")

    def calculate_normalization(self, data):
        # Calculate maximum and minimum of edge features and use that to normalize
        edge_min, edge_max = [
            f(data.edge_weight, axis=(0)).values for f in (torch.min, torch.max)
        ]
        self.edge_normalize.attr_mean = 0.5 * (edge_max - edge_min)
        self.edge_normalize.attr_std = edge_max - edge_min

        # Calculate the mean and std. dev. of features and targets and use that to normalize
        x_mean, x_std = _running_mean_and_std(data.features, axis=0)
        self.feature_normalize.attr_mean = torch.tensor(x_mean)
        self.feature_normalize.attr_std = torch.tensor(x_std)

        y_mean, y_std = _running_mean_and_std(data.targets, axis=0)
        self.target_normalize.attr_mean = torch.tensor(y_mean)
        self.target_normalize.attr_std = torch.tensor(y_std)

    def load_dataset(self):
        return read_foam(self.case_name, ("U", "p"), read_boundaries=True,)

    def generate_features(self, graph_previous):
        return torch.cat((graph_previous.U[:, 0:2], graph_previous.boundary,), -1)

    def generate_targets(self, graph_previous, graph):
        return torch.cat((graph.U[:, 0:2] - graph_previous.U[:, 0:2], graph.p,), -1)

    def add_features_graph(self, graph):
        graph.x = self.generate_features(graph)
        graph = self.edge_transform(graph)
        graph = self.edge_normalize(graph)
        return graph

    def preprocess_graph(self, graph):
        graph = self.feature_normalize(graph)
        return graph

    def postprocess_graph(self, graph):
        graph = self.feature_normalize.unscale(graph)
        graph = self.target_normalize.unscale(graph)
        return graph

    def state_dict(self):
        return {
            "edge_transform": self.edge_transform,
            "edge_normalize": self.edge_normalize,
            "feature_normalize": self.feature_normalize,
            "target_normalize": self.target_normalize,
        }

    def load_state_dict(self, state_dict):
        self.edge_transform = state_dict["edge_transform"]
        self.edge_normalize = state_dict["edge_normalize"]
        self.feature_normalize = state_dict["feature_normalize"]
        self.target_normalize = state_dict["target_normalize"]

    def setup(self, stage=None):
        if not self.has_setup_fit and (stage == "fit" or stage is None):
            self.dataset = self.load_dataset()

            # Add noise to the velocity
            rng = np.random.default_rng()
            for variable, noise_level in self.noise.items():
                for f in getattr(self.dataset, variable):
                    f += rng.normal(scale=noise_level, size=f.shape)

            # Apply transforms to generate edge features
            edge_attr = self.edge_transform(self.dataset[0]).edge_attr

            # Generate features and targets
            features = [
                self.generate_features(graph_previous).detach().numpy()
                for graph_previous, _ in _pairwise(self.dataset)
            ]
            targets = [
                self.generate_targets(graph_previous, graph).detach().numpy()
                for graph_previous, graph in _pairwise(self.dataset)
            ]

            self.dataset = StaticGraphTemporalSignal(
                edge_index=self.dataset.edge_index,
                edge_weight=edge_attr,
                features=features,
                targets=targets,
                pos=self.dataset.pos[: len(targets)],
            )

            # Split into train and test datasets
            self.train_dataset, self.val_dataset = temporal_signal_split(
                self.dataset, train_ratio=self.train_ratio
            )

            # Calculate statistics for normalization
            self.calculate_normalization(self.train_dataset)

            # Normalize
            self.dataset.edge_attr = (
                self.edge_normalize(self.dataset[0]).edge_attr.detach().numpy()
            )

            for i, data in enumerate(self.dataset):
                features = self.feature_normalize(data).x
                self.dataset.features[i] = features.detach().numpy()

                targets = self.target_normalize(data).y
                self.dataset.targets[i] = targets.detach().numpy()

            self.num_node_features = self.dataset.features[0].shape[-1]
            self.num_edge_features = self.dataset.edge_weight.shape[-1]
            self.num_targets = self.dataset.targets[0].shape[-1]

            self.has_setup_fit = True

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset
