# %%
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer

from foam_graph.utils.graph_from_foam import read_foam
from foam_graph.nn.graph_network import GraphNetwork
from foam_graph.transforms import Make2D

from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split

# %% Set up the network and the data


class SimpleGraphNetwork(pl.LightningModule):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        num_target_features,
        hidden_channels,
        num_hidden,
        num_blocks,
    ):
        super().__init__()
        self.model = GraphNetwork(
            num_node_features,
            num_edge_features,
            num_target_features,
            hidden_channels,
            num_hidden,
            num_blocks,
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        y = self(batch)
        loss = F.mse_loss(y, batch.y)
        self.log_dict({'train_loss': loss})
        return loss

    def validation_step(self, val_batch, batch_idx):
        y = self(val_batch)
        loss = F.mse_loss(y, val_batch.y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class CFDDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        data = read_foam(
            "damBreak", ("U", "alpha.water"), read_boundaries=False,
        )
        self.transform = T.Compose(
            [
                Make2D("xy"),
                T.Cartesian(norm=False),
                T.Distance(norm=False),
            ]
        )
        data = self.transform(data)

        features = []
        targets = []
        for i, t in enumerate(data.time[1:]):
            features.append(torch.cat((data.U[i][:,0:2], data["alpha.water"][i-1]), -1).detach().numpy())
            targets.append(data["alpha.water"][i].detach().numpy())
        edge_attr = data.edge_attr.detach().numpy()

        self.num_node_features = features[0].shape[-1]
        self.num_edge_features = edge_attr.shape[-1]
        self.num_targets = targets[0].shape[-1]

        self.dataset = StaticGraphTemporalSignal(edge_index=data.edge_index, edge_weight=edge_attr, features=features, targets=targets)
        self.dataset.pos = data.pos

    def setup(self, stage = None):
        self.train_dataset, self.val_dataset = temporal_signal_split(self.dataset, train_ratio=0.8)

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset


cfd_data = CFDDataModule()
model = SimpleGraphNetwork(
    cfd_data.num_node_features,
    cfd_data.num_edge_features,
    cfd_data.num_targets,
    128,
    2,
    3,
)

# %% Train the network

trainer = Trainer(max_epochs=50)
trainer.fit(model, cfd_data)

# %% Plot the comparison between prediction and output

import matplotlib.pyplot as plt
from foam_graph.visualization.graph_plotting import plot_graph

graph = cfd_data.val_dataset[1]
graph.pos = cfd_data.dataset.pos
graph.y_pred = model(graph).detach()
graph.err = graph.y_pred - graph.y

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

plot_graph(graph, "y", ax=axes[0])
axes[0].set_title("Data")
axes[0].set_xlim([graph.pos[:, 0].min(), graph.pos[:, 0].max()])
axes[0].set_xlim([graph.pos[:, 1].min(), graph.pos[:, 1].max()])

plot_graph(graph, "y_pred", ax=axes[1])
axes[1].set_title("Prediction")
axes[1].set_xlim([graph.pos[:, 0].min(), graph.pos[:, 0].max()])
axes[1].set_xlim([graph.pos[:, 1].min(), graph.pos[:, 1].max()])

plt.tight_layout()
plt.show()
