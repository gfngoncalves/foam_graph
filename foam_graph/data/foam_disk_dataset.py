import os.path as osp
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset, download_url

from foam_graph.utils.graph_from_foam import read_case


class FoamDiskDataset(Dataset):
    def __init__(
        self,
        root,
        field_names,
        features,
        targets,
        features_comps,
        targets_comps,
        pos_comps,
        zero_and_last_times=False,
        transform=None,
        pre_transform=None,
    ):

        self.field_names = field_names
        self.features = features
        self.features_comps = features_comps
        self.targets = targets
        self.targets_comps = targets_comps
        self.pos_comps = pos_comps
        self.zero_and_last_times = zero_and_last_times
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        p = Path(self.raw_dir)
        return [x.name for x in p.iterdir() if x.is_dir()]

    @property
    def processed_file_names(self):
        p = Path(self.raw_dir)
        return [f"{x.name}.pt" for x in p.iterdir() if x.is_dir()]

    # def download(self):
    # Download to `self.raw_dir`.
    # path = download_url(url, self.raw_dir)

    def process(self):
        for i, raw_path in enumerate(self.raw_paths):
            # Read data from `raw_path`.
            p = Path(raw_path)
            data = read_case(raw_path, self.field_names, self.zero_and_last_times)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data.x = torch.hstack(
                [
                    data[f] if data[f].size(-1) == 1 else data[f][:, c]
                    for f, c in zip(self.features, self.features_comps)
                ]
            )
            data.y = torch.hstack(
                [
                    data[t] if data[t].size(-1) == 1 else data[t][:, c]
                    for t, c in zip(self.targets, self.targets_comps)
                ]
            )

            data.pos = data.pos[:, self.pos_comps]

            data = Data(edge_index=data.edge_index, pos=data.pos, x=data.x, y=data.y)

            torch.save(data, osp.join(self.processed_dir, f"{p.name}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data
