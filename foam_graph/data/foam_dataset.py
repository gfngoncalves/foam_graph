import os.path as osp
from pathlib import Path
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
import torch_geometric.transforms as T
from foam_graph.utils.graph_from_foam import read_case
from foam_graph.transforms import Stack, KeepOnlyMainAttrs


class FoamDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        field_names,
        times="all",
        transform=None,
        pre_transform=None,
    ):

        self.field_names = field_names
        self.times = times

        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        p = Path(self.raw_dir)
        return [x.name for x in p.iterdir() if x.is_dir()]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    # def download(self):
    # Download to `self.raw_dir`.
    # path = download_url(url, self.raw_dir)

    def process(self):
        data_list = []
        for i, raw_path in enumerate(self.raw_paths):
            p = Path(raw_path)
            data = read_case(raw_path, self.field_names, self.times)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
