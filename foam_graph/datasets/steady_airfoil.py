import torch
from foam_graph.data.foam_dataset import FoamDataset
from torch_geometric.data import extract_tar


class SteadyAirfoil(FoamDataset):
    field_names = ("p", "U", "yWall")
    times = "first_and_last"
    process = FoamDataset.process

    def __init__(self, root="data", transform=None, pre_transform=None):
        super().__init__(
            root,
            self.field_names,
            self.times,
            transform=transform,
            pre_transform=pre_transform,
        )

    def download(self):
        extract_tar("airfoils.tar.xz", self.raw_dir, mode="r:xz")
