import os
import torch
from torch_geometric.data import Data
import pytest

@pytest.fixture
def rootdir():
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def simple_graph_3d():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.tensor([[-1], [0]], dtype=torch.float)
    pos = torch.tensor([[-1, 0, 0], [0, 0, 1]], dtype=torch.float)
    edge_attr = torch.tensor([[0]], dtype=torch.float)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return data