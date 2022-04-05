import torch
from torch_geometric.data import Data
from foam_graph.nn import GraphNetwork


def test_graph_network():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    edge_attr = torch.tensor([[0], [1], [1], [0.5]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    net = GraphNetwork(1, 1, 1, 1, 1, 1,)
    out = net(data)

    assert out.size() == (3, 1)
