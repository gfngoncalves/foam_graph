import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_scatter import scatter_add
from foam_graph.transforms import NormalizeZScore
from foam_graph.utils.physics_calcs import div


def test_div():
    pos = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],],
        dtype=torch.double,
        requires_grad=True,
    )
    edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2],])
    g = Data(pos=pos, edge_index=edges)

    transforms = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False),])
    graph = transforms(g)

    wx = torch.tensor(3.0)
    y1 = (
        0.5
        * wx
        * scatter_add(g.edge_attr[:, 0], g.edge_index[1], dim=0, dim_size=g.num_nodes)
    )
    y2 = y1
    y = torch.column_stack((y1, y2))

    edge_normalize = NormalizeZScore("edge_attr", attr_mean=0)
    target_normalize = NormalizeZScore("y", attr_mean=0)
    div_loss = div(y, graph, 0, edge_normalize, target_normalize)

    torch.testing.assert_close(div_loss.float(), wx.float())
