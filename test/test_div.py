import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_scatter import scatter_add
from foam_graph.transforms import NormalizeZScore
from foam_graph.utils.physics_calcs import div

def test_div():
    pos = torch.tensor(
        [[0.0, 0.0], [0.0, -1.0], [-1.0, 1.0], [1.0, 2.0],],
        dtype=torch.double,
        requires_grad=True,
    )
    edges = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0],])
    g = Data(pos=pos, edge_index=edges)

    transforms = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False),])
    g = transforms(g)

    wx = torch.tensor([1.69654, -1.40198, 1.20593])
    k = -2.80395

    y_target = torch.tensor([0, -3.0, 2.0, 1.0])
    div_target = 5.40302

    y1 = wx * scatter_add(g.edge_attr, g.edge_index[1], dim=0, dim_size=g.num_nodes)
    y1 = torch.sum(y1, 1) + k
    y2 = torch.zeros_like(y1)
    y = torch.column_stack((y1, y2))

    edge_normalize = NormalizeZScore("edge_attr", attr_mean=0.0)
    target_normalize = NormalizeZScore("y", attr_mean=0.0)
    div_loss = div(y, g, 0, edge_normalize, target_normalize)

    torch.testing.assert_close(y1.float(), y_target, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(div_loss.float(), torch.tensor(div_target))