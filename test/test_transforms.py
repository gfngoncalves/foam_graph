import torch
from foam_graph.transforms import (
    KeepOnlyMainAttrs,
    Make2D,
    NormalizeZScore,
    Scale,
    Stack,
)

def test_KeepOnlyMainAttrs(simple_graph_3d):
    t = KeepOnlyMainAttrs("x")
    data_transform = t(simple_graph_3d)
    assert data_transform.keys == ["x"]


def test_Make2D(simple_graph_3d):
    t = Make2D("xy")
    data_transform = t(simple_graph_3d)
    torch.testing.assert_close(
        data_transform.pos, torch.tensor([[-1.0, 0.0], [0.0, 0.0]])
    )


def test_NormalizeZScore(simple_graph_3d):
    t = NormalizeZScore("x", -0.5, 0.5)
    data_transform = t(simple_graph_3d)
    torch.testing.assert_close(data_transform.x, torch.tensor([[-1.0], [1.0]]))


def test_Scale(simple_graph_3d):
    t = Scale("x", -1, 0)
    data_transform = t(simple_graph_3d)
    torch.testing.assert_close(data_transform.x, torch.tensor([[0.0], [1.0]]))


def test_Stack(simple_graph_3d):
    simple_graph_3d.y = torch.tensor([[2.0], [2.0]], dtype=torch.float)
    t = Stack("x", ["x", "y"])
    data_transform = t(simple_graph_3d)
    torch.testing.assert_close(
        data_transform.x, torch.tensor([[-1.0, 2.0], [0.0, 2.0]])
    )

