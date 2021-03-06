import torch
from foam_graph.transforms import (
    ApplyLambdaTemporalSignal,
    KeepOnlyMainAttrs,
    Make2D,
    NormalizeZScore,
    Scale,
    Stack,
)
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import pytest


def test_KeepOnlyMainAttrs(simple_graph_3d):
    t = KeepOnlyMainAttrs("x")
    data_transform = t(simple_graph_3d)
    assert data_transform.keys == ["x"]


def test_Make2D(simple_graph_3d):
    pos_orig = simple_graph_3d.pos.clone().detach()
    assert pos_orig.size(1) == 3

    t = Make2D("xy")
    data_transform = t(simple_graph_3d)
    assert data_transform.pos.size(1) == 2
    torch.testing.assert_close(data_transform.pos, pos_orig[:, :2])

    with pytest.raises(ValueError):
        t = Make2D("")


def test_NormalizeZScore(simple_graph_3d):
    x_orig = simple_graph_3d.x.clone().detach()

    t = NormalizeZScore("x", x_orig.mean(), x_orig.max() - x_orig.min())
    data_transform = t(simple_graph_3d)
    assert data_transform.x.min() == pytest.approx(-0.5)
    assert data_transform.x.max() == pytest.approx(0.5)

    data_transform = t.unscale(data_transform)
    torch.testing.assert_close(data_transform.x, x_orig)


def test_Scale(simple_graph_3d):
    x_orig = simple_graph_3d.x.clone().detach()

    t = Scale("x", simple_graph_3d.x.min(), simple_graph_3d.x.max())
    data_transform = t(simple_graph_3d)
    assert data_transform.x.min() == pytest.approx(0)
    assert data_transform.x.max() == pytest.approx(1)

    data_transform = t.unscale(data_transform)
    torch.testing.assert_close(data_transform.x, x_orig)


def test_Stack(simple_graph_3d):
    simple_graph_3d.y = torch.tensor([[2.0], [2.0]], dtype=torch.float)
    x_orig = simple_graph_3d.x.clone().detach()
    y_orig = simple_graph_3d.y.clone().detach()

    t = Stack("z", ["x", "y"])
    data_transform = t(simple_graph_3d)
    torch.testing.assert_close(data_transform.z[:, 0:1], x_orig)
    torch.testing.assert_close(data_transform.z[:, 1:2], y_orig)

    data_transform = t.unstack(data_transform)
    torch.testing.assert_close(data_transform.x, x_orig)
    torch.testing.assert_close(data_transform.y, y_orig)


def test_ApplyLambdaTemporalSignal(simple_graph_3d):
    n = 2
    datas = StaticGraphTemporalSignal(
        simple_graph_3d.edge_index.detach().numpy(),
        None,
        [simple_graph_3d.x.detach().numpy() for i in range(n)],
        [None for i in range(n)],
        pos=[simple_graph_3d.x.detach().numpy() for i in range(n)],
    )
    t = ApplyLambdaTemporalSignal("targets", lambda datas: datas[1].x - datas[0].x, 1)
    datas_transform = t(datas)
    data_transform = datas_transform[0]
    torch.testing.assert_close(data_transform.y, simple_graph_3d.x - simple_graph_3d.x)
