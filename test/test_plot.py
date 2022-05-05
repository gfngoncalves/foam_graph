import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from foam_graph.visualization.graph_plotting import plot_graph, plot_graph_contour
from foam_graph.visualization.graph_3d_plotting import plot_3d_graph


def test_plot_2d():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.tensor([[-1], [0]], dtype=torch.float)
    pos = torch.tensor([[-1, 0], [0, 0]], dtype=torch.float)
    edge_attr = torch.tensor([[0]], dtype=torch.float)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    fig, ax = plt.subplots()
    plot_graph(data, "x", 0, ax=ax)

    lines = ax.collections[1].get_segments()
    np.testing.assert_equal(lines, [[[-1.0, 0.0], [0.0, 0.0]]])

    points = ax.collections[0]
    np.testing.assert_equal(points.get_offsets().data, [[-1.0, 0.0], [0.0, 0.0]])


def test_plot_contour_2d():
    edge_index = torch.tensor([], dtype=torch.long)
    x = torch.tensor([[1], [1], [1], [1], [1]], dtype=torch.float)
    pos = torch.tensor([[-1, 0], [0, 0], [-1, 1], [1, 1], [0, 1]], dtype=torch.float)
    internal = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)

    data = Data(x=x, pos=pos, edge_index=edge_index)

    fig, ax = plt.subplots()
    plot_graph_contour(data, "x", 0, ax=ax, internal_nodes_mask=internal)
    paths = ax.collections[1].get_paths()
    np.testing.assert_equal(paths[0].vertices, [[0.0, 0.0], [-1.0, 1.0], [-1.0, 0.0]])


def test_plot_3d():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.tensor([[-1], [0]], dtype=torch.float)
    pos = torch.tensor([[-1, 0, 0], [0, 0, 1]], dtype=torch.float)
    edge_attr = torch.tensor([[0]], dtype=torch.float)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    fig = plot_3d_graph(data, "x", 0)

    lines = fig.data[0]

    assert lines.x == (-1.0, 0.0, None, 0.0, -1.0, None)
    assert lines.y == (0.0, 0.0, None, 0.0, 0.0, None)
    assert lines.z == (0.0, 1.0, None, 1.0, 0.0, None)

    points = fig.data[1]
    np.testing.assert_equal(points.x, [-1.0, 0.0])
    np.testing.assert_equal(points.y, [0.0, 0.0])
    np.testing.assert_equal(points.z, [0.0, 1.0])

