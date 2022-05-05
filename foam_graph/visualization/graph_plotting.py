import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import torch
import numpy as np

from typing import Optional, Union


def plot_graph(
    graph: Data,
    field_name: Optional[str] = None,
    field_component: int = 0,
    ax: Optional[plt.Axes] = None,
    plot_colorbar: bool = False,
):
    """Plots a 2D graph.

    Args:
        graph (Data): Graph to be plotted.
        field_name (Optional[str], optional): Graph attribute used for coloring. Defaults to None.
        field_component (int, optional): Component of graph attribute used for coloring. Defaults to 0.
        ax (Optional[plt.Axes], optional): Axis to plot. Defaults to None.
        plot_colorbar (bool, optional): Flag for including a colorbar. Defaults to False.
    """
    if ax is None:
        ax = plt.gca()

    value_plotted = None
    if field_name is not None:
        value_plotted = graph[field_name]
    node_attrs = (
        ["pos", "y"] if value_plotted is not None else ["pos",]
    )
    graph = Data(edge_index=graph.edge_index, pos=graph.pos, y=value_plotted)

    graphnx = to_networkx(graph, node_attrs=node_attrs, to_undirected=True)
    selected_nodes = [n for n, v in graphnx.nodes(data=True)]
    subgraphnx = graphnx.subgraph(selected_nodes)
    n_comps = graph.y.size(-1)
    pos = nx.get_node_attributes(subgraphnx, "pos")
    if value_plotted is not None:
        node_color_dict = nx.get_node_attributes(graphnx, "y")
    for node in pos:
        pos[node] = pos[node][0:2]
        if value_plotted is not None and n_comps > 1:
            node_color_dict[node] = node_color_dict[node][field_component]
    node_color = (
        [node_color_dict[g] for g in subgraphnx.nodes]
        if value_plotted is not None
        else "k"
    )
    nodes = nx.draw_networkx_nodes(
        subgraphnx,
        pos=pos,
        node_color=node_color,
        node_size=10,
        cmap=plt.cm.viridis,
        ax=ax,
    )
    nx.draw_networkx_edges(
        subgraphnx, pos=pos, ax=ax,
    )
    if plot_colorbar:
        plt.colorbar(nodes, ax=ax)
    ax.axis("off")
    ax.axis("scaled")


def plot_graph_contour(
    graph: Data,
    field_name: Optional[str] = None,
    field_component: int = 0,
    ax: Optional[plt.Axes] = None,
    plot_colorbar: bool = False,
    internal_nodes_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
):
    """Plots a 2D graph with a triangulated filled countour.

    Args:
        graph (Data): Graph to be plotted.
        field_name (Optional[str], optional): Graph attribute used for coloring. Defaults to None.
        field_component (int, optional): Component of graph attribute used for coloring. Defaults to 0.
        ax (Optional[plt.Axes], optional): Axis to plot. Defaults to None.
        internal_nodes_mask (Optional[torch.Tensor], optional): Tensor of zeros and ones, where ones indicate nodes inside the domain. Defaults to None.    
    """
    if ax is None:
        ax = plt.gca()

    value_plotted = None
    if field_name is not None:
        value_plotted = graph[field_name]

    n_comps = value_plotted.size(-1)
    if value_plotted is not None and n_comps > 1:
        value_plotted = value_plotted[:, field_component]
    triang = Triangulation(graph.pos[:, 0], graph.pos[:, 1])

    if internal_nodes_mask is not None:
        internal = (
            internal_nodes_mask
            if isinstance(internal_nodes_mask, np.ndarray)
            else internal_nodes_mask.detach().numpy()
        )
        mask = np.sum(internal[triang.triangles], axis=1) < 1
        triang.set_mask(mask)
    ax.tricontourf(triang, value_plotted.flatten())

    if plot_colorbar:
        plt.colorbar(ax=ax)
    ax.axis("off")
    ax.axis("scaled")
