import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt


def plot_graph(
    graph, field_name=None, field_component=0, time=None, ax=None, plot_colorbar=False
):
    if ax is None:
        ax = plt.gca()

    value_plotted = None
    if field_name is not None:
        if time is not None:
            value_plotted = graph[field_name][time]
        else:
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
