import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt


def plot_graph(graph, field_name=None, field_component=0, time=None, ax=None):
    if ax is None:
        ax = plt.gca()
    field_name_with_time = field_name
    if time is not None and field_name is not None:
        field_name_with_time = f"{field_name_with_time}_time{time}"
    node_attrs = (
        ("pos", field_name_with_time) if field_name_with_time is not None else ("pos",)
    )
    graphnx = to_networkx(graph, node_attrs=node_attrs, to_undirected=True)
    selected_nodes = [n for n, v in graphnx.nodes(data=True)]
    subgraphnx = graphnx.subgraph(selected_nodes)
    n_comps = graph[field_name_with_time].size(-1)
    pos = nx.get_node_attributes(subgraphnx, "pos")
    if field_name_with_time is not None:
        node_color_dict = nx.get_node_attributes(graphnx, field_name_with_time)
    for node in pos:
        pos[node] = pos[node][0:2]
        if field_name_with_time is not None and n_comps > 1:
            node_color_dict[node] = node_color_dict[node][field_component]
    node_color = (
        [node_color_dict[g] for g in subgraphnx.nodes]
        if field_name_with_time is not None
        else "k"
    )
    nx.draw(
        subgraphnx,
        pos=pos,
        node_color=node_color,
        node_size=10,
        cmap=plt.cm.viridis,
        ax=ax,
    )
    ax.axis("scaled")
