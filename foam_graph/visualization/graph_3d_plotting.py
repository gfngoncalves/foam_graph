import plotly.graph_objects as go


def plot_3d_graph(graph, field_name=None, field_component=0, time=None):
    field_name_with_time = field_name
    if time is not None:
        field_name_with_time = f"{field_name_with_time}_time{time}"

    node_color = graph[field_name_with_time][:, field_component]

    x_nodes = [graph.pos[i][0] for i in range(graph.num_nodes)]
    y_nodes = [graph.pos[i][1] for i in range(graph.num_nodes)]
    z_nodes = [graph.pos[i][2] for i in range(graph.num_nodes)]

    x_edges = []
    y_edges = []
    z_edges = []

    for edge in graph.edge_index.T:
        x_coords = [graph.pos[edge[0]][0], graph.pos[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [graph.pos[edge[0]][1], graph.pos[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [graph.pos[edge[0]][2], graph.pos[edge[1]][2], None]
        z_edges += z_coords

    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line=dict(color="black", width=2),
        hoverinfo="none",
    )

    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers",
        marker=dict(
            symbol="circle",
            size=3,
            color=node_color,
            colorscale="Viridis",
            line=dict(color="black", width=0.5),
        ),
        hoverinfo="none",
    )

    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    layout = go.Layout(
        width=650,
        height=625,
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
        margin=dict(t=0),
        hovermode="closest",
    )

    data = [trace_edges, trace_nodes]
    fig = go.Figure(data=data, layout=layout)
    return fig
