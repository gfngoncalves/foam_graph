# %%
import numpy as np
from torch_geometric.data import extract_tar
from foam_graph.utils.graph_from_foam import read_case
from foam_graph.visualization.graph_3d_plotting import plot_3d_graph

# %% Extract tar and read case as a graph
extract_tar("angledDuct.tar.xz", ".", mode="r:xz")

graph = read_case(
    "angledDuct",
    ("alpha.water",),
    read_boundaries=False,
    times="first_and_last",
)

# %% Plot alpha field for 2D case

field_name = "alpha.water"
field_component = 0
time = 10

fig = plot_3d_graph(
    graph, field_name=field_name, field_component=field_component, time=time
)
fig.show()
