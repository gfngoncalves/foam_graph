# %%
import numpy as np
from torch_geometric.data import download_url, extract_tar
from foam_graph.utils.graph_from_foam import read_foam
from foam_graph.visualization.graph_3d_plotting import plot_3d_graph

# %% Extract tar and read case as a graph
download_url("https://github.com/gfngoncalves/openfoam_cases/blob/main/angledDuct.tar.xz?raw=true", ".")
extract_tar("angledDuct.tar.xz", ".", mode="r:xz")

graph = read_foam(
    "angledDuct",
    ("alpha.water",),
    read_boundaries=False,
    times="first_and_last",
)

# %% Plot alpha field for 3D case

field_name = "alpha.water"
field_component = 0
time = -1

fig = plot_3d_graph(graph[time], field_name=field_name, field_component=field_component)
fig.show()
