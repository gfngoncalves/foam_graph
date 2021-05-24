# %%
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import extract_tar
from foam_graph.utils.graph_from_foam import read_case
from foam_graph.visualization.graph_plotting import plot_graph

# %% Extract tar and read case as a graph
extract_tar("damBreak.tar.xz", ".", mode="r:xz")

graph = read_case(
    "damBreak",
    ("alpha.water",),
    read_boundaries=True,
    times="first_and_last",
)

#%% Plot alpha field for 3D case

field_name_plot = "alpha.water"
field_component_plot = 0
time = 1

fig, ax = plt.subplots(figsize=(10, 10))
plot_graph(graph, field_name_plot, field_component_plot, time=time, ax=ax)

plt.tight_layout()
plt.show()
