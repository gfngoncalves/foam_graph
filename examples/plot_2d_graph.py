# %%
import matplotlib.pyplot as plt
from torch_geometric.data import download_url, extract_tar
from foam_graph.utils.graph_from_foam import read_foam
from foam_graph.visualization.graph_plotting import plot_graph

# %% Extract tar and read case as a graph
download_url("https://github.com/gfngoncalves/openfoam_cases/blob/main/damBreak.tar.xz?raw=true", ".")
extract_tar("damBreak.tar.xz", ".", mode="r:xz")

graph = read_foam(
    "damBreak",
    ("alpha.water",),
    read_boundaries=True,
)

#%% Plot alpha field for 2D case

field_name_plot = "alpha.water"
field_component_plot = 0
time = -1

fig, ax = plt.subplots(figsize=(10, 10))
plot_graph(graph[time], field_name_plot, field_component_plot, ax=ax)

plt.tight_layout()
plt.show()
