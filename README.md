# FoamGraph

![foam_graph_logo](docs/source/images/foam_graph_logo.svg)

--------------------------------------------------------------------------------

FoamGraph is a Python library for manipulating OpenFOAM cases as graphs.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foam_graph.

```bash
pip install .
```
For 3D plotting, *plotly* is required.

## Example

Currently, the package requires that cell centers are written by OpenFOAM.
That can be achieved with the following command (requires [OpenFOAM](https://www.openfoam.com/)):

```bash
postProcess -func writeCellCentres
```

The following example loads a case folder as a graph:

```python
from torch_geometric.data import download_url, extract_tar
from foam_graph.utils.graph_from_foam import read_case

download_url("https://github.com/gfngoncalves/openfoam_cases/blob/main/damBreak.tar.xz?raw=true", ".")
extract_tar("damBreak.tar.xz", ".", mode="r:xz")

graph = read_case(
    "damBreak",
    ("alpha.water",),
    read_boundaries=True,
    times="all",
)
```

The resulting graph is a [StaticGraphTemporalSignal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/signal.html) object.

More examples are provided in the *examples* folder.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)