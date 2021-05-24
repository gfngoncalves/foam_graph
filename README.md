# foam_graph

foam_graph is a Python library for manipulating OpenFOAM cases as graphs.

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
from torch_geometric.data import extract_tar
from foam_graph.utils.graph_from_foam import read_case

extract_tar("examples/damBreak.tar.xz", ".", mode="r:xz")

graph = read_case(
    "damBreak",
    ("alpha.water",),
    read_boundaries=True,
    times="first_and_last",
)
```

The resulting graph is a [Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) object.

More examples are provided in the *examples* folder.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)