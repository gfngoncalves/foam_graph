Example
======================================

Currently, the package requires that cell centers are written by OpenFOAM.
That can be achieved with the following command (requires OpenFOAM)::

    postProcess -func writeCellCentres

The following example loads a case folder as a graph::

    from torch_geometric.data import extract_tar
    from foam_graph.utils.graph_from_foam import read_case

    extract_tar("examples/damBreak.tar.xz", ".", mode="r:xz")

    graph = read_case(
        "damBreak",
        ("alpha.water",),
        read_boundaries=True,
        times="first_and_last",
    )

The resulting graph is a Data object.