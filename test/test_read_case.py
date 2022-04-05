import os
from foam_graph.utils.graph_from_foam import read_case

def test_read_case(rootdir):
    case_path = os.path.join(rootdir, 'data/minimumCase')
    graph = read_case(
        case_path,
        ("U", "p"),
        read_boundaries=True,
        times="all",
    )
    assert graph.num_nodes == 32
    assert graph.num_edges == 72

    assert len(graph.U) == 1
    assert graph.U[0].shape == (32, 3)

    assert len(graph.p) == 1
    assert graph.p[0].shape == (32, 1)

    assert graph.pos.shape == (32, 3)
