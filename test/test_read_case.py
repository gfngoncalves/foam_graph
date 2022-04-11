import os
from foam_graph.utils.graph_from_foam import read_case

def test_read_case(rootdir):
    case_path = os.path.join(rootdir, 'data/minimumCase')
    graph = read_case(
        case_path,
        ("U", "p"),
        read_boundaries=True,
        times="all",
    )[0]
    assert graph.num_edges == 72
    assert graph.U.shape == (32, 3)
    assert graph.p.shape == (32, 1)
    assert graph.pos.shape == (32, 3)
