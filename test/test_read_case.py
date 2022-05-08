import os
from foam_graph.utils.graph_from_foam import read_foam
import pytest

def test_read_foam(rootdir):
    case_path = os.path.join(rootdir, 'data/minimumCase')
    graph = read_foam(
        case_path,
        ("U", "p", "U.list", "p.list"),
        read_boundaries=True,
        times=[0]
    )
    assert len(graph.U) == 1

    assert graph[0].num_edges == 56
    assert graph[0].U.shape == (24, 3)
    assert graph[0].p.shape == (24, 1)
    assert graph[0].pos.shape == (24, 3)

    with pytest.raises(ValueError):
        graph = read_foam(
            case_path,
            ("U", "p"),
            read_boundaries=True,
            times=[]
        )

    graph = read_foam(
        case_path,
        ("U", "p"),
        read_boundaries=True,
        times_indices=[0]
    )
    assert len(graph.U) == 1

    graph = read_foam(
        case_path,
        ("U", "p"),
        read_boundaries=True,
        times_indices=[slice(0, 1)]
    )
    assert len(graph.U) == 1

    graph = read_foam(
        case_path,
        ("U", "p"),
        read_boundaries=True,
        times_indices=[slice(0, 1)],
        global_attrs=(("nu", "constant/transportProperties", ("nu",)),)
    )
    assert graph[0].nu == pytest.approx(0.01)

def test_read_foam_dynamic(rootdir):
    case_path = os.path.join(rootdir, 'data/minimumCase')
    graph = read_foam(
        case_path,
        ("U", "p"),
        read_boundaries=True,
    )[1]
    assert graph.num_edges == 192
    assert graph.U.shape == (64, 3)
    assert graph.p.shape == (64, 1)
    assert graph.pos.shape == (64, 3)
