import os
import torch
from torch_geometric.data import Data

from distutils.dir_util import copy_tree
from foam_graph.utils.graph_from_foam import read_foam
from foam_graph.utils.foam_from_graph import write_foam
import pytest


def test_write_case(rootdir, tmp_path):
    case_path = (tmp_path / "minimumCase").as_posix()
    copy_tree(os.path.join(rootdir, "data/minimumCase"), case_path)
    n = 32
    x = torch.reshape(torch.tensor(range(n * 3)), (-1, 3))
    y = torch.reshape(torch.tensor(range(n * 1)), (-1, 1))
    data = Data(x=x, y=y)

    write_foam(case_path, 0, data, ["x", "y"], ["U", "p"], ["U.out", "p.out"])
    graph = read_foam(case_path, ("U.out", "p.out"),)[0]
    torch.testing.assert_close(graph["U.out"].float(), x.float())
    torch.testing.assert_close(graph["p.out"].float(), y.float())

    with pytest.raises(ValueError):
        x = torch.reshape(torch.tensor(range(1 * 3)), (-1, 3))
        data = Data(x=x)
        write_foam(case_path, 0, data, ["x"], ["U"], ["U.out"])
