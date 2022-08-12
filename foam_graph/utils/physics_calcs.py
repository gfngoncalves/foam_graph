from typing import Callable, Iterable
import torch
from torch import Tensor
from torch_geometric.data import Data


def _mul_sparse_dense(m_sparse, m_dense):
    return torch.mul(m_dense.expand(m_sparse.shape).sparse_mask(m_sparse), m_sparse)


def _d_edge_attr_d_pos_node(graph: Data, j: int, indices: Tensor, i_mag_dx: int):
    vals = graph.edge_attr[indices]

    idxs_dx = torch.full_like(vals[:, 0], j)
    idxs_mag_dx = torch.full_like(vals[:, 0], i_mag_dx % vals.size(1))

    idxs_dx_edges = torch.stack((indices, idxs_dx), 0)
    idxs_mag_dx_edges = torch.stack((indices, idxs_mag_dx), 0)

    vals_dx = -torch.ones_like(vals[:, 0])
    vals_mag_dx = -vals[:, j] / vals[:, i_mag_dx]

    v1 = torch.sparse_coo_tensor(idxs_dx_edges, vals_dx, graph.edge_attr.size())
    v2 = torch.sparse_coo_tensor(idxs_mag_dx_edges, vals_mag_dx, graph.edge_attr.size())

    return v1 + v2


def _d_edge_attr_d_pos(graph: Data, i: int, j: int, i_mag_dx: int):
    indices_orig = torch.transpose((graph.edge_index[0, :] == i).nonzero(), 0, 1)
    indices_dest = torch.transpose((graph.edge_index[1, :] == i).nonzero(), 0, 1)

    s_orig = _d_edge_attr_d_pos_node(graph, j, indices_orig[0], i_mag_dx)
    s_dest = _d_edge_attr_d_pos_node(graph, j, indices_dest[0], i_mag_dx)

    s = s_dest - s_orig
    return s.coalesce().detach()


def _d_edge_attr_norm_d_edge_attr(graph: Data, edge_normalize: Callable[[Data], Data]):
    return torch.autograd.functional.vjp(
        lambda x: edge_normalize(Data(edge_attr=x.clone())).edge_attr,
        torch.ones_like(graph.edge_attr[0]),
        torch.ones_like(graph.edge_attr[0]),
    )[1]


def _d_y_d_y_norm(y: Tensor, target_normalize: Callable[[Data], Data]):
    return torch.autograd.functional.vjp(
        lambda x: target_normalize.unscale(Data(y=x.clone())).y,
        torch.ones_like(y[0]),
        torch.ones_like(y[0]),
    )[1]


def _d_y_norm_d_edge_attr_norm(y: Tensor, graph: Data, i: int, j: int):
    ones = torch.ones_like(y[i, j])
    return torch.autograd.grad(
        y[i, j],
        graph.edge_attr,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]


def div(
    y: Tensor,
    graph: Data,
    i: int,
    edge_normalize: Callable[[Data], Data],
    target_normalize: Callable[[Data], Data],
    indices_distances: Iterable[int] = (0, 1),
    indice_distance_mag: int = -1,
    indices_target_vector: Iterable[int] = (0, 1),
) -> Tensor:
    """Computes the divergent of a node target in respect to node positions.

    Assumes that a vector quantity in y is a function of node distance components and magnitude, 
    stored in graph.edge_attr.

    Calculates the value with the chain rule:
        d(y)/d(pos) = 
           d(y)/d(y_norm) * d(y_norm)/d(edge_attr_norm)
         * d(edge_attr_norm)/d(edge_attr) * d(edge_attr)/d(pos)

    The terms on the RHS are calculated as:
    2) Autograd applied to target_normalize
    2) Autograd applied to y
    3) Autograd applied to edge_normalize
    4) Analytical expression.

    Args:
        y (Tensor): Node targets.
        graph (Data): Input graph.
        i (int): Index of the node.
        edge_normalize (Callable[[Data], Data]): Function that normalizes the edge attibutes.
        indices_distances (Iterable[int], optional): Indices of the graph.edge_attr that correspond to the distance components. Defaults to (0, 1).
        indice_distance_mag (int, optional): Indice of the graph.edge_attr that correspond to the distance magnitude. Defaults to -1.
        indices_target_vector (Iterable[int], optional): Indices of the graph.y that correspond to the vector components. Defaults to (0, 1).

    Returns:
        Tensor: Value of the divergent.

    """
    indices_distances = (0, 1)
    indice_distance_mag = -1
    indices_target_vector = (0, 1)

    d_edge_attr_norm_d_edge_attr = _d_edge_attr_norm_d_edge_attr(graph, edge_normalize)
    d_y_d_y_norm = _d_y_d_y_norm(y, target_normalize)

    div_value = torch.tensor(0.0)
    for i_dx, i_y in zip(indices_distances, indices_target_vector):
        d_y_norm_d_edge_attr_norm = _d_y_norm_d_edge_attr_norm(y, graph, i, i_y)
        d_edge_attr_d_pos = _d_edge_attr_d_pos(graph, i, i_dx, indice_distance_mag)

        s = _mul_sparse_dense(d_edge_attr_d_pos, d_edge_attr_norm_d_edge_attr)
        s = _mul_sparse_dense(s, d_y_norm_d_edge_attr_norm)
        s *= d_y_d_y_norm[i_y]
        div_value += s.coalesce().values().sum()

    return div_value
