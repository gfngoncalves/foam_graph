import openfoamparser as op
import numpy as np
import torch
from torch_geometric_temporal.signal import (
    StaticGraphTemporalSignal,
    DynamicGraphTemporalSignal,
)
from torch_geometric.utils import to_undirected

from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory

import os.path

from typing import Iterable, Optional, Union, Callable, Tuple
from collections.abc import Mapping

from operator import itemgetter
import warnings


class _FoamMeshExtended(op.FoamMesh):
    """ FoamMesh class """

    def __init__(self, path):
        self.path = os.path.join(path, "polyMesh/")
        self._parse_mesh_data(self.path)
        self.num_point = len(self.points)
        self.num_face = len(self.owner)
        self.num_inner_face = len(self.neighbour)
        self.num_cell = max(self.owner)
        self._set_boundary_faces()
        self._construct_cells()
        self.cell_centres = None
        self.cell_volumes = None
        self.face_areas = None


def _read_mesh(
    case_name: str,
    read_boundaries: bool = True,
    time: float = 0,
    dynamic_mesh: bool = False,
) -> _FoamMeshExtended:
    mesh_path = f"{case_name}/constant"
    if dynamic_mesh and os.path.isdir(f"{case_name}/{time}/polyMesh"):
        mesh_path = f"{case_name}/{time}"
    mesh = _FoamMeshExtended(mesh_path)
    mesh.read_cell_centres(f"{case_name}/{time}/C")
    if read_boundaries:
        mesh.boundary_face_centres = op.parse_boundary_field(f"{case_name}/{time}/C")
    return mesh


def _internal_connectivity(mesh: op.FoamMesh) -> np.ndarray:
    return np.array(
        [mesh.owner[0 : mesh.num_inner_face], mesh.neighbour[0 : mesh.num_inner_face]]
    )


def _boundary_connectivity(mesh: op.FoamMesh) -> np.ndarray:
    bd_orig = []
    bd_dest = []
    n_empty = 0
    offsets = {}
    for bd in mesh.boundary.keys():
        b = mesh.boundary[bd]
        if b.type == b"empty":
            n_empty += b.num
        else:
            bd_offset = -mesh.num_inner_face + mesh.num_cell + 1 - n_empty
            offsets[bd] = bd_offset

            bd_orig.append(np.array(range(b.start, b.start + b.num)) + bd_offset)
            bd_dest.append(np.fromiter(mesh.boundary_cells(bd), int))

    bd_orig = np.hstack(bd_orig)
    bd_dest = np.hstack(bd_dest)

    return np.array([bd_orig, bd_dest])


def _boundary_positions(mesh: op.FoamMesh) -> np.ndarray:
    pos_f = []
    for bd in mesh.boundary.keys():
        face_centres = mesh.boundary_face_centres[bd]
        fc = face_centres.get(b"value")
        if fc is not None:
            pos_f.append(fc)

    return np.vstack(pos_f)


def _mesh_to_edges_and_nodes(
    mesh: op.FoamMesh, read_boundaries: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = _internal_connectivity(mesh)
    pos = mesh.cell_centres

    if read_boundaries:
        edge_index_bd = _boundary_connectivity(mesh)
        edge_index = np.hstack([edge_index, edge_index_bd])

        pos_bd = _boundary_positions(mesh)
        pos = np.vstack([pos, pos_bd])

    edge_index = torch.as_tensor(edge_index)
    edge_index = to_undirected(edge_index).detach().numpy()

    return (edge_index, pos)


def _expand_field_shape(
    field: Optional[Union[float, np.ndarray]], n_vals: int, n_comps: int
) -> np.ndarray:
    if field is None:
        field = np.zeros((n_vals)) if n_comps == 1 else np.zeros((n_vals, n_comps))
    if not isinstance(field, np.ndarray):
        field = np.array([field])
    if len(field) != n_vals:
        field = np.tile(field, (n_vals, 1))
    if field.ndim == 1:
        field = np.expand_dims(field, axis=1)
    return field


def _number_of_components(field: Union[float, np.ndarray], mesh: op.FoamMesh) -> int:
    if not isinstance(field, np.ndarray):
        return 1
    elif len(field) != len(mesh.cell_centres):
        return len(field)
    elif field.ndim == 1:
        return 1
    else:
        return field.shape[1]


def _get_value_from_field_name(field_boundary: Mapping, bd: str):
    if bd in field_boundary:
        return field_boundary[bd]
    for b in field_boundary.keys():
        if b.decode("utf-8")[0] == '"' and b.decode("utf-8")[-1] == '"':
            bds = b.decode("utf-8")[2:-2].split("|")
            bds = [bi.encode() for bi in bds]
            if bd in bds:
                return field_boundary[b]
    return None


def _read_field(
    case_name: str,
    mesh: op.FoamMesh,
    field_name: str,
    read_boundaries: bool = True,
    time: float = 0,
) -> torch.Tensor:
    field, field_boundary = op.parse_field_all(f"{case_name}/{time}/{field_name}")
    n_comps = _number_of_components(field, mesh)
    field = _expand_field_shape(field, len(mesh.cell_centres), n_comps)
    if read_boundaries:
        for bd in mesh.boundary.keys():
            if mesh.boundary[bd].type == b"empty":
                continue

            field_value = _get_value_from_field_name(field_boundary, bd).get(b"value")
            field_bd = _expand_field_shape(field_value, mesh.boundary[bd].num, n_comps,)
            field = np.vstack([field, field_bd])
    return field


def _boundary_encoding(bd: Optional[str], boundaries: Mapping) -> np.ndarray:
    return np.array([1]) if bd is not None else np.array([0])


def read_foam(
    case_path: str,
    field_names: Iterable[str],
    read_boundaries: bool = True,
    times: Optional[Iterable[float]] = None,
    times_indices: Optional[Iterable[Union[slice, int]]] = None,
    boundary_encoding: Callable[
        [Optional[str], Mapping], np.ndarray
    ] = _boundary_encoding,
) -> StaticGraphTemporalSignal:
    """Reads an OpenFOAM case as a PyTorch Geometric graph.

    Args:
        case_path (str): Path to the folder containg an OpenFOAM case
        field_names (Iterable[str]): List of field names extracted from the case
        read_boundaries (bool, optional): Flag for reading boundary patch faces as graph nodes. Also adds a binary mask to the graph for those faces. Defaults to True.
        times (Iterable[float], optional): List of times to be read. Defaults to None, which reads all. Overrides 'times_indices'.
        times_indices (Iterable[Union[slice, int]], optional): List of time indices to be read. Defaults to None, which reads all.

    Raises:
        ValueError: no times selected.

    Returns:
        StaticGraphTemporalSignal: PyTorch Geometric Temporal data iterator. Fields are stored as extra attributes, with the same name as in the case.
    """
    case = SolutionDirectory(case_path)
    
    if times is not None:
        selected_times = [str(t) for t in times if str(t) in case.times]
        if set(selected_times) != set(str(t) for t in times):
            warnings.warn("Not all times requested were found in the simulation.")
        if times_indices is not None:
            warnings.warn(
                "Pass only one of 'times' or 'times_indices'. Ignoring 'times_indices'."
            )
    elif times_indices is not None:
        selected_times = itemgetter(*times_indices)(case.times)
    else:
        selected_times = case.times

    if not selected_times:
        raise ValueError("No times have been selected.")

    fields = {f: [] for f in field_names}
    fields["time"] = []

    dynamic_mesh = any(
        True for time in selected_times if os.path.isdir(f"{case_path}/{time}/polyMesh")
    )
    edge_index_list = []
    pos_list = []
    boundary_flags_list = []

    mesh_read = False
    for time in selected_times:
        fields["time"].append(np.full(1, float(time)))
        if (not mesh_read) or dynamic_mesh:
            mesh = _read_mesh(case.name, read_boundaries, time, dynamic_mesh)
            edge_index, pos = _mesh_to_edges_and_nodes(mesh, read_boundaries)
            mesh_read = True

            if read_boundaries:
                flag_internal = boundary_encoding(None, mesh.boundary)
                boundary_flags = np.tile(flag_internal, (len(mesh.cell_centres), 1))
                for bd in mesh.boundary.keys():
                    b = mesh.boundary[bd]
                    if b.type != b"empty":
                        flag_bd = boundary_encoding(bd, mesh.boundary)
                        flag_bd = np.tile(flag_bd, (b.num, 1))
                        boundary_flags = np.vstack((boundary_flags, flag_bd))

            if dynamic_mesh:
                edge_index_list.append(edge_index)
                pos_list.append(pos)
                if read_boundaries:
                    boundary_flags_list.append(boundary_flags)

        for f in field_names:
            field = _read_field(case.name, mesh, f, read_boundaries, time)
            fields[f].append(field)

    if dynamic_mesh:
        if read_boundaries:
            fields["boundary"] = boundary_flags_list
        graph = DynamicGraphTemporalSignal(
            edge_indices=edge_index_list,
            edge_weights=[None for _ in selected_times],
            features=[None for _ in selected_times],
            targets=[None for _ in selected_times],
            pos=pos_list,
            **fields,
        )
    else:
        if read_boundaries:
            fields["boundary"] = [boundary_flags for _ in selected_times]
        graph = StaticGraphTemporalSignal(
            edge_index=edge_index,
            edge_weight=None,
            features=[None for _ in selected_times],
            targets=[None for _ in selected_times],
            pos=[pos for _ in selected_times],
            **fields,
        )
    return graph
