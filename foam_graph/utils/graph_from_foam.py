import numpy as np
import torch
from torch.nn.functional import one_hot
from torch_geometric_temporal.signal import (
    StaticGraphTemporalSignal,
    DynamicGraphTemporalSignal,
)
from torch_geometric.utils import to_undirected

from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.Basics.DataStructures import Field, FixedLength

import os.path

from typing import Iterable, Optional, Union, Callable, Tuple
from collections.abc import Mapping

from operator import itemgetter
from functools import reduce
import warnings

from PyFoam.RunDictionary.ParsedParameterFile import (
    ParsedBoundaryDict,
    ParsedParameterFile,
)


from pathlib import Path

IGNORED_PATCH_TYPES = ["empty", "wedge"]


def parse_boundary_field(fn):
    return ParsedParameterFile(fn)["boundaryField"]


def parse_field_all(fn):
    f = ParsedParameterFile(fn)
    return f["internalField"], f["boundaryField"]


class FoamMesh(object):
    def __init__(self, path, read_boundaries=False):
        self.path = Path(path) / Path("polyMesh")

        self.boundary = ParsedBoundaryDict(self.path / Path("boundary")).content
        self.faces = ParsedParameterFile(
            self.path / Path("faces"), listDictWithHeader=True
        ).content
        self.neighbour = ParsedParameterFile(
            self.path / Path("neighbour"), listDictWithHeader=True
        ).content
        self.owner = ParsedParameterFile(
            self.path / Path("owner"), listDictWithHeader=True
        ).content

        self.read_boundaries = read_boundaries
        self.num_face = len(self.owner)
        self.num_inner_face = len(self.neighbour)
        self.num_cell = max(self.owner)
        self.boundary_face_centres = None

    def read_cell_centres(self, fn):
        self.cell_centres = ParsedParameterFile(fn)["internalField"]
        if self.read_boundaries:
            self.boundary_face_centres = parse_boundary_field(fn)

    def boundary_cells(self, bd):
        b = self.boundary[bd]
        return self.owner[b["startFace"] : b["startFace"] + b["nFaces"]]


def _read_mesh(
    case_name: str,
    read_boundaries: bool = True,
    time: float = 0,
    dynamic_mesh: bool = False,
    read_centres: bool = True,
) -> FoamMesh:
    mesh_path = f"{case_name}/constant"
    if dynamic_mesh and os.path.isdir(f"{case_name}/{time}/polyMesh"):
        mesh_path = f"{case_name}/{time}"
    mesh = FoamMesh(mesh_path, read_boundaries=True)
    if read_centres:
        mesh.read_cell_centres(f"{case_name}/{time}/C")
    if read_boundaries:
        mesh.boundary_face_centres = parse_boundary_field(f"{case_name}/{time}/C")
    return mesh


def _internal_connectivity(mesh: FoamMesh) -> np.ndarray:
    return np.array(
        [mesh.owner[0 : mesh.num_inner_face], mesh.neighbour[0 : mesh.num_inner_face]]
    )


def _boundary_connectivity(mesh: FoamMesh) -> np.ndarray:
    bd_orig = []
    bd_dest = []
    n_empty = 0
    offsets = {}
    for bd in mesh.boundary.keys():
        b = mesh.boundary[bd]
        if b["type"] in IGNORED_PATCH_TYPES:
            n_empty += b["nFaces"]
        else:
            bd_offset = -mesh.num_inner_face + mesh.num_cell + 1 - n_empty
            offsets[bd] = bd_offset

            bd_orig.append(
                np.array(range(b["startFace"], b["startFace"] + b["nFaces"]))
                + bd_offset
            )
            bd_dest.append(np.fromiter(mesh.boundary_cells(bd), int))

    bd_orig = np.hstack(bd_orig)
    bd_dest = np.hstack(bd_dest)

    return np.array([bd_orig, bd_dest])


def _boundary_positions(mesh: FoamMesh) -> np.ndarray:
    pos_f = []
    for bd in mesh.boundary.keys():
        face_centres = mesh.boundary_face_centres[bd]
        fc = face_centres.get("value")
        if fc is not None:
            pos_f.append(fc)

    return np.vstack(pos_f)


def _mesh_to_edges_and_nodes(
    mesh: FoamMesh, read_boundaries: bool = True
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
    field: Optional[Field], n_vals: int, n_comps: int
) -> np.ndarray:
    if field is None:
        return np.zeros((n_vals, 1)) if n_comps == 1 else np.zeros((n_vals, n_comps))

    if field.isUniform():
        return np.tile(np.array(field.val), (n_vals, 1))
    else:
        field_out = np.array(field.val)
        if field_out.ndim == 1:
            field_out = np.expand_dims(field, axis=1)
        return field_out


def _number_of_components(field: Field, mesh: FoamMesh) -> int:
    if field.isUniform():
        if isinstance(field.val, FixedLength):
            return len(field.val)
        else:
            return 1
    elif isinstance(field.val, list):
        if isinstance(field.val[0], FixedLength):
            return len(field.val[0])
        else:
            return 1


def _read_field(
    case_name: str,
    mesh: FoamMesh,
    field_name: str,
    read_boundaries: bool = True,
    time: float = 0,
) -> np.ndarray:
    field, field_boundary = parse_field_all(f"{case_name}/{time}/{field_name}")
    n_comps = _number_of_components(field, mesh)
    field = _expand_field_shape(field, len(mesh.cell_centres), n_comps)
    if read_boundaries:
        for bd_name, bd in mesh.boundary.items():
            if bd["type"] in IGNORED_PATCH_TYPES:
                continue

            field_value = field_boundary[bd_name].get("value")

            if field_value is None:
                owners = mesh.owner[bd["startFace"]:bd["startFace"] + bd["nFaces"]]
                field_bd = field[owners]
            else:
                field_bd = _expand_field_shape(
                    field_value, bd["nFaces"], n_comps,
                )
            field = np.vstack([field, field_bd])
    return field


def _boundary_encoding_name_as_one_hot(
    bd: Optional[str], boundaries: Mapping
) -> np.ndarray:
    """Encodes a boundary as a one-hot tensor. 
    Class 0 is used for internal cells, and the other classes are the boundary names, 
    in the order present in "boundaries".

    Args:
        bd (Optional[str]): boundary name, or None for internal cell.
        boundaries (Mapping): dictionary of boundaries.

    Returns:
        np.ndarray: encoded boundary.
    """
    valid_bds = [
        b for b in boundaries.keys() if boundaries[b]["type"] not in IGNORED_PATCH_TYPES
    ]
    category = 0 if bd is None else valid_bds.index(bd) + 1
    return one_hot(torch.tensor(category), len(valid_bds) + 1)


def _deep_get(dictionary: Mapping, keys: Iterable[str]) -> float:
    return reduce(lambda d, key: d.get(key) if d else None, keys, dictionary)


def read_foam(
    case_path: str,
    field_names: Iterable[str],
    read_boundaries: bool = True,
    times: Optional[Iterable[float]] = None,
    times_indices: Optional[Iterable[Union[slice, int]]] = None,
    boundary_encoding: Callable[
        [Optional[str], Mapping], np.ndarray
    ] = _boundary_encoding_name_as_one_hot,
    global_attrs: Optional[Tuple[str, str, Iterable[str]]] = None,
) -> Union[StaticGraphTemporalSignal, DynamicGraphTemporalSignal]:
    """Reads an OpenFOAM case as a PyTorch Geometric graph.

    Args:
        case_path (str): Path to the folder containg an OpenFOAM case
        field_names (Iterable[str]): List of field names extracted from the case
        read_boundaries (bool, optional): Flag for reading boundary patch faces as graph nodes. Also adds a binary mask to the graph for those faces. Defaults to True.
        times (Iterable[float], optional): List of times to be read. Defaults to None, which reads all. Overrides 'times_indices'.
        times_indices (Iterable[Union[slice, int]], optional): List of time indices to be read. Defaults to None, which reads all.
        boundary_encoding (Callable[[Optional[str], Mapping], np.ndarray], optional):
            Callable that takes a boundary name (or None, for internal cells)
            and a dictionary of all boundaries and returns a tensor for
            identifying that boundary/region. Defaults to
            _boundary_encoding_name_as_one_hot, which encodes each boundary
            as a category with a one-hot tensor. API is subject to change.
        global_attrs (Optional[Tuple[str, str, Iterable[str]]], optional): 
            List of global properties to be extracted from the case.
            Each property is defined by an attribute name, a relative file path
            and a list of nested dictionary indices to access the value. 
            Defaults to None. API is subject to future change.

    Raises:
        ValueError: no times selected.

    Returns:
        Union[StaticGraphTemporalSignal, DynamicGraphTemporalSignal]: PyTorch Geometric Temporal data iterator. Fields are stored as extra attributes, with the same name as in the case.
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
        if isinstance(selected_times, str) :
            selected_times = [selected_times]
    else:
        selected_times = case.times

    if not selected_times:
        raise ValueError("No times have been selected.")

    fields = {f: [] for f in field_names}
    if global_attrs is not None:
        for attr, _, _ in global_attrs:
            fields[attr] = []
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
            mesh = _read_mesh(str(case.name), read_boundaries, time, dynamic_mesh)
            edge_index, pos = _mesh_to_edges_and_nodes(mesh, read_boundaries)
            mesh_read = True

            if read_boundaries:
                flag_internal = boundary_encoding(None, mesh.boundary)
                boundary_flags = np.tile(flag_internal, (len(mesh.cell_centres), 1))
                for bd in mesh.boundary.keys():
                    b = mesh.boundary[bd]
                    if b["type"] not in IGNORED_PATCH_TYPES:
                        flag_bd = boundary_encoding(bd, mesh.boundary)
                        flag_bd = np.tile(flag_bd, (b["nFaces"], 1))
                        boundary_flags = np.vstack((boundary_flags, flag_bd))

            if dynamic_mesh:
                edge_index_list.append(edge_index)
                pos_list.append(pos)
                if read_boundaries:
                    boundary_flags_list.append(boundary_flags)

        for f in field_names:
            field = _read_field(str(case.name), mesh, f, read_boundaries, time)
            fields[f].append(field)

        if global_attrs is not None:
            for attr, path, keys in global_attrs:
                f = ParsedParameterFile(os.path.join(case_path, path)).content
                v = _deep_get(f, keys)
                fields[attr].append(np.array(v))

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
