# graph_from_foam.py

import os
import re
from pathlib import Path
from operator import itemgetter
from typing import Iterable, Optional, Union, Callable, Tuple
from collections.abc import Mapping
import warnings
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch_geometric.utils import to_undirected
from torch_geometric_temporal.signal import (
    StaticGraphTemporalSignal,
    DynamicGraphTemporalSignal,
)
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.Basics.DataStructures import Field, FixedLength
from PyFoam.RunDictionary.ParsedParameterFile import (
    ParsedBoundaryDict,
    ParsedParameterFile,
)

IGNORED_PATCH_TYPES = ["empty", "wedge"]

# ------------------------------ Binary parsing helpers ------------------------------

_BIN_WS = b"[ \t\r\n]*"

def _read_bytes(path: Union[str, Path]) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _get_header_block(data: bytes) -> bytes:
    m = re.search(rb"FoamFile" + _BIN_WS + rb"\{.*?\}", data, re.S)
    if not m:
        raise ValueError("FoamFile header not found.")
    return m.group(0)

def _header_is_binary(header: bytes) -> bool:
    return re.search(rb"format" + _BIN_WS + rb"binary\s*;", header) is not None

def _parse_arch(header: bytes) -> Tuple[str, int]:
    """Return ('<' or '>', scalar_bytes)."""
    m = re.search(rb'arch' + _BIN_WS + rb'("?)([^";]+)\1\s*;', header)
    endian_char = "<"
    scalar_size = 8
    if m:
        arch = m.group(2).decode("utf-8", "ignore")
        if "MSB" in arch:
            endian_char = ">"
        if ("scalar=32" in arch) or ("scalar 32" in arch):
            scalar_size = 4
        elif ("scalar=64" in arch) or ("scalar 64" in arch):
            scalar_size = 8
    return endian_char, scalar_size

def _components_for_list_type(list_type: str) -> int:
    lt = list_type.lower()
    return {
        "scalar": 1,
        "vector": 3,
        "tensor": 9,
        "symmtensor": 6,
        "sphericaltensor": 1,
    }.get(lt, 1)

def _parse_uniform_value(token: bytes, comps: int) -> np.ndarray:
    tok = token.strip()
    if tok.startswith(b"(") and tok.endswith(b")"):
        nums = [float(x) for x in tok[1:-1].split()]
    else:
        nums = [float(tok)]
    if comps == 1:
        return np.array(nums, dtype=float).reshape(1, 1)
    out = np.zeros((1, comps), dtype=float)
    out[0, : min(comps, len(nums))] = nums[:comps]
    return out

def _extract_binary_list_from_raw(raw: bytes, offset: int, n_items: int, item_comps: int, dtype: np.dtype) -> Tuple[np.ndarray, int]:
    count = n_items * item_comps
    nbytes = count * dtype.itemsize
    if offset + nbytes > len(raw):
        raise ValueError(f"Binary block exceeds file size at offset {offset} (need {nbytes} bytes).")
    arr = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
    return arr.reshape((n_items, item_comps)), offset + nbytes

def _find_boundaryfield_start(raw: bytes) -> int:
    m = re.search(rb"boundaryField" + _BIN_WS + rb"\{", raw)
    return m.end() if m else 0

def _locate_patch_regions(raw: bytes, boundary_names: Iterable[str], start_idx: int) -> Mapping[str, Tuple[int, int]]:
    """Find 'patchName {' positions after boundaryField. Define [start, next_start) regions for regex, but never slice the binary block."""
    indices = []
    for nm in boundary_names:
        pat = rb"(?<!\w)" + re.escape(nm.encode()) + _BIN_WS + rb"\{"
        m = re.search(pat, raw[start_idx:], re.S)
        if m:
            indices.append((start_idx + m.start(), nm))
    indices.sort()
    regions = {}
    for i, (s, nm) in enumerate(indices):
        e = indices[i + 1][0] if i + 1 < len(indices) else len(raw)
        regions[nm] = (s, e)
    return regions

def _as_numpy(obj, default_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Best-effort convert PyFoam Field/BinaryList/list to np.ndarray."""
    if isinstance(obj, np.ndarray):
        return obj
    # Field?
    if hasattr(obj, "isUniform") and hasattr(obj, "val"):
        v = obj.val
        try:
            arr = np.array(v)
        except Exception:
            arr = None
        if obj.isUniform():
            arr = np.array(v)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
        return arr
    # Has .val (BinaryList or similar)
    if hasattr(obj, "val"):
        try:
            return np.array(obj.val)
        except Exception:
            return None
    # Plain python list?
    try:
        return np.array(obj)
    except Exception:
        pass
    # Fallback default
    if default_shape is not None:
        return np.zeros(default_shape, dtype=float)
    return None

def _parse_field_binary_file(raw: bytes,
                             path: str,
                             n_cells: int,
                             mesh_boundaries: Mapping,
                             read_boundaries: bool = True) -> Tuple[np.ndarray, Mapping]:
    """Binary-safe parser for vol*Field with 'format binary;'."""
    header = _get_header_block(raw)
    if not _header_is_binary(header):
        raise RuntimeError("Not a binary Foam field; use text parser.")
    endian_char, scalar_size = _parse_arch(header)
    dtype = np.dtype(("<f4" if endian_char == "<" else ">f4") if scalar_size == 4 else ("<f8" if endian_char == "<" else ">f8"))

    # internalField
    m_uni = re.search(rb"internalField" + _BIN_WS + rb"uniform" + _BIN_WS + rb"([^;]+);", raw)
    if m_uni:
        token = m_uni.group(1)
        comps_guess = 3 if token.strip().startswith(b"(") else 1
        internal = np.repeat(_parse_uniform_value(token, comps_guess), repeats=n_cells, axis=0)
    else:
        m_non = re.search(
            rb"internalField" + _BIN_WS + rb"nonuniform" + _BIN_WS + rb"List<([^>]+)>" +
            _BIN_WS + rb"(\d+)" + _BIN_WS + rb"\(" + _BIN_WS, raw, re.S
        )
        if not m_non:
            raise ValueError(f"Could not find internalField in {path}")
        list_type = m_non.group(1).decode().strip()
        n_items = int(m_non.group(2))
        comps = _components_for_list_type(list_type)
        start = m_non.end()
        internal, _ = _extract_binary_list_from_raw(raw, start, n_items, comps, dtype)

    # boundaryField
    boundary_out = {}
    if read_boundaries:
        bf_start = _find_boundaryfield_start(raw)
        valid_names = [nm for nm, bd in mesh_boundaries.items() if bd["type"] not in IGNORED_PATCH_TYPES]
        regions = _locate_patch_regions(raw, valid_names, bf_start)

        for nm in valid_names:
            boundary_out[nm] = {"value": None}
            if nm not in regions:
                continue
            s, e = regions[nm]
            mu = re.search(rb"value" + _BIN_WS + rb"uniform" + _BIN_WS + rb"([^;]+);", raw[s:e], re.S)
            if mu:
                token = mu.group(1)
                comps = internal.shape[1]
                uv = _parse_uniform_value(token, comps)
                nFaces = mesh_boundaries[nm]["nFaces"]
                boundary_out[nm]["value"] = np.repeat(uv, repeats=nFaces, axis=0)
                continue
            mn = re.search(
                rb"value" + _BIN_WS + rb"nonuniform" + _BIN_WS + rb"List<([^>]+)>" +
                _BIN_WS + rb"(\d+)" + _BIN_WS + rb"\(" + _BIN_WS,
                raw[s:e], re.S
            )
            if mn:
                list_type = mn.group(1).decode().strip()
                n_items = int(mn.group(2))
                comps = _components_for_list_type(list_type)
                abs_start = s + mn.end()
                arr, _ = _extract_binary_list_from_raw(raw, abs_start, n_items, comps, dtype)
                boundary_out[nm]["value"] = arr

    return internal, boundary_out

def _read_field_binary_as_array(field_path: str, mesh, read_boundaries: bool = True) -> np.ndarray:
    raw = _read_bytes(field_path)
    internal, bdmap = _parse_field_binary_file(
        raw, field_path, n_cells=mesh.num_cell + 1, mesh_boundaries=mesh.boundary, read_boundaries=read_boundaries
    )
    out = internal
    if read_boundaries:
        for bd_name, bd in mesh.boundary.items():
            if bd["type"] in IGNORED_PATCH_TYPES:
                continue
            v = bdmap.get(bd_name, {}).get("value")
            if v is None:
                owners = mesh.owner[bd["startFace"]: bd["startFace"] + bd["nFaces"]]
                v = internal[owners]
            out = np.vstack([out, v])
    return out

def parse_boundary_field(fn):
    return ParsedParameterFile(fn)["boundaryField"]

def parse_field_all(fn):
    f = ParsedParameterFile(fn)
    return f["internalField"], f["boundaryField"]

class FoamMesh(object):
    def __init__(self, path, read_boundaries=False):
        self.path = Path(path) / Path("polyMesh")
        self.boundary = ParsedBoundaryDict(self.path / Path("boundary")).content
        self.faces = ParsedParameterFile(self.path / Path("faces"), listDictWithHeader=True).content
        self.neighbour = ParsedParameterFile(self.path / Path("neighbour"), listDictWithHeader=True).content
        self.owner = ParsedParameterFile(self.path / Path("owner"), listDictWithHeader=True).content

        self.read_boundaries = read_boundaries
        self.num_face = len(self.owner)
        self.num_inner_face = len(self.neighbour)
        self.num_cell = max(self.owner)
        self.boundary_face_centres = None
        self.cell_centres = None

    def read_cell_centres(self, fn, binary=False):
        """Read C (volVectorField). Always coerce to numpy [n_cells,3]."""
        n_cells = self.num_cell + 1
        if binary:
            raw = _read_bytes(fn)
            header = _get_header_block(raw)
            if _header_is_binary(header):
                internal, bd = _parse_field_binary_file(
                    raw, fn, n_cells=n_cells, mesh_boundaries=self.boundary, read_boundaries=self.read_boundaries
                )
                self.cell_centres = internal  # numpy
                if self.read_boundaries:
                    self.boundary_face_centres = {k: {"value": v["value"]} for k, v in bd.items()}
                return
        # ASCII fallback (coerce)
        C_field = ParsedParameterFile(fn)["internalField"]
        if hasattr(C_field, "isUniform") and C_field.isUniform():
            v = np.array(C_field.val, dtype=float)
            self.cell_centres = np.tile(v, (n_cells, 1))
        else:
            arr = _as_numpy(C_field)
            if arr is None:
                self.cell_centres = np.zeros((n_cells, 3), dtype=float)
            else:
                # Ensure [n_cells, 3]
                if arr.ndim == 1 and arr.size == n_cells * 3:
                    arr = arr.reshape(n_cells, 3)
                self.cell_centres = arr
        if self.read_boundaries:
            bd = parse_boundary_field(fn)
            # coerce each patch value to numpy [nFaces,3] if present
            out = {}
            for name, d in self.boundary.items():
                if name in bd:
                    v = bd[name].get("value")
                    if v is None:
                        out[name] = {"value": None}
                    else:
                        a = _as_numpy(v)
                        if a is not None and a.ndim == 1 and a.size == d["nFaces"] * 3:
                            a = a.reshape(d["nFaces"], 3)
                        out[name] = {"value": a}
            self.boundary_face_centres = out

    def boundary_cells(self, bd):
        b = self.boundary[bd]
        return self.owner[b["startFace"] : b["startFace"] + b["nFaces"]]

def _read_mesh(case_name: str, read_boundaries: bool = True, time: float = 0, dynamic_mesh: bool = False, binary: bool = False) -> FoamMesh:
    mesh_path = f"{case_name}/constant"
    if dynamic_mesh and os.path.isdir(f"{case_name}/{time}/polyMesh"):
        mesh_path = f"{case_name}/{time}"
    mesh = FoamMesh(mesh_path, read_boundaries=True)
    mesh.read_cell_centres(f"{case_name}/{time}/C", binary=binary)
    if read_boundaries and mesh.boundary_face_centres is None:
        mesh.boundary_face_centres = parse_boundary_field(f"{case_name}/{time}/C")
    return mesh

def _internal_connectivity(mesh: FoamMesh) -> np.ndarray:
    return np.array([mesh.owner[0:mesh.num_inner_face], mesh.neighbour[0:mesh.num_inner_face]])

def _boundary_connectivity(mesh: FoamMesh) -> np.ndarray:
    bd_orig, bd_dest = [], []
    n_empty = 0
    for bd in mesh.boundary.keys():
        b = mesh.boundary[bd]
        if b["type"] in IGNORED_PATCH_TYPES:
            n_empty += b["nFaces"]
        else:
            bd_offset = -mesh.num_inner_face + mesh.num_cell + 1 - n_empty
            bd_orig.append(np.arange(b["startFace"], b["startFace"] + b["nFaces"]) + bd_offset)
            bd_dest.append(np.fromiter(mesh.boundary_cells(bd), int))
    if len(bd_orig) == 0:
        return np.empty((2, 0), dtype=int)
    return np.array([np.hstack(bd_orig), np.hstack(bd_dest)])

def _boundary_positions(mesh: FoamMesh) -> np.ndarray:
    pos_f = []
    if mesh.boundary_face_centres is None:
        return np.empty((0, 3), dtype=float)
    for bd in mesh.boundary.keys():
        d = mesh.boundary_face_centres.get(bd, {})
        fc = d.get("value") if isinstance(d, dict) else None
        if fc is not None:
            a = _as_numpy(fc)
            if a is not None:
                if a.ndim == 1 and a.size % 3 == 0:
                    a = a.reshape(-1, 3)
                pos_f.append(a)
    return np.vstack(pos_f) if pos_f else np.empty((0, 3), dtype=float)

def _mesh_to_edges_and_nodes(mesh: FoamMesh, read_boundaries: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = _internal_connectivity(mesh)
    pos = mesh.cell_centres
    if not isinstance(pos, np.ndarray):
        pos = _as_numpy(pos, default_shape=(mesh.num_cell + 1, 3))
    if read_boundaries:
        edge_index_bd = _boundary_connectivity(mesh)
        if edge_index_bd.size > 0:
            edge_index = np.hstack([edge_index, edge_index_bd])
        pos_bd = _boundary_positions(mesh)
        if pos_bd.size > 0:
            pos = np.vstack([pos, pos_bd])
    edge_index = torch.as_tensor(edge_index)
    edge_index = to_undirected(edge_index).detach().numpy()
    return (edge_index, pos)

def _expand_field_shape(field: Optional[Field], n_vals: int, n_comps: int) -> np.ndarray:
    if field is None:
        return np.zeros((n_vals, n_comps))
    if hasattr(field, "isUniform") and field.isUniform():
        return np.tile(np.array(field.val), (n_vals, 1))
    field_out = _as_numpy(field)
    if field_out is None:
        return np.zeros((n_vals, n_comps))
    if field_out.ndim == 1:
        field_out = field_out.reshape(-1, 1 if n_comps == 1 else n_comps)
    return field_out

def _number_of_components(field: Field, mesh: FoamMesh) -> int:
    if hasattr(field, "isUniform") and field.isUniform():
        return len(field.val) if isinstance(field.val, FixedLength) else 1
    elif isinstance(field.val, list):
        return len(field.val[0]) if isinstance(field.val[0], FixedLength) else 1
    # Default fallback
    return 1

def _read_field(case_name: str, mesh: FoamMesh, field_name: str, read_boundaries: bool = True, time: float = 0, binary: bool = False) -> np.ndarray:
    fp = f"{case_name}/{time}/{field_name}"
    n_cells = mesh.num_cell + 1
    if binary:
        try:
            raw = _read_bytes(fp)
            if _header_is_binary(_get_header_block(raw)):
                return _read_field_binary_as_array(fp, mesh, read_boundaries)
        except Exception:
            pass  # fallback to ASCII
    # ASCII via PyFoam
    field, field_boundary = parse_field_all(fp)
    n_comps = _number_of_components(field, mesh)
    field_arr = _expand_field_shape(field, n_cells, n_comps)
    if read_boundaries:
        for bd_name, bd in mesh.boundary.items():
            if bd["type"] in IGNORED_PATCH_TYPES:
                continue
            field_value = field_boundary[bd_name].get("value")
            if field_value is None:
                owners = mesh.owner[bd["startFace"]: bd["startFace"] + bd["nFaces"]]
                field_bd = field_arr[owners]
            else:
                field_bd = _expand_field_shape(field_value, bd["nFaces"], n_comps)
            field_arr = np.vstack([field_arr, field_bd])
    return field_arr

def _boundary_encoding_name_as_one_hot(bd: Optional[str], boundaries: Mapping) -> np.ndarray:
    valid_bds = [b for b in boundaries.keys() if boundaries[b]["type"] not in IGNORED_PATCH_TYPES]
    category = 0 if bd is None else valid_bds.index(bd) + 1
    return one_hot(torch.tensor(category), len(valid_bds) + 1)

def read_foam(
    case_path: str,
    field_names: Iterable[str],
    read_boundaries: bool = True,
    times: Optional[Iterable[float]] = None,
    times_indices: Optional[Iterable[Union[slice, int]]] = None,
    boundary_encoding: Callable[[Optional[str], Mapping], np.ndarray] = _boundary_encoding_name_as_one_hot,
    binary: bool = False,
) -> Union[StaticGraphTemporalSignal, DynamicGraphTemporalSignal]:
    """
    Reads an OpenFOAM case as a PyTorch Geometric graph.

    Args:
        case_path (str): Path to the folder containg an OpenFOAM case
        field_names (Iterable[str]): List of field names extracted from the case
        read_boundaries (bool, optional): Flag for reading boundary patch faces as graph nodes. Also adds a binary mask to the graph for those faces. Defaults to True.
        times (Iterable[float], optional): List of times to be read. Defaults to None, which reads all. Overrides 'times_indices'.
        times_indices (Iterable[Union[slice, int]], optional): List of time indices to be read. Defaults to None, which reads all.
        boundary_encoding (Callable[[Optional[str], Mapping], np.ndarray], optional): Function that encodes a boundary patch name into a one-hot vector.
        binary (bool, optional): Whether to read fields in binary format when possible.

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
            warnings.warn("Pass only one of 'times' or 'times_indices'. Ignoring 'times_indices'.")
    elif times_indices is not None:
        sel = itemgetter(*times_indices)(case.times)
        selected_times = [sel] if isinstance(sel, str) else list(sel)
    else:
        selected_times = list(case.times)

    # Always drop "0"
    selected_times = [t for t in selected_times if t != "0"]
    if not selected_times:
        raise ValueError("No times have been selected.")

    fields = {f: [] for f in field_names}
    fields["time"] = []

    dynamic_mesh = any(os.path.isdir(f"{case_path}/{time}/polyMesh") for time in selected_times)
    edge_index_list, pos_list, boundary_flags_list = [], [], []

    mesh_read = False
    for time in selected_times:
        fields["time"].append(np.full(1, float(time)))
        if (not mesh_read) or dynamic_mesh:
            mesh = _read_mesh(str(case.name), read_boundaries, time, dynamic_mesh, binary=binary)
            edge_index, pos = _mesh_to_edges_and_nodes(mesh, read_boundaries)
            mesh_read = True

            if read_boundaries:
                flag_internal = boundary_encoding(None, mesh.boundary)
                n_cells = mesh.num_cell + 1
                boundary_flags = np.tile(flag_internal, (n_cells, 1))
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
            field_arr = _read_field(str(case.name), mesh, f, read_boundaries, time, binary=binary)
            fields[f].append(field_arr)

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
