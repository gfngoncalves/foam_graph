import openfoamparser as op
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory

import os.path


def read_mesh(case_name, read_boundaries=True, time=0):
    mesh = op.FoamMesh(case_name)
    mesh.read_cell_centres(f"{case_name}/{time}/C")
    if read_boundaries:
        mesh.boundary_face_centres = op.parse_boundary_field(f"{case_name}/{time}/C")
    return mesh


def internal_connectivity(mesh):
    return np.array(
        [mesh.owner[0 : mesh.num_inner_face], mesh.neighbour[0 : mesh.num_inner_face]]
    )


def boundary_connectivity(mesh):
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


def boundary_positions(mesh):
    pos_f = []
    for bd in mesh.boundary.keys():
        face_centres = mesh.boundary_face_centres[bd]
        fc = face_centres.get(b"value")
        if fc is not None:
            pos_f.append(fc)

    return np.vstack(pos_f)


def mesh_to_edges_and_nodes(mesh, read_boundaries=True):
    edge_index = internal_connectivity(mesh)
    pos = mesh.cell_centres

    if read_boundaries:
        edge_index_bd = boundary_connectivity(mesh)
        edge_index = np.hstack([edge_index, edge_index_bd])

        pos_bd = boundary_positions(mesh)
        pos = np.vstack([pos, pos_bd])

    edge_index = torch.as_tensor(edge_index)
    edge_index = to_undirected(edge_index)

    pos = torch.as_tensor(pos, dtype=torch.float32)

    return (edge_index, pos)


def expand_field_shape(field, n_vals, n_comps):
    if field is None:
        field = np.zeros((n_vals)) if n_comps == 1 else np.zeros((n_vals, n_comps))
    if not isinstance(field, np.ndarray):
        field = np.array([field])
    if len(field) != n_vals:
        field = np.tile(field, (n_vals, 1))
    if field.ndim == 1:
        field = np.expand_dims(field, axis=1)
    return field


def number_of_components(field, mesh):
    if not isinstance(field, np.ndarray):
        return 1
    elif len(field) != len(mesh.cell_centres):
        return len(field)
    elif field.ndim == 1:
        return 1
    else:
        return field.shape[1]


def get_value_from_field_name(field_boundary, bd):
    if bd in field_boundary:
        return field_boundary[bd]
    for b in field_boundary.keys():
        if b.decode("utf-8")[0] == '"' and b.decode("utf-8")[-1] == '"':
            bds = b.decode("utf-8")[2:-2].split("|")
            bds = [bi.encode() for bi in bds]
            if bd in bds:
                return field_boundary[b]
    return None


def read_field(case_name, mesh, field_name, read_boundaries=True, time=0):
    field, field_boundary = op.parse_field_all(f"{case_name}/{time}/{field_name}")
    n_comps = number_of_components(field, mesh)
    field = expand_field_shape(field, len(mesh.cell_centres), n_comps)
    if read_boundaries:
        for bd in mesh.boundary.keys():
            if mesh.boundary[bd].type == b"empty":
                continue

            field_value = get_value_from_field_name(field_boundary, bd).get(
                b"value"
            )
            field_bd = expand_field_shape(
                field_value,
                mesh.boundary[bd].num,
                n_comps,
            )
            field = np.vstack([field, field_bd])
    return torch.as_tensor(field, dtype=torch.float32)


def read_case(case_path, field_names, read_boundaries=True, times="all"):
    """Reads an OpenFOAM case as a PyTorch Geometric graph.

    Args:
        case_path (str): Path to the folder containg an OpenFOAM case
        field_names (_type_): List of field names extracted from the case
        read_boundaries (bool, optional): Flag for reading boundary patch faces as graph nodes. Also adds a binary mask to the graph for those faces. Defaults to True.
        times (str/List, optional): List of times to be read, or strings "first_and_last" or "all" . Defaults to "all".

    Returns:
        Data: PyTorch Geometric graph. Fields are stored as attributes with shape [timesteps, cells, components]
    """
    case = SolutionDirectory(case_path)

    mesh = read_mesh(case.name, read_boundaries, case.getFirst())
    edge_index, pos = mesh_to_edges_and_nodes(mesh, read_boundaries)

    if isinstance(times, (list, np.ndarray)):
        selected_times = [str(t) for t in times]
    elif times == "first_and_last":
        selected_times = [case.getFirst(), case.getLast()]
    elif times == "all":
        selected_times = case.times
    else:
        raise ValueError("times must be an array, 'all' or 'first_and_last'")

    fields = {f: [] for f in field_names}
    fields["time"] = []
    for time in selected_times:
        fields["time"].append(float(time))
        for f in field_names:
            field = read_field(case.name, mesh, f, read_boundaries, time)
            fields[f].append(field)
    
    if read_boundaries:
        fields["boundary"] = torch.as_tensor(
            np.zeros((len(pos), 1)), dtype=torch.float32
        )
        fields["boundary"][mesh.num_cell + 1 :, :] = 1
    
    graph = Data(edge_index=edge_index, pos=pos, name=os.path.basename(case.name), **fields)
    return graph
