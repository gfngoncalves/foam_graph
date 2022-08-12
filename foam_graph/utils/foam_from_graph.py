import io
import numpy as np
from PyFoam.RunDictionary.ParsedParameterFile import (
    ParsedParameterFile,
    WriteParameterFile,
)
from torch_geometric.data import Data
import torch
from foam_graph.utils.graph_from_foam import _read_mesh, FoamMesh, IGNORED_PATCH_TYPES

from typing import Iterable, Optional

def _guess_data_type(n_components: int) -> str:
    data_sizes = {1: "scalar", 3: "vector", 6: "symmTensor", 9: "tensor"}
    if n_components in data_sizes:
        return data_sizes[n_components]
    else:
        raise ValueError(f"Unsupported data size: {n_components}")


def _generate_field(data_out: torch.Tensor, precision: int = 8) -> str:
    n_comps = data_out.size(-1)
    val_type = _guess_data_type(n_comps)

    newline = ")\n(" if n_comps > 1 else "\n"
    begtoken = "(" if n_comps > 1 else ""
    endtoken = ")" if n_comps > 1 else ""
    desc = f"nonuniform List<{val_type}> \n{len(data_out)}"
    s = io.BytesIO()
    np.savetxt(s, data_out.detach().numpy(), newline=newline, fmt=f"%.{precision}g")
    vals = (
        "\n(\n"
        + begtoken
        + s.getvalue().decode("utf-8")[: -len(newline)]
        + endtoken
        + "\n)\n"
    )

    return desc + vals


def _write_tensor(
    case_name: str,
    time_out: float,
    data_out: torch.Tensor,
    field_base: str,
    field_out: str,
    time_base: float = 0,
    mesh: Optional[FoamMesh] = None,
) -> None:
    if mesh is None:
        mesh = _read_mesh(
            case_name, read_boundaries=True, time=time_out, dynamic_mesh=True
        )

    n_comps = data_out.size(-1)
    val_type = _guess_data_type(n_comps)

    n_bds = sum(bd["nFaces"] for _, bd in mesh.boundary.items() if bd["type"] not in IGNORED_PATCH_TYPES)
    n_internal = mesh.num_cell + 1
    if len(data_out) != n_internal + n_bds:
        raise ValueError(
            f"Invalid dimensions: expected {n_internal + n_bds}, got {len(data_out)}."
        )

    template = ParsedParameterFile(f"{case_name}/{time_base}/{field_base}")
    result = WriteParameterFile(
        f"{case_name}/{time_out}/{field_out}",
        className=f"vol{val_type.capitalize()}Field",
    )
    result.content = template.content
    result.content._decoration["boundaryField"] = (
        "\n" + result.content._decoration["boundaryField"]
    )  # decoration at the bottom led to problems with the parser

    result.content["internalField"] = _generate_field(data_out[:n_internal, :])

    n_start = n_internal
    for bd_name, bd in mesh.boundary.items():
        if bd["type"] in IGNORED_PATCH_TYPES:
            continue
        result.content["boundaryField"][bd_name]["value"] = _generate_field(
            data_out[n_start : n_start + bd["nFaces"]]
        )
        n_start += bd["nFaces"]

    result.writeFile()


def write_foam(
    case_name: str,
    time_out: float,
    data_out: Data,
    data_names: Iterable[str],
    fields_base: Iterable[str],
    fields_out: Iterable[str],
    time_base: float = 0,
) -> None:
    """Writes attributes in a graph as OpenFOAM fields, based on templates.

    Args:
        case_name (str): Path to the folder containg an OpenFOAM case.
        time_out (float): Time where fields will be saved.
        data_out (Data): Graph to be saved.
        data_names (Iterable[str]): Attributes selected.
        fields_base (Iterable[str]): Names of the template fields.
        fields_out (Iterable[str]): Names of the output fields.
        time_base (float, optional): Time where template fields are located. Defaults to 0.
    """
    mesh = _read_mesh(case_name, read_boundaries=False, time=time_out, dynamic_mesh=True, read_centres=False)

    for data_name, field_base, file_out in zip(data_names, fields_base, fields_out):
        _write_tensor(
            case_name,
            time_out,
            data_out[data_name],
            field_base,
            file_out,
            time_base,
            mesh,
        )
