import io
import numpy as np
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile


def write_to_internal_field(file_out, data_out):
    n_comps = data_out.size(-1)
    val_type = "vector" if n_comps == 3 else "scalar"
    newline = ")\n(" if n_comps == 3 else "\n"
    begtoken = "(" if n_comps == 3 else ""
    offile = ParsedParameterFile(file_out)
    s = io.BytesIO()
    np.savetxt(s, data_out.detach().numpy(), newline=newline)
    desc = f"nonuniform List<{val_type}> {len(data_out)}"
    vals = "(\n" + begtoken + s.getvalue().decode("utf-8")[:-1] + ")"
    offile.content["internalField"] = desc + vals
    return offile.writeFile()
