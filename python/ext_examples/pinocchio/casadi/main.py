import os
from tempfile import TemporaryDirectory
from pathlib import Path
import time
import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import xml.etree.ElementTree as ET


if __name__ == "__main__":
    simplified = True
    prefix = "simplified_" if simplified else ""
    # urdf_path ...
    model = pin.buildModelFromUrdf(str(urdf_path))

    if simplified:
        for i, inertia in enumerate(model.inertias):
            print(f"original inertia: {inertia.inertia}")
            I = np.diag(np.diag(inertia.inertia))
            inertia_new = pin.Inertia(inertia.mass, inertia.lever, I)
            model.inertias[i] = inertia_new
            # print(f"Zeroed out inertia: {inertia.inertia}")

    data = model.createData()
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    cq = casadi.SX.sym("x", model.nq, 1)
    cv = casadi.SX.sym("v", model.nv, 1)

    for inertia in cmodel.inertias:
        print(f"Zeroed out inertia: {inertia.inertia}")

    c_mass = cpin.crba(cmodel, cdata, cq)
    c_mass_diag = casadi.diag(c_mass)
    c_coriolis = cpin.computeCoriolisMatrix(cmodel, cdata, cq, cv)
    c_coriolis_diag = casadi.diag(c_coriolis)
    c_grav = cpin.computeGeneralizedGravity(cmodel, cdata, cq)

    # generate the functions
    c_mass = casadi.Function(f"{prefix}compute_mass", [cq], [c_mass])
    c_mass.generate(f"{prefix}compute_m.c", {"with_header": True})

    c_mass_diag = casadi.Function(f"{prefix}compute_mass_diag", [cq], [c_mass_diag])
    c_mass_diag.generate(f"{prefix}compute_m_diag.c", {"with_header": True})

    f_coriolis = casadi.Function(f"{prefix}compute_coriolis", [cq, cv], [c_coriolis])
    f_coriolis.generate(f"{prefix}compute_coriolis.c", {"with_header": True})

    f_coriolis_diag = casadi.Function(f"{prefix}compute_coriolis_diag", [cq, cv], [c_coriolis_diag])
    f_coriolis_diag.generate(f"{prefix}compute_coriolis_diag.c", {"with_header": True})

    f_grav = casadi.Function(f"{prefix}compute_grav", [cq], [c_grav])
    f_grav.generate(f"{prefix}compute_g.c", {"with_header": True})
