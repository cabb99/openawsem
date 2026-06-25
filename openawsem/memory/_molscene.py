"""molscene helpers, isolated so the rest of openawsem.memory stays molscene-agnostic.

molscene stores coordinates in Angstrom; the AWSEM ``.gro`` is in nm, so reads divide
by 10 to return nm (and molscene's writer multiplies by 10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_CA_CB = ("CA", "CB")

# Ideal virtual-CB construction (trRosetta / OpenAWSEM convention), matching the fragment database:
# CB = c0*cross(CA-N, C-CA) + c1*(CA-N) + c2*(C-CA) + CA, in ANGSTROM (the formula mixes a length^2
# cross product with length terms, so it is NOT scale-invariant -- build in Angstrom, then convert).
VIRTUAL_CB_COEFFS = (-0.58273431, 0.56802827, -0.54067466)


def virtual_cb(n, ca, c):
    """Construct a virtual CB from backbone N, CA, C (each (3,)), in the SAME units as the inputs
    (must be Angstrom; nm gives a CB about 1 A off)."""
    b, cv = ca - n, c - ca
    c0, c1, c2 = VIRTUAL_CB_COEFFS
    return c0 * np.cross(b, cv) + c1 * b + c2 * cv + ca


def _require_molscene():
    try:
        from molscene import Scene
    except ImportError as exc:  # pragma: no cover - molscene is a declared dependency
        raise ImportError(
            "openawsem.memory requires the 'molscene' package for structure parsing "
            "(pip install molscene, or `pip install -e <checkout>`)."
        ) from exc
    return Scene


def cacb_nm(scene_or_frame) -> pd.DataFrame:
    """Return CA/CB atoms as ``[chain, resid, name, resname, x, y, z]`` in **nm** (Angstrom/10)."""
    frame = pd.DataFrame(scene_or_frame)[["chain", "resid", "name", "resname", "x", "y", "z"]]
    frame = frame[frame["name"].isin(_CA_CB)].copy()
    xyz = frame[["x", "y", "z"]].to_numpy(dtype=float) / 10.0
    return pd.DataFrame({
        "chain": frame["chain"].astype(object).to_numpy(),
        "resid": frame["resid"].astype("int64").to_numpy(),
        "name": frame["name"].astype(object).to_numpy(),
        "resname": frame["resname"].astype(object).to_numpy(),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
    }).reset_index(drop=True)


def cacb_with_virtual_cb(scene_or_frame) -> pd.DataFrame:
    """Like :func:`cacb_nm` but adds a virtual CB (nm) at every residue that has N/CA/C but no observed
    CB (i.e. glycine), built in Angstrom with :func:`virtual_cb`.  Observed CB atoms are kept as-is
    (policy ``observed_then_virtual``, matching the fragment database)."""
    frame = pd.DataFrame(scene_or_frame)[["chain", "resid", "name", "resname", "x", "y", "z"]]
    bb = frame[frame["name"].isin(("N", "CA", "C", "CB"))].copy()
    bb["resid"] = bb["resid"].astype("int64")
    out_rows = []
    for (chain, resid), g in bb.groupby(["chain", "resid"], sort=False):
        atoms = {row["name"]: np.array([row.x, row.y, row.z], float) for _, row in g.iterrows()}
        resname = g["resname"].iloc[0]
        if "CA" in atoms:
            out_rows.append((chain, resid, "CA", resname, *(atoms["CA"] / 10.0)))
        if "CB" in atoms:
            out_rows.append((chain, resid, "CB", resname, *(atoms["CB"] / 10.0)))
        elif all(a in atoms for a in ("N", "CA", "C")):                       # build a virtual CB
            cb = virtual_cb(atoms["N"], atoms["CA"], atoms["C"]) / 10.0
            out_rows.append((chain, resid, "CB", resname, *cb))
    cols = ["chain", "resid", "name", "resname", "x", "y", "z"]
    return pd.DataFrame(out_rows, columns=cols).reset_index(drop=True)


def read_awsem_gro(path) -> pd.DataFrame:
    """Read an AWSEM ``.gro`` and return its CA/CB atoms as a nm DataFrame."""
    Scene = _require_molscene()
    return cacb_nm(Scene.from_awsem_gro(str(path)))


def load_structure(path):
    """Load a PDB/mmCIF into a molscene ``Scene`` (Angstrom). Used by structure backends."""
    Scene = _require_molscene()
    path = str(path)
    if path.lower().endswith((".cif", ".mmcif")):
        return Scene.from_cif(path)
    return Scene.from_pdb(path)
