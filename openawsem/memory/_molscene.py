"""molscene helpers, isolated so the rest of openawsem.memory stays molscene-agnostic.

molscene stores coordinates in Angstrom; the AWSEM ``.gro`` is in nm, so reads divide
by 10 to return nm (and molscene's writer multiplies by 10).
"""
from __future__ import annotations

import pandas as pd

_CA_CB = ("CA", "CB")


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
    """Return CA/CB atoms as ``[resid, name, resname, x, y, z]`` in **nm** (Angstrom/10)."""
    frame = pd.DataFrame(scene_or_frame)[["resid", "name", "resname", "x", "y", "z"]]
    frame = frame[frame["name"].isin(_CA_CB)].copy()
    xyz = frame[["x", "y", "z"]].to_numpy(dtype=float) / 10.0
    return pd.DataFrame({
        "resid": frame["resid"].astype("int64").to_numpy(),
        "name": frame["name"].astype(object).to_numpy(),
        "resname": frame["resname"].astype(object).to_numpy(),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
    }).reset_index(drop=True)


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
