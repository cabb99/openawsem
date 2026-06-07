"""Fetch protein structures as molscene ``Scene`` objects.

Three sources, in priority order:
  1. a **local mmCIF database** in the wwPDB *divided* layout
     (``<cif_dir>/<id[1:3]>/<id>.cif.gz``, also flat / uncompressed);
  2. **CIF download** from RCSB (``files.rcsb.org/download/<ID>.cif.gz``);
  3. **PDB download** via PDBFixer (molscene ``from_fixPDB``) -- the original method.

molscene parses CIF/PDB; this module only handles locating files, gunzipping (molscene's
``from_cif`` wants plain text), and downloading.
"""
from __future__ import annotations

import gzip
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def local_cif_path(pdb_id, cif_dir):
    """Return the path to ``pdb_id`` in a local mmCIF database, or ``None``."""
    pid = pdb_id.lower()
    cif_dir = Path(cif_dir)
    candidates = [cif_dir / pid[1:3] / f"{pid}.cif.gz", cif_dir / pid[1:3] / f"{pid}.cif",
                  cif_dir / f"{pid}.cif.gz", cif_dir / f"{pid}.cif"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _scene_from_cif_text(text):
    from molscene import Scene
    with tempfile.NamedTemporaryFile("w", suffix=".cif", delete=False) as out:
        out.write(text)
        tmp = out.name
    try:
        return Scene.from_cif(tmp)
    finally:
        os.unlink(tmp)


def _read_text(path):
    path = str(path)
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return handle.read()
    with open(path) as handle:
        return handle.read()


def fetch_structure(pdb_id, *, cif_dir=None, download="cif", timeout=30):
    """Return a molscene ``Scene`` for ``pdb_id`` (coordinates in Angstrom).

    ``cif_dir`` (if given and the entry is present) is used first; otherwise the
    structure is downloaded as ``download`` = ``"cif"`` (RCSB mmCIF) or ``"pdb"``
    (PDBFixer ``from_fixPDB``).
    """
    pdb_id = pdb_id.lower()
    if cif_dir is not None:
        path = local_cif_path(pdb_id, cif_dir)
        if path is not None:
            return _scene_from_cif_text(_read_text(path))
        logger.debug("%s not found in local cif_dir %s; downloading", pdb_id, cif_dir)

    if download == "pdb":
        from openawsem.memory._molscene import _require_molscene
        return _require_molscene().from_fixPDB(pdbid=pdb_id)
    if download == "cif":
        url = RCSB_CIF_URL.format(pdb_id=pdb_id.upper())
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = gzip.decompress(response.read()).decode()
        return _scene_from_cif_text(text)
    raise ValueError(f"unknown download format {download!r} (use 'cif' or 'pdb')")
