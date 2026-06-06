"""``alphafold`` backend (experimental): AlphaFold model(s) -> memory."""
from __future__ import annotations

from openawsem.memory.generators.base import Registry, StructureBackend


@Registry.register("alphafold", experimental=True)
class AlphaFold(StructureBackend):
    """AlphaFold model(s) -> memory (optionally pLDDT/PAE-weighted)."""
