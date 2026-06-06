"""``web_blast`` backend (experimental): NCBI/web BLAST + CIF fetch."""
from __future__ import annotations

from openawsem.memory.generators.base import Registry, StructureBackend


@Registry.register("web_blast", experimental=True)
class WebBlast(StructureBackend):
    """NCBI/web BLAST + CIF fetch (will fold in the v2 downloaders via molscene)."""
