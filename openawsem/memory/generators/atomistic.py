"""``atomistic`` backend (experimental): MD of short peptides -> distance distributions."""
from __future__ import annotations

from openawsem.memory.generators.base import FragmentBackend, Registry


@Registry.register("atomistic", experimental=True)
class Atomistic(FragmentBackend):
    """Atomistic MD of short peptides -> sampled distance distributions (MemoryTable)."""
