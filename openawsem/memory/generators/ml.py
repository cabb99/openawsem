"""``ml`` backend (experimental): learned distograms -> MemoryTable."""
from __future__ import annotations

from openawsem.memory.generators.base import FragmentBackend, Registry


@Registry.register("ml", experimental=True)
class Ml(FragmentBackend):
    """Learned distograms -> MemoryTable (the non-parametric case)."""
