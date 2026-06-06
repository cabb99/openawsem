"""``sequence_structure`` backend (experimental): decoy-trained potential."""
from __future__ import annotations

from openawsem.memory.generators.base import FragmentBackend, Registry


@Registry.register("sequence_structure", experimental=True)
class SequenceStructure(FragmentBackend):
    """Decoy-trained sequence-structure potential (scores good vs bad fragments)."""
