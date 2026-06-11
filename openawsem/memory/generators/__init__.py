"""Fragment-memory generation backends.

Each method lives in its own module and registers itself with :class:`Registry` on
import; importing this package imports them all so the registry is populated.  Usable
methods: ``single``, ``local_blast``.  The rest are experimental stubs.
"""
from __future__ import annotations

from openawsem.memory.generators.base import FragmentBackend, Registry, StructureBackend
from openawsem.memory.generators import selection
from openawsem.memory.generators import homology
from openawsem.memory.generators.single import SingleStructure
from openawsem.memory.generators.local_blast import LocalBlast
from openawsem.memory.generators.fragment_db import FragmentDB
from openawsem.memory.generators.local_db import LocalDB

# importing the remaining modules registers their (experimental) backends
from openawsem.memory.generators import (  # noqa: F401
    alphafold,
    atomistic,
    esm,
    ml,
    sequence_structure,
    web_blast,
)

__all__ = [
    "Registry", "FragmentBackend", "StructureBackend", "selection",
    "SingleStructure", "LocalBlast", "FragmentDB", "LocalDB",
]
