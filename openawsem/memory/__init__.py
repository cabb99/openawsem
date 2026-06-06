"""Fragment-memory generation as a submodule.

A fragment memory is a ``pandas.DataFrame`` of pairwise CA/CB distance restraints
(:class:`FragmentMemory`), in one of two representations:

* :class:`MemoryWells` -- a sum of Gaussian wells (the AWSEM fragment-memory potential)
* :class:`MemoryTable` -- a sampled potential (for learned/sampled methods)

Both reduce to the tabulated potential the OpenMM ``fragment_memory_term`` consumes via
``.to_openmm_force(oa)``.  Generation *methods* are pluggable backends found through the
:class:`Registry`; :func:`generate` is the top-level entry point.
"""
from __future__ import annotations

from openawsem.memory.memory import (
    FragmentMemory,
    MemoryTable,
    MemoryWells,
    TABLE_COLUMNS,
    WELLS_COLUMNS,
)
from openawsem.memory.generators import (
    FragmentBackend,
    Registry,
    StructureBackend,
)

__all__ = [
    "FragmentMemory", "MemoryWells", "MemoryTable", "WELLS_COLUMNS", "TABLE_COLUMNS",
    "Registry", "FragmentBackend", "StructureBackend", "generate", "available_methods",
]


def generate(sequence, method="local_blast", **opts) -> FragmentMemory:
    """Generate a fragment memory for ``sequence`` with the named ``method``."""
    return Registry.create(method, **opts).generate(sequence)


def available_methods():
    """Names of usable generation methods (experimental stubs excluded)."""
    return Registry.available()
