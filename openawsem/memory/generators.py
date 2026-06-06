"""Generation backends (the "bigger class") and their Registry.

A backend turns a sequence into a :class:`~openawsem.memory.memory.FragmentMemory`.
Usable backends register with ``@Registry.register("name")``; in-progress ones use
``experimental=True`` and raise a clear error until implemented (Phase 1B+).
"""
from __future__ import annotations

import pandas as pd

from openawsem.memory.memory import FragmentMemory


class Registry:
    """Maps a method name -> backend class."""

    _backends: dict = {}
    _experimental: dict = {}

    @classmethod
    def register(cls, name, *, experimental=False):
        def decorate(klass):
            (cls._experimental if experimental else cls._backends)[name] = klass
            return klass
        return decorate

    @classmethod
    def available(cls):
        return sorted(cls._backends)

    @classmethod
    def create(cls, name, **opts):
        if name in cls._backends:
            return cls._backends[name](**opts)
        if name in cls._experimental:
            raise NotImplementedError(
                f"fragment-memory method {name!r} is registered but not implemented yet. "
                f"Available now: {sorted(cls._backends) or 'none'}.")
        raise ValueError(f"unknown fragment-memory method {name!r}. "
                         f"Available: {sorted(cls._backends) or 'none'}.")


class FragmentBackend:
    """Base backend: produce a FragmentMemory from a sequence."""

    def generate(self, sequence) -> FragmentMemory:
        raise NotImplementedError


class StructureBackend(FragmentBackend):
    """Shared pipeline for structure-based methods: search -> select -> fetch -> align.

    Subclasses implement :meth:`search` and :meth:`fetch`; the base wires alignment
    (molscene ``get_sequence`` + Needleman-Wunsch) and feeds aligned fragments to
    :meth:`MemoryWells.from_fragments`.  Implemented in Phase 1B.
    """

    def search(self, sequence) -> pd.DataFrame:
        raise NotImplementedError

    def fetch(self, hit):
        raise NotImplementedError

    def align(self, scene, sequence, hit):
        raise NotImplementedError

    def generate(self, sequence) -> FragmentMemory:
        raise NotImplementedError


# --- backends (Phase 1B fills single/local_blast; the rest are later phases) --- #
@Registry.register("single", experimental=True)
class SingleStructure(StructureBackend):
    """One template structure -> memory (ports create_single_memory)."""


@Registry.register("local_blast", experimental=True)
class LocalBlast(StructureBackend):
    """PSI/BLAST vs a local database -> fragments (ports create_fragment_memory)."""


@Registry.register("web_blast", experimental=True)
class WebBlast(StructureBackend):
    """NCBI/web BLAST + CIF fetch (folds in the v2 downloaders via molscene)."""


@Registry.register("sequence_structure", experimental=True)
class SequenceStructure(FragmentBackend):
    """Decoy-trained sequence-structure potential (good vs bad fragments)."""


@Registry.register("alphafold", experimental=True)
class AlphaFold(StructureBackend):
    """AlphaFold model(s) -> memory (optionally pLDDT/PAE-weighted)."""


@Registry.register("esm", experimental=True)
class Esm(FragmentBackend):
    """ESMFold structure or ESM distance distribution -> memory."""


@Registry.register("atomistic", experimental=True)
class Atomistic(FragmentBackend):
    """Atomistic MD of short peptides -> sampled distance distributions (MemoryTable)."""


@Registry.register("ml", experimental=True)
class Ml(FragmentBackend):
    """Learned distograms -> MemoryTable (the non-parametric case)."""
