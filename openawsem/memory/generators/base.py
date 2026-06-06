"""Backend framework: the :class:`Registry` and the base backend classes.

Each generation method is its own module in this package (``single``, ``local_blast``,
...), registered with ``@Registry.register("name")``.  Structure-based methods share the
:class:`StructureBackend` pipeline (search -> select -> fetch -> materialize); selection is
shared via :mod:`openawsem.memory.generators.selection`, and a method may add its own
selection/filtering in its own module.
"""
from __future__ import annotations

import logging

import pandas as pd

from openawsem.memory.memory import MemoryWells
from openawsem.memory.generators.selection import select_n_per_position

logger = logging.getLogger(__name__)


class Registry:
    """Maps a method name -> backend class."""

    _backends: dict = {}
    _experimental: dict = {}

    @classmethod
    def register(cls, name, *, experimental=False):
        def decorate(klass):
            (cls._experimental if experimental else cls._backends)[name] = klass
            klass.method_name = name
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

    def generate(self, sequence) -> "MemoryWells":
        raise NotImplementedError


class StructureBackend(FragmentBackend):
    """Shared pipeline for structure-based methods: search -> select -> fetch -> materialize.

    Subclasses implement :meth:`search` (sequence -> hits DataFrame) and :meth:`fetch`
    (hit -> molscene ``Scene``).  The base de-duplicates/ranks/keeps-N
    (:func:`select_n_per_position`), materializes each hit into an aligned fragment, and
    builds a :class:`MemoryWells`.
    """

    n_mem = 20
    min_seq_sep = 3
    max_seq_sep = 9
    well_width = 0.1
    weight = 1.0

    def search(self, sequence) -> pd.DataFrame:
        raise NotImplementedError

    def fetch(self, hit):
        raise NotImplementedError

    def materialize(self, hit, scene) -> pd.DataFrame:
        """Aligned fragment: CA/CB of the hit's subject range, mapped to target residues.

        Uses the (gap-free) hit alignment: subject residue ``s`` -> target residue
        ``query_start + (s - subject_start)``.  If the hit carries the BLAST subject
        sequence (``sseq``), the by-residue selection is validated against it and the
        fragment is dropped on a mismatch -- a cheap guard against structures whose ATOM
        numbering differs from the database sequence positions.
        """
        from openawsem.memory import _molscene
        from openawsem.memory.align import THREE_TO_ONE
        atoms = _molscene.cacb_nm(scene)
        if hit.get("chain"):
            atoms = atoms[atoms["chain"] == hit["chain"]]
        lo, hi = int(hit["subject_start"]), int(hit["subject_end"])
        window = atoms[(atoms["resid"] >= lo) & (atoms["resid"] <= hi)].copy()
        window["target_resid"] = window["resid"] - lo + int(hit["query_start"])
        window["weight"] = float(hit.get("weight", self.weight))

        sseq = hit.get("sseq")
        if sseq:
            sseq = sseq.replace("-", "")
            ca = window[window["name"] == "CA"].sort_values("resid")
            got = "".join(ca["resname"].map(THREE_TO_ONE).fillna("X"))
            identity = sum(a == b for a, b in zip(got, sseq))
            if len(got) != len(sseq) or identity < 0.8 * len(sseq):
                logger.warning("residue-mapping mismatch for %s/%s %d-%d (%r vs sseq %r); dropping",
                               hit.get("pdb_id"), hit.get("chain"), lo, hi, got, sseq)
                return window.iloc[0:0]
        return window

    def generate(self, sequence) -> "MemoryWells":
        hits = select_n_per_position(self.search(sequence), self.n_mem)
        fragments = []
        cache: dict = {}
        for hit in hits.to_dict("records"):
            key = (hit.get("pdb_id"), hit.get("chain"))
            if key not in cache:
                try:
                    cache[key] = self.fetch(hit)
                except Exception as exc:  # noqa: BLE001 - a failed fetch just drops the hit
                    logger.warning("fetch failed for %s: %s", key, exc)
                    cache[key] = None
            scene = cache[key]
            if scene is not None:
                fragments.append(self.materialize(hit, scene))
        memory = MemoryWells.from_fragments(fragments, min_seq_sep=self.min_seq_sep,
                                            max_seq_sep=self.max_seq_sep, well_width=self.well_width)
        memory.provenance.update({"method": getattr(self, "method_name", type(self).__name__),
                                  "n_hits": int(len(hits))})
        return memory
