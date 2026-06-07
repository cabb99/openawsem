"""Backend framework: the :class:`Registry` and the base backend classes.

Each generation method is its own module in this package (``single``, ``local_blast``,
...), registered with ``@Registry.register("name")``.  Structure-based methods share the
:class:`StructureBackend` pipeline (search -> select -> fetch -> materialize); selection is
shared via :mod:`openawsem.memory.generators.selection`, and a method may add its own
selection/filtering in its own module.
"""
from __future__ import annotations

import logging
from pathlib import Path

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

    def _filter_hits(self, hits, sequence):
        """Hook for method-specific hit filtering (e.g. homolog exclusion). Default: no-op."""
        return hits

    def materialize(self, hit, scene) -> pd.DataFrame:
        """Aligned fragment: CA/CB of the hit's subject range, mapped to target residues.

        The BLAST subject *positions* (``sstart``/``send``) index the database sequence,
        which need not equal the structure's ATOM residue numbering (SEQRES vs author
        numbering, unmodeled residues).  So we map by **sequence content**: locate the
        gap-free subject sequence ``sseq`` within the chain's ordered CA sequence (the
        occurrence nearest ``sstart``), and take those residues.  The k-th fragment
        residue maps to target residue ``query_start + k``.  Fragments whose ``sseq`` is
        not found are dropped.
        """
        from openawsem.memory import _molscene
        from openawsem.memory.align import THREE_TO_ONE
        atoms = _molscene.cacb_nm(scene)
        if hit.get("chain"):
            atoms = atoms[atoms["chain"] == hit["chain"]]
        sseq = (hit.get("sseq") or "").replace("-", "")
        if not sseq:
            return atoms.iloc[0:0]

        ca = atoms[atoms["name"] == "CA"].drop_duplicates("resid").sort_values("resid")
        resids = ca["resid"].to_numpy()
        chain_seq = "".join(ca["resname"].map(THREE_TO_ONE).fillna("X"))
        starts = [i for i in range(len(chain_seq) - len(sseq) + 1) if chain_seq[i:i + len(sseq)] == sseq]
        if not starts:
            logger.warning("subject sequence %r not found in %s/%s; dropping",
                           sseq, hit.get("pdb_id"), hit.get("chain"))
            return atoms.iloc[0:0]

        expected = int(hit["subject_start"]) - 1
        offset = min(starts, key=lambda i: abs(i - expected))
        frag_resids = resids[offset:offset + len(sseq)]
        target_of = {int(r): int(hit["query_start"]) + k for k, r in enumerate(frag_resids)}
        window = atoms[atoms["resid"].isin(frag_resids)].copy()
        window["target_resid"] = window["resid"].map(target_of)
        window["weight"] = float(hit.get("weight", self.weight))
        return window

    def generate(self, sequence, gro_dir=None) -> "MemoryWells":
        """Generate a memory; with ``gro_dir`` also write a gro library and record the
        fragments used (so :meth:`MemoryWells.to_frags_mem` can emit a legacy frags.mem)."""
        hits = self._filter_hits(self.search(sequence), sequence)
        hits = select_n_per_position(hits, self.n_mem)
        if gro_dir is not None:
            gro_dir = Path(gro_dir)
            gro_dir.mkdir(parents=True, exist_ok=True)
        fragments, used, scene_cache, gro_written = [], [], {}, {}
        for hit in hits.to_dict("records"):
            key = (hit.get("pdb_id"), hit.get("chain"))
            if key not in scene_cache:
                try:
                    scene_cache[key] = self.fetch(hit)
                except Exception as exc:  # noqa: BLE001 - a failed fetch just drops the hit
                    logger.warning("fetch failed for %s: %s", key, exc)
                    scene_cache[key] = None
            scene = scene_cache[key]
            if scene is None:
                continue
            window = self.materialize(hit, scene)
            if len(window) == 0:
                continue
            fragments.append(window)
            if gro_dir is not None:
                self._record_used(used, gro_written, gro_dir, hit, scene, window)
        memory = MemoryWells.from_fragments(fragments, min_seq_sep=self.min_seq_sep,
                                            max_seq_sep=self.max_seq_sep, well_width=self.well_width)
        memory.provenance.update({"method": getattr(self, "method_name", type(self).__name__),
                                  "n_hits": int(len(hits))})
        if used:
            memory.provenance["fragments"] = used
        return memory

    @staticmethod
    def _record_used(used, gro_written, gro_dir, hit, scene, window):
        """Write the source-chain gro (once) and record one frags.mem line for ``window``.

        Only contiguous-residue fragments are recorded, since a legacy frags.mem line
        indexes a residue *range* in the gro.
        """
        ca_resids = sorted(int(r) for r in window.loc[window["name"] == "CA", "resid"].unique())
        if len(ca_resids) < 2 or ca_resids[-1] - ca_resids[0] != len(ca_resids) - 1:
            return
        key = (hit.get("pdb_id"), hit.get("chain"))
        if key not in gro_written:
            path = Path(gro_dir) / f"{hit.get('pdb_id')}{hit.get('chain') or ''}.gro"
            scene.write_awsem_gro(str(path), chain=hit.get("chain"))
            gro_written[key] = str(path)
        used.append({"location": gro_written[key],
                     "target_start": int(window["target_resid"].min()),
                     "fragment_start": ca_resids[0], "frag_len": len(ca_resids),
                     "weight": float(window["weight"].iloc[0])})
