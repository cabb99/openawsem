"""``local_db`` backend: real fragments retrieved from the local residue database.

Like ``local_blast`` (PSI-BLAST + structure fetch) but single-sequence and fetch-free: for
each sliding window of the target, a BLOSUM62 k-mer search over a residue-centric database
(:class:`openawsem.memory.retrieval.FragmentDatabase`) returns the most sequence-similar real
fragments, and their **stored** pairwise distances become Gaussian wells.  This places each
well on the distances of actual homologous fragments (target-specific geometry), unlike
``fragment_db`` which uses the database-average distance per residue-pair type.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from openawsem.memory.memory import MemoryWells, WELLS_COLUMNS
from openawsem.memory.generators import homology
from openawsem.memory.generators.base import FragmentBackend, Registry

logger = logging.getLogger(__name__)


@Registry.register("local_db")
class LocalDB(FragmentBackend):
    """Database-retrieved real fragments -> a :class:`MemoryWells`.

    Parameters
    ----------
    database:
        Path to a ``fragdb`` output directory containing ``residues.csv[.gz]``.
    n_mem, fragment_length:
        Fragments kept per window and the sliding-window length (defaults match
        ``local_blast``: top-20 hits of 9-mers).
    min_seq_sep, max_seq_sep, well_width, weight:
        Same well parameters as ``local_blast`` (``sigma = well_width*sep**0.15``).
    brain_damage, cutoff_identical, homolog_evalue, homolog_database, blast_exe, num_threads,
    homolog_oversample:
        Homolog handling, identical flags/semantics to ``local_blast`` (0 all | 1 exclude
        homologs | 2 homologs except self | 0.5 self only).  Homologs are found with one
        full-sequence PSI-BLAST against ``homolog_database`` (default ``openawsem.data_path.blast``,
        since this method's own ``database`` is the residue DB, not a BLAST DB) and fragments
        are filtered by ``pdb_id``; ``homolog_oversample`` widens the per-window search so
        ``n_mem`` survives the exclusion.
    """

    _ATOM_PAIRS = (("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB"))
    _db_cache: dict = {}

    def __init__(self, database, *, n_mem=20, fragment_length=9, min_seq_sep=3,
                 max_seq_sep=9, well_width=0.1, weight=1.0, brain_damage=0,
                 cutoff_identical=90, homolog_evalue=0.005, homolog_database=None,
                 blast_exe="psiblast", num_threads=1, homolog_oversample=10, seeded=True):
        self.database = str(database)
        self.seeded = bool(seeded)
        self.n_mem = int(n_mem)
        self.fragment_length = int(fragment_length)
        self.min_seq_sep = int(min_seq_sep)
        self.max_seq_sep = int(max_seq_sep)
        self.well_width = float(well_width)
        self.weight = float(weight)
        self.brain_damage = brain_damage
        self.cutoff_identical = cutoff_identical
        self.homolog_evalue = homolog_evalue
        self.homolog_database = homolog_database
        self.blast_exe = blast_exe
        self.num_threads = num_threads
        self.homolog_oversample = int(homolog_oversample)

    @property
    def db(self):
        if self.database not in LocalDB._db_cache:
            from openawsem.memory.retrieval import FragmentDatabase, FragmentIndex
            if FragmentIndex.exists(self.database):
                logger.info("loading prebuilt fragment index for %s", self.database)
                LocalDB._db_cache[self.database] = FragmentIndex.load(self.database)
            else:
                LocalDB._db_cache[self.database] = FragmentDatabase.load(self.database)
        return LocalDB._db_cache[self.database]

    @staticmethod
    def _as_records(sequence):
        """One record per chain; a plain string is a single record (cf. ``local_blast``)."""
        return [sequence] if isinstance(sequence, str) else list(sequence)

    def _resolve_homolog_database(self):
        if self.homolog_database:
            return str(self.homolog_database)
        import openawsem
        return str(openawsem.data_path.blast)

    def _homologs(self, sequence) -> dict:
        """Target's homologs ``{pdb_id: max pident}`` (empty when brain_damage off)."""
        if not self.brain_damage:
            return {}
        return homology.blast_homologs(self._as_records(sequence),
                                       database=self._resolve_homolog_database(),
                                       blast_exe=self.blast_exe, homolog_evalue=self.homolog_evalue,
                                       num_threads=self.num_threads, brain_damage=self.brain_damage)

    def _search(self, window, top):
        """Search the DB for ``window``; use the seed index when available (FragmentIndex)."""
        from openawsem.memory.retrieval import FragmentIndex
        if self.seeded and isinstance(self.db, FragmentIndex):
            return self.db.search(window, L=self.fragment_length, top=top, seeded=True)
        return self.db.search(window, L=self.fragment_length, top=top)

    def _window_hits(self, window, homologs):
        """Search a window, apply brain_damage by ``pdb_id``, keep the top ``n_mem`` survivors."""
        top = self.n_mem * self.homolog_oversample if self.brain_damage else self.n_mem
        hits = self._search(window, top)
        if self.brain_damage and len(hits):
            hits = hits.assign(pdb_id=[str(c).split("_")[0] for c in hits["chain_uid"]])
            hits = homology.apply_brain_damage(hits, homologs, brain_damage=self.brain_damage,
                                               cutoff_identical=self.cutoff_identical)
        return hits.head(self.n_mem)

    def _window_wells(self, start, base, window, homologs, cols) -> int:
        """Append the wells of one window (query=``window``) to ``cols``; return n_hits used."""
        n_hits = 0
        for hit in self._window_hits(window, homologs).itertuples(index=False):
            frag = self.db.get_fragment(hit.chain_uid, int(hit.start), self.fragment_length)
            if len(frag) and self._accumulate(cols, frag, window, start, base, int(hit.start)):
                n_hits += 1
        return n_hits

    def generate(self, sequence) -> MemoryWells:
        L = self.fragment_length
        homologs = self._homologs(sequence)
        cols: dict = {c: [] for c in WELLS_COLUMNS}
        n_windows = n_hits = 0
        base = 0
        for record in self._as_records(sequence):
            n = len(record)
            for start in range(n - L + 1):
                n_hits += self._window_wells(start, base, record[start:start + L], homologs, cols)
                n_windows += 1
            base += n

        frame = pd.DataFrame(cols)
        memory = MemoryWells(frame) if len(frame) else MemoryWells.empty()
        memory.attrs["params"] = {"min_seq_sep": self.min_seq_sep, "max_seq_sep": self.max_seq_sep,
                                  "well_width": self.well_width, "units": "nm", "cb_policy": "legacy"}
        memory.provenance.update({"method": "local_db", "database": self.database,
                                  "brain_damage": self.brain_damage,
                                  "n_windows": int(n_windows), "n_hits": int(n_hits)})
        return memory

    def _accumulate(self, cols, frag, window, start, base, hit_start) -> bool:
        """Append wells for one retrieved fragment; return True if it contributed any.

        ``window`` is the (possibly mutated) query letters of this window; the GLY test uses
        the query identity so a mutation to/from G drops/adds the right CB pairs.
        """
        contributed = False
        for row in frag.itertuples(index=False):
            sep = int(row.sep)
            if not (self.min_seq_sep <= sep <= self.max_seq_sep):
                continue
            a = int(row.internal_i) - hit_start
            b = int(row.internal_j) - hit_start
            resid_i = base + start + a + 1
            resid_j = base + start + b + 1
            gly_i = window[a] == "G"
            gly_j = window[b] == "G"
            sigma = self.well_width * sep ** 0.15
            for name_i, name_j in self._ATOM_PAIRS:
                if (name_i == "CB" and gly_i) or (name_j == "CB" and gly_j):
                    continue
                d = getattr(row, f"d_{name_i}_{name_j}")
                if d is None or (isinstance(d, float) and math.isnan(d)) or d == "":
                    continue
                cols["resid_i"].append(resid_i)
                cols["name_i"].append(name_i)
                cols["resid_j"].append(resid_j)
                cols["name_j"].append(name_j)
                cols["r_mean"].append(float(d) / 10.0)   # Angstrom -> nm
                cols["sigma"].append(sigma)
                cols["weight"].append(self.weight)
                contributed = True
        return contributed
