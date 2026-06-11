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
    """

    _ATOM_PAIRS = (("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB"))
    _db_cache: dict = {}

    def __init__(self, database, *, n_mem=20, fragment_length=9, min_seq_sep=3,
                 max_seq_sep=9, well_width=0.1, weight=1.0):
        self.database = str(database)
        self.n_mem = int(n_mem)
        self.fragment_length = int(fragment_length)
        self.min_seq_sep = int(min_seq_sep)
        self.max_seq_sep = int(max_seq_sep)
        self.well_width = float(well_width)
        self.weight = float(weight)

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

    def generate(self, sequence) -> MemoryWells:
        L = self.fragment_length
        cols: dict = {c: [] for c in WELLS_COLUMNS}
        n_windows = n_hits = 0
        base = 0
        for record in self._as_records(sequence):
            n = len(record)
            for start in range(n - L + 1):
                window = record[start:start + L]
                hits = self.db.search(window, L=L, top=self.n_mem)
                n_windows += 1
                for hit in hits.itertuples(index=False):
                    frag = self.db.get_fragment(hit.chain_uid, int(hit.start), L)
                    if len(frag) == 0:
                        continue
                    if self._accumulate(cols, frag, record, start, base, int(hit.start)):
                        n_hits += 1
            base += n

        frame = pd.DataFrame(cols)
        memory = MemoryWells(frame) if len(frame) else MemoryWells.empty()
        memory.attrs["params"] = {"min_seq_sep": self.min_seq_sep, "max_seq_sep": self.max_seq_sep,
                                  "well_width": self.well_width, "units": "nm", "cb_policy": "legacy"}
        memory.provenance.update({"method": "local_db", "database": self.database,
                                  "n_windows": int(n_windows), "n_hits": int(n_hits)})
        return memory

    def _accumulate(self, cols, frag, record, start, base, hit_start) -> bool:
        """Append wells for one retrieved fragment; return True if it contributed any."""
        contributed = False
        for row in frag.itertuples(index=False):
            sep = int(row.sep)
            if not (self.min_seq_sep <= sep <= self.max_seq_sep):
                continue
            a = int(row.internal_i) - hit_start
            b = int(row.internal_j) - hit_start
            resid_i = base + start + a + 1
            resid_j = base + start + b + 1
            gly_i = record[start + a] == "G"
            gly_j = record[start + b] == "G"
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
