"""``ml`` backend: learned-embedding fragment retrieval -> a :class:`MemoryWells`.

For each sliding window of the target, a learned encoder maps the 9-mer to an L2-normalized
vector; the :class:`~openawsem.memory.retrieval.embedding.EmbeddingIndex` returns the top-``k``
database fragments by cosine similarity, and their stored CA/CB distances become Gaussian wells
(the same soft accumulation as ``local_db``).  Retrieving by a learned *structural* metric lets
the memory include structurally-similar fragments whose sequences are far from the target --
coverage past BLOSUM sequence search.

``v0`` is the non-parametric BLOSUM-cosine encoder (the baseline); learned encoders are selected
by passing their saved-weights path as ``encoder``.

Thin coverage (a window whose best similarity is below ``backoff_sim``) falls back to the
context-free marginal: a Gaussian well moment-matched to the database-average distance
distribution for each residue-pair/separation/channel (the ``tensors_openawsem.npz`` the
``fragment_db`` backend uses).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from openawsem.memory.memory import MemoryWells, WELLS_COLUMNS
from openawsem.memory.generators.base import FragmentBackend, Registry
from openawsem.memory.generators.local_db import accumulate_soft_wells, _CACB_PAIRS

logger = logging.getLogger(__name__)


@Registry.register("ml")
class Ml(FragmentBackend):
    """Learned-embedding retrieval -> :class:`MemoryWells`.

    Parameters
    ----------
    database:
        Path to a ``fragdb`` output directory with a built ``fragment_index/`` (the distance
        store and amino-acid context).
    encoder:
        ``"v0"`` (BLOSUM-cosine, non-parametric) or a path to learned encoder weights.
    embedding_index:
        Optional directory of a prebuilt :class:`EmbeddingIndex`; defaults to building one in
        memory from the fragment index with ``encoder``.
    n_mem, fragment_length, top_k, temp:
        ``top_k`` fragments are retrieved per window and weighted ``softmax(similarity/temp)``,
        scaled to sum ``n_mem`` (so the well depth matches the soft ``local_db`` memory).
    min_seq_sep, max_seq_sep, well_width, weight:
        Well parameters (``sigma = well_width*sep**0.15``), identical to ``local_db``.
    backoff, backoff_sim, marginal_db:
        When ``backoff`` and a window's best similarity is below ``backoff_sim``, that window's
        wells come from the marginal tensor (``marginal_db``; defaults to
        ``<database>/tensors_openawsem.npz``) moment-matched to one Gaussian per pair/channel.
    """

    _ATOM_PAIRS = _CACB_PAIRS
    _emb_cache: dict = {}
    _marginal_cache: dict = {}

    def __init__(self, database, *, encoder="v0", embedding_index=None, n_mem=20,
                 fragment_length=9, top_k=20, temp=0.07, min_seq_sep=3, max_seq_sep=9,
                 well_width=0.1, weight=1.0, backoff=True, backoff_sim=0.3, marginal_db=None,
                 index_type="flat", ef_search=64):
        self.database = str(database)
        self.encoder = encoder
        self.embedding_index = embedding_index
        self.n_mem = int(n_mem)
        self.fragment_length = int(fragment_length)
        self.top_k = int(top_k)
        self.temp = float(temp)
        self.min_seq_sep = int(min_seq_sep)
        self.max_seq_sep = int(max_seq_sep)
        self.well_width = float(well_width)
        self.weight = float(weight)
        self.backoff = bool(backoff)
        self.backoff_sim = float(backoff_sim)
        self.marginal_db = marginal_db
        self.index_type = index_type
        self.ef_search = int(ef_search)

    # ----- lazily-loaded resources --------------------------------------- #
    @property
    def db(self):
        from openawsem.memory.retrieval import FragmentIndex
        key = ("idx", self.database)
        if key not in Ml._emb_cache:
            if not FragmentIndex.exists(self.database):
                raise FileNotFoundError(
                    f"No fragment_index at {self.database!r}; build one with "
                    "`python -m openawsem.memory.retrieval.index <db_dir>`")
            Ml._emb_cache[key] = FragmentIndex.load(self.database)
        return Ml._emb_cache[key]

    @property
    def emb(self):
        from openawsem.memory.retrieval.embedding import EmbeddingIndex
        spec = self.embedding_index if self.embedding_index is not None else self.database
        key = ("emb", spec, str(self.encoder), self.fragment_length)
        if key not in Ml._emb_cache:
            if self.embedding_index is not None:
                Ml._emb_cache[key] = EmbeddingIndex.load(self.embedding_index, encoder=self.encoder)
            elif EmbeddingIndex.exists(self.database) and self.encoder in (None, "v0"):
                Ml._emb_cache[key] = EmbeddingIndex.load(self.database, encoder=self.encoder)
            else:
                logger.info("building in-memory %s embedding index (%s) over %s",
                            self.encoder, self.index_type, self.database)
                Ml._emb_cache[key] = EmbeddingIndex.build(
                    self.db, self.encoder, L=self.fragment_length,
                    index_type=self.index_type, ef_search=self.ef_search)
        return Ml._emb_cache[key]

    @staticmethod
    def _as_records(sequence):
        return [sequence] if isinstance(sequence, str) else list(sequence)

    def mutation_engine(self, sequence, structure, *, k_fm=0.04184):
        """A reusable mutation-ΔE engine for the ml memory on a fixed ``structure``:
        ``engine.dE(position, new_aa)`` for a trial, ``engine.commit(position, new_aa)`` to advance
        a Monte-Carlo walk.  The learned cosine score is not BLOSUM-additive (no closed form), so
        each ``dE`` re-embeds and re-retrieves the windows covering the mutation; build the ``Ml``
        with ``index_type='hnsw'`` for MC-rate throughput (~1000 mut/s) vs ~2-3 mut/s exact-flat."""
        from openawsem.memory.mutation import MlMutationEnergy
        return MlMutationEnergy(self, sequence, structure, k_fm=k_fm)

    # ----- generation ---------------------------------------------------- #
    def generate(self, sequence) -> MemoryWells:
        L = self.fragment_length
        emb = self.emb
        idx = self.db
        cols: dict = {c: [] for c in WELLS_COLUMNS}
        n_windows = n_backoff = 0
        base = 0
        for record in self._as_records(sequence):
            n = len(record)
            nw = n - L + 1
            # one query encode per chain (so context encoders see the whole chain), then a single
            # batched ANN search over all windows.
            rows_all = sims_all = None
            if nw > 0:
                rows_all, sims_all = emb.search_vectors(emb.encoder.encode_query(record), self.top_k)
            for start in range(nw):
                rows, sims = rows_all[start], sims_all[start]
                if self.backoff and (len(sims) == 0 or float(np.max(sims)) < self.backoff_sim):
                    self._accumulate_backoff(cols, record, start, base)
                    n_backoff += 1
                else:
                    w = np.exp((sims - sims.max()) / self.temp)
                    w = self.n_mem * w / w.sum()
                    accumulate_soft_wells(cols, idx, record, start, base, rows, w,
                                          fragment_length=L, min_seq_sep=self.min_seq_sep,
                                          max_seq_sep=self.max_seq_sep, well_width=self.well_width,
                                          atom_pairs=self._ATOM_PAIRS)
                n_windows += 1
            base += n

        frame = pd.DataFrame(cols)
        memory = MemoryWells(frame) if len(frame) else MemoryWells.empty()
        memory.attrs["params"] = {"min_seq_sep": self.min_seq_sep, "max_seq_sep": self.max_seq_sep,
                                  "well_width": self.well_width, "units": "nm", "cb_policy": "legacy"}
        memory.provenance.update({"method": "ml", "database": self.database,
                                  "encoder": str(self.encoder), "top_k": self.top_k,
                                  "temp": self.temp, "n_windows": int(n_windows),
                                  "n_backoff": int(n_backoff)})
        return memory

    # ----- marginal backoff ---------------------------------------------- #
    @property
    def _marginal(self):
        """``(mean_nm, std_nm, aa_index, ap_index, sep_index)`` of the marginal tensor, or None.

        ``mean``/``std`` are ``(n_ap, n_sep, 20, 20)`` Gaussian moments of each cell's distance
        density, computed once and cached per tensor file."""
        if not self.backoff:
            return None
        path = self.marginal_db or os.path.join(self.database, "tensors_openawsem.npz")
        if path in Ml._marginal_cache:
            return Ml._marginal_cache[path]
        if not os.path.isfile(path):
            logger.warning("no marginal tensor at %s; backoff disabled", path)
            Ml._marginal_cache[path] = None
            return None
        from openawsem.memory.generators.fragment_db import FragmentDB
        t = FragmentDB._load(path)
        density, grid = t["local_density"], t["local_grid"]          # (ap,sep,20,20,R), (sep,R) A
        dr = (grid[:, 1] - grid[:, 0])[None, :, None, None]
        g = grid[None, :, None, None, :]
        m1 = (density * g).sum(-1) * dr
        m2 = (density * g ** 2).sum(-1) * dr
        mean = (m1 / 10.0).astype(np.float64)                        # nm
        std = (np.sqrt(np.clip(m2 - m1 ** 2, 0.0, None)) / 10.0).astype(np.float64)
        res = (mean, std, t["aa_index"], t["ap_index"], t["sep_index"])
        Ml._marginal_cache[path] = res
        return res

    def _accumulate_backoff(self, cols, record, start, base):
        """One marginal Gaussian well per pair/channel for a low-coverage window."""
        marg = self._marginal
        if marg is None:
            return
        mean, std, aa_index, ap_index, sep_index = marg
        L = self.fragment_length
        for a in range(L):
            for b in range(a + 1, L):
                sep = b - a
                if not (self.min_seq_sep <= sep <= self.max_seq_sep) or sep not in sep_index:
                    continue
                si = sep_index[sep]
                ci, cj = aa_index.get(record[start + a], -1), aa_index.get(record[start + b], -1)
                if ci < 0 or cj < 0:
                    continue
                resid_i, resid_j = base + start + a + 1, base + start + b + 1
                gly_i, gly_j = record[start + a] == "G", record[start + b] == "G"
                for name_i, name_j in self._ATOM_PAIRS:
                    if (name_i == "CB" and gly_i) or (name_j == "CB" and gly_j):
                        continue
                    ap = ap_index.get(f"{name_i}-{name_j}")
                    if ap is None:
                        continue
                    r_mean, sigma = float(mean[ap, si, ci, cj]), float(std[ap, si, ci, cj])
                    if not (r_mean > 0 and sigma > 0):
                        continue
                    cols["resid_i"].append(resid_i); cols["name_i"].append(name_i)
                    cols["resid_j"].append(resid_j); cols["name_j"].append(name_j)
                    cols["r_mean"].append(r_mean); cols["sigma"].append(sigma)
                    cols["weight"].append(float(self.n_mem))
