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
from openawsem.memory.generators.local_db import (accumulate_soft_wells, emit_mixture_wells,
                                                  _CACB_PAIRS)

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
        """Marginal wells for a low-coverage window: the K=1, context-free case of a predicted
        mixture (so it shares :func:`emit_mixture_wells`'s GLY/CB drop and resid bookkeeping)."""
        marg = self._marginal
        if marg is None:
            return
        from openawsem.memory.ml.fingerprint import in_window_pairs
        mean, std, aa_index, ap_index, sep_index = marg
        L = self.fragment_length
        pa, pb, ps = in_window_pairs(L, min_sep=self.min_seq_sep, max_sep=self.max_seq_sep)
        P, C = len(pa), len(self._ATOM_PAIRS)
        pi = np.ones((P, C, 1)); mu = np.full((P, C, 1), np.nan); sg = np.full((P, C, 1), np.nan)
        for p, (a, b, sep) in enumerate(zip(pa, pb, ps)):
            si = sep_index.get(int(sep))
            ci, cj = aa_index.get(record[start + a], -1), aa_index.get(record[start + b], -1)
            if si is None or ci < 0 or cj < 0:
                continue
            for c, (name_i, name_j) in enumerate(self._ATOM_PAIRS):
                ap = ap_index.get(f"{name_i}-{name_j}")
                if ap is None:
                    continue
                r_mean, sigma = float(mean[ap, si, ci, cj]), float(std[ap, si, ci, cj])
                if r_mean > 0 and sigma > 0:
                    mu[p, c, 0], sg[p, c, 0] = r_mean, sigma
        emit_mixture_wells(cols, record, start, base, fragment_length=L,
                           min_seq_sep=self.min_seq_sep, max_seq_sep=self.max_seq_sep,
                           pair_a=pa, pair_b=pb, pair_sep=ps, pi=pi, mu=mu, sigma=sg,
                           n_mem=self.n_mem, atom_pairs=self._ATOM_PAIRS)


@Registry.register("ml_gen")
class MlGen(FragmentBackend):
    """v4 (generative): predict the fragment-memory wells directly from ESM-2 context, no database.

    A :class:`~openawsem.memory.ml.distogram.DistogramModel` predicts, per in-window CA/CB pair, a
    ``K``-component Gaussian mixture over the pair distance from the window's frozen ESM-2 context.
    Each component becomes a well (``weight_k = n_mem*pi_k``, ``r_mean=mu_k``, ``sigma=sigma_k``), so
    the output is an ordinary :class:`MemoryWells` -- but built from one ESM-2 pass per chain rather
    than a database search.  This is the K-component generalization of :meth:`Ml._accumulate_backoff`
    (which emits one Gaussian per pair from a context-free marginal table).

    Parameters
    ----------
    model:
        A :class:`DistogramModel` or a path to a saved v4 head (loaded query-only, ESM run on the
        target chain at generation time).
    """

    _ATOM_PAIRS = _CACB_PAIRS

    def __init__(self, model=None, *, database=None, n_mem=20, fragment_length=9, min_seq_sep=3,
                 max_seq_sep=9, well_width=0.1, weight=1.0):
        self._model = model
        self.database = str(database) if database is not None else None
        self.n_mem = int(n_mem)
        self.fragment_length = int(fragment_length)
        self.min_seq_sep = int(min_seq_sep)
        self.max_seq_sep = int(max_seq_sep)
        self.well_width = float(well_width)
        self.weight = float(weight)

    @property
    def model(self):
        if isinstance(self._model, str):
            from openawsem.memory.ml.distogram import load_distogram
            self._model = load_distogram(self._model)
        if self._model is None:
            raise ValueError("MlGen needs a `model` (a DistogramModel or a saved-head path)")
        return self._model

    def generate(self, sequence) -> MemoryWells:
        import torch
        L = self.fragment_length
        model = self.model
        model.module.eval()
        cols: dict = {c: [] for c in WELLS_COLUMNS}
        n_windows = 0
        base = 0
        for record in self._as_records(sequence):
            n = len(record)
            nw = n - L + 1
            if nw > 0:
                feat = model.residue_features(record)                     # one ESM-2 pass per chain
                ctx = np.stack([feat[s:s + L] for s in range(nw)])        # (nw, L, d_esm)
                with torch.no_grad():
                    pi, mu, sigma = model.forward(torch.as_tensor(ctx, device=model.device))
                pi, mu, sigma = pi.cpu().numpy(), mu.cpu().numpy(), sigma.cpu().numpy()
                for start in range(nw):
                    emit_mixture_wells(cols, record, start, base, fragment_length=L,
                                       min_seq_sep=self.min_seq_sep, max_seq_sep=self.max_seq_sep,
                                       pair_a=model.pair_a, pair_b=model.pair_b,
                                       pair_sep=model.pair_sep, pi=pi[start], mu=mu[start],
                                       sigma=sigma[start], n_mem=self.n_mem,
                                       atom_pairs=self._ATOM_PAIRS)
                    n_windows += 1
            base += n

        frame = pd.DataFrame(cols)
        memory = MemoryWells(frame) if len(frame) else MemoryWells.empty()
        memory.attrs["params"] = {"min_seq_sep": self.min_seq_sep, "max_seq_sep": self.max_seq_sep,
                                  "well_width": self.well_width, "units": "nm", "cb_policy": "legacy"}
        memory.provenance.update({"method": "ml_gen", "model": str(self._model),
                                  "K": int(model.K), "n_windows": int(n_windows)})
        return memory

    @staticmethod
    def _as_records(sequence):
        return [sequence] if isinstance(sequence, str) else list(sequence)
