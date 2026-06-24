"""ANN index over learned fragment embeddings (cosine retrieval).

Each valid length-``L`` database fragment is encoded to an L2-normalized vector by an
:class:`~openawsem.memory.ml.encoder.Encoder`; :class:`EmbeddingIndex` stores those vectors and
answers ``search(query) -> (rows, sims)`` where ``rows`` are global
:class:`~openawsem.memory.retrieval.index.FragmentIndex` rows (so the caller can gather distances
with ``FragmentIndex.dist``) and ``sims`` are cosine similarities.

The index is encoder-agnostic: the same code serves v0 (BLOSUM-cosine) and learned encoders; the
encoder ``version`` is recorded in ``meta.json``.  Two backends:

* ``index_type="flat"`` (default) -- exact cosine via FAISS ``IndexFlatIP`` (numpy matmul fallback
  when FAISS is absent, so v0 runs anywhere).
* ``index_type="hnsw"`` -- approximate graph search (``IndexHNSWFlat``, inner product), ~100-1000x
  faster per query at ~0.98 top-20 recall.  The graph is persisted (build is slow), so the index
  loads without rebuilding -- this is the backend that makes the ml mutation engine fast enough for
  Monte-Carlo.
"""
from __future__ import annotations

import json
import os

import numpy as np

from openawsem.memory.ml.encoder import codes_from_windows, load_encoder

INDEX_DIRNAME = "embedding_index"


class _ANN:
    """Top-k cosine search over normalized vectors (FAISS flat / HNSW, or numpy fallback)."""

    def __init__(self, Z, *, index_type="flat", ef_search=64, ef_construction=200, M=32,
                 faiss_index=None):
        self.Z = np.ascontiguousarray(Z, dtype=np.float32)
        self.index_type = index_type
        self.ef_search = int(ef_search)
        self._faiss = faiss_index
        if self._faiss is not None:
            self._apply_ef()
            return
        try:
            import faiss
        except Exception:                      # noqa: BLE001
            if index_type == "hnsw":
                raise RuntimeError("index_type='hnsw' needs faiss (use the openawsem_torch env)")
            self._faiss = None                  # flat -> exact numpy fallback
            return
        d = self.Z.shape[1]
        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(d, int(M), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(ef_construction)
            index.add(self.Z)
            index.hnsw.efSearch = self.ef_search
        else:
            index = faiss.IndexFlatIP(d)
            index.add(self.Z)
        self._faiss = index

    def _apply_ef(self):
        if self.index_type == "hnsw":
            try:
                self._faiss.hnsw.efSearch = self.ef_search
            except Exception:                  # noqa: BLE001
                pass

    def search(self, Q, k):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        k = min(int(k), self.Z.shape[0])
        if self._faiss is not None:
            sims, idx = self._faiss.search(Q, k)
            return sims, idx
        scores = Q @ self.Z.T                                       # (q, N) cosine
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        sims = np.take_along_axis(scores, idx, axis=1)
        return sims, idx


class EmbeddingIndex:
    """Cosine-retrieval index over encoded database fragments.

    Attributes
    ----------
    Z : (N, dim) float32
        L2-normalized fragment embeddings.
    rows : (N,) int
        Global ``FragmentIndex`` row for each embedding (the fragment start).
    encoder :
        The encoder used to build it; also used to encode queries.
    index_type : "flat" | "hnsw"
    """

    def __init__(self, Z, rows, encoder, L, *, version=None, index_type="flat", ef_search=64,
                 faiss_index=None):
        self.Z = np.ascontiguousarray(Z, dtype=np.float32)
        self.rows = np.asarray(rows)
        self.encoder = encoder
        self.L = int(L)
        self.version = version or getattr(encoder, "version", "v0")
        self.index_type = index_type
        self.ef_search = int(ef_search)
        self._ann = _ANN(self.Z, index_type=index_type, ef_search=ef_search, faiss_index=faiss_index)

    # ----- build / persist ---------------------------------------------- #
    @staticmethod
    def index_dir(out_dir):
        return os.path.join(out_dir, INDEX_DIRNAME)

    @staticmethod
    def exists(out_dir):
        return os.path.isfile(os.path.join(EmbeddingIndex.index_dir(out_dir), "Z.npy"))

    @classmethod
    def build(cls, fragment_index, encoder="v0", *, L=9, batch=200_000, index_type="flat",
              ef_search=64):
        """Encode every valid length-``L`` fragment of ``fragment_index`` into an index.

        ``fragment_index`` is a loaded :class:`FragmentIndex`; ``encoder`` is an
        :class:`Encoder`, ``"v0"``, or a path to learned weights.
        """
        enc = load_encoder(encoder, L=L)
        rows = np.asarray(fragment_index._encoded_for_length(L)[0])
        N = len(rows)
        Z = np.empty((N, enc.dim), dtype=np.float32)
        for c0 in range(0, N, batch):
            c1 = min(c0 + batch, N)
            Z[c0:c1] = enc.encode_db(fragment_index, rows[c0:c1])
        return cls(Z, rows, enc, L, version=enc.version, index_type=index_type,
                   ef_search=ef_search)

    def save(self, out_dir, dest=None):
        dest = dest or self.index_dir(out_dir)
        os.makedirs(dest, exist_ok=True)
        np.save(os.path.join(dest, "Z.npy"), self.Z)
        np.save(os.path.join(dest, "rows.npy"), self.rows)
        with open(os.path.join(dest, "meta.json"), "w") as f:
            json.dump({"version": self.version, "L": self.L, "dim": int(self.Z.shape[1]),
                       "n": int(len(self.rows)), "index_type": self.index_type,
                       "ef_search": self.ef_search}, f)
        if self.index_type == "hnsw" and self._ann._faiss is not None:
            import faiss
            faiss.write_index(self._ann._faiss, os.path.join(dest, "ann.faiss"))
        return dest

    @classmethod
    def load(cls, out_dir, encoder=None, dest=None):
        dest = dest or cls.index_dir(out_dir)
        with open(os.path.join(dest, "meta.json")) as f:
            meta = json.load(f)
        Z = np.load(os.path.join(dest, "Z.npy"))
        rows = np.load(os.path.join(dest, "rows.npy"))
        enc = load_encoder(encoder if encoder is not None else meta["version"], L=meta["L"])
        index_type = meta.get("index_type", "flat")
        ef_search = int(meta.get("ef_search", 64))
        faiss_index = None
        ann_path = os.path.join(dest, "ann.faiss")
        if index_type == "hnsw" and os.path.isfile(ann_path):
            import faiss
            faiss_index = faiss.read_index(ann_path)
        return cls(Z, rows, enc, meta["L"], version=meta["version"], index_type=index_type,
                   ef_search=ef_search, faiss_index=faiss_index)

    # ----- query --------------------------------------------------------- #
    def search_vectors(self, Q, top=20):
        """Top-``top`` for precomputed normalized query vectors ``Q`` (``(q, dim)``) ->
        ``(rows, sims)`` each ``(q, top)`` (global rows, cosine)."""
        sims, idx = self._ann.search(np.atleast_2d(Q), top)
        return self.rows[idx], sims

    def search(self, window, top=20):
        """Top-``top`` database fragments for a 9-mer ``window`` (or list): ``(rows, sims)``.

        Code-based (v0/v1) convenience for single windows; context encoders (v3) retrieve through
        :meth:`search_vectors` with vectors from ``encoder.encode_query(full_sequence)``.  For a
        single query both outputs are 1-D; for a list both are ``(q, top)``.
        """
        single = isinstance(window, str)
        windows = [window] if single else list(window)
        Q = self.encoder.encode_codes(codes_from_windows(windows, self.L))
        rows, sims = self.search_vectors(Q, top)
        return (rows[0], sims[0]) if single else (rows, sims)
