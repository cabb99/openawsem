"""Fragment encoders: sequence 9-mer -> L2-normalized vector ``z`` for cosine retrieval.

Encoders share one interface so the retrieval index and the ``ml`` backend are encoder-agnostic
(see the model roadmap in the plan):

* ``version``     -- short tag stored with a built index (``"v0"``, ``"v1"``, ...).
* ``dim``         -- output dimensionality.
* ``encode_codes(codes)`` -- the primitive: ``(M, L)`` int amino-acid codes -> ``(M, dim)``
  float32, L2-normalized.  Operating on codes avoids materializing millions of strings when
  encoding the whole database.
* ``encode(windows)`` -- convenience wrapper over one-letter strings.

``v0`` (:class:`BlosumCosineEncoder`) is non-parametric: each residue becomes its BLOSUM62 row
over the 20 standard amino acids, the 9 rows are concatenated (180-dim) and L2-normalized.  It is
the baseline every learned encoder must beat; it needs only numpy.  Learned encoders (v1 MLP,
...) import torch lazily and are added alongside.
"""
from __future__ import annotations

import numpy as np

from openawsem.memory.retrieval.blosum import SCORE_TABLE, UNKNOWN_INDEX, encode_sequence

N_STD = 20                       # standard amino acids (BLOSUM row width used as features)


def codes_from_windows(windows, L=None):
    """One-letter 9-mer strings -> ``(M, L)`` int8 amino-acid codes (unknown -> X)."""
    windows = list(windows)
    L = L or (len(windows[0]) if windows else 0)
    out = np.empty((len(windows), L), dtype=np.int16)
    for i, w in enumerate(windows):
        out[i] = encode_sequence(w)[:L]
    return out


def _l2_normalize(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


#: BLOSUM62 row features over the 20 standard AAs, indexed by amino-acid code (X = row 20).
_BLOSUM_ROWS = np.asarray(SCORE_TABLE[:, :N_STD], dtype=np.float32)


def blosum_features(codes):
    """``(M, L)`` amino-acid codes -> ``(M, L*20)`` concatenated BLOSUM62 rows (un-normalized).

    Shared input representation for v0 (then L2-normalized) and the learned v1 MLP.
    """
    codes = np.asarray(codes)
    feats = _BLOSUM_ROWS[np.clip(codes, 0, UNKNOWN_INDEX)]        # (M, L, 20)
    return feats.reshape(len(codes), -1)


class Encoder:
    """Encoder interface.  Two pathways drive the index and the ``ml`` backend so the same code
    serves sequence encoders (v0/v1) and context encoders (v3):

    * :meth:`encode_db` -- encode database fragments by their start rows (index build / training).
    * :meth:`encode_query` -- encode every window of a query sequence (inference).

    Sequence encoders implement :meth:`encode_codes` and inherit the row/sequence pathways below;
    context encoders (ESM) override :meth:`encode_db`/:meth:`encode_query` directly.
    """

    version = "base"
    dim = 0
    L = 9

    def encode_codes(self, codes) -> np.ndarray:
        raise NotImplementedError

    def encode(self, windows) -> np.ndarray:
        return self.encode_codes(codes_from_windows(windows))

    def encode_db(self, fragment_index, rows) -> np.ndarray:
        """Encode database fragments by start ``rows`` (window codes read from the index)."""
        codes = np.asarray(fragment_index.ctx[np.asarray(rows), :self.L])
        return self.encode_codes(codes)

    def encode_query(self, sequence) -> np.ndarray:
        """Encode every length-``L`` window of a query ``sequence`` -> ``(n_windows, dim)``."""
        n = len(sequence) - self.L + 1
        if n <= 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        windows = [sequence[s:s + self.L] for s in range(n)]
        return self.encode_codes(codes_from_windows(windows, self.L))

    def train_inputs(self, fragment_index, rows) -> np.ndarray:
        """Per-fragment array consumed by :meth:`forward` during training (codes for sequence
        encoders; pooled features for context encoders)."""
        return np.asarray(fragment_index.ctx[np.asarray(rows), :self.L])


class BlosumCosineEncoder(Encoder):
    """v0: concatenated BLOSUM62 rows, L2-normalized.  Non-parametric, numpy-only."""

    version = "v0"

    def __init__(self, L=9):
        self.L = int(L)
        self.dim = self.L * N_STD

    def encode_codes(self, codes) -> np.ndarray:
        return _l2_normalize(blosum_features(codes))


def load_encoder(spec, L=9):
    """Resolve an encoder ``spec``: ``"v0"`` / a :class:`BlosumCosineEncoder`, or a path to saved
    learned weights (``.pt`` + meta) -> the matching learned encoder."""
    if isinstance(spec, Encoder):
        return spec
    if spec in (None, "v0", "blosum"):
        return BlosumCosineEncoder(L=L)
    from openawsem.memory.ml.encoder_torch import load_torch_encoder
    return load_torch_encoder(spec, L=L)
