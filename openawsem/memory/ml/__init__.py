"""Learned fragment-retrieval embedding (the ``ml`` backend's machinery).

A 9-mer is encoded to an L2-normalized vector ``z``; fragments are then retrieved by cosine
similarity (an ANN index) instead of by BLOSUM sequence search, so retrieval can pull in
structurally-similar fragments whose sequences are far apart -- coverage past BLOSUM.

This package holds the encoder-agnostic machinery:

* :mod:`~openawsem.memory.ml.fingerprint` -- assemble the structural fingerprint ``S`` and the
  well-width-weighted distance ``d_struct`` used as the *training signal* (never at inference).
* :mod:`~openawsem.memory.ml.encoder` -- the encoders (v0 BLOSUM-cosine, v1 learned MLP, ...).
* :mod:`~openawsem.memory.ml.train` -- mining + metric-loss training.

The ANN index itself lives in :mod:`openawsem.memory.retrieval.embedding`, and the user-facing
backend in :mod:`openawsem.memory.generators.ml`.
"""
from __future__ import annotations

from openawsem.memory.ml import fingerprint

__all__ = ["fingerprint"]
