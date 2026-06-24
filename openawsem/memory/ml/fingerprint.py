"""Structural fingerprints and the well-width-weighted structural distance.

A length-``L`` fragment is summarized by the vector of its in-window pairwise distances over all
nine directional atom-pair channels (CA/CB/O x CA/CB/O).  Because
:class:`~openawsem.memory.retrieval.index.FragmentIndex` already stores, per residue row, the
forward distances ``dist[row, sep, channel]`` (Angstrom, NaN past a segment edge), assembling a
fragment's fingerprint is an *indexing pass*: the in-window pair ``(a, b)`` of the fragment
starting at row ``r`` is read at ``dist[r + a, sep_index[b - a], :]``.

The fingerprint ``S`` and the distance ``d_struct`` defined here are the **training signal** for
the learned encoder; they are never needed at inference (the encoder sees only sequence).

``d_struct(a, b) = || w (S[a] - S[b]) ||`` with ``w_ij = 1 / sigma_ij`` and
``sigma_ij = well_width * sep**0.15`` -- distances measured in well-width units, matching the
fragment-memory energy's sensitivity.
"""
from __future__ import annotations

import numpy as np

#: nm per Angstrom (the index stores Angstrom; the energy/wells use nm).
_NM_PER_A = 0.1


def in_window_pairs(L, *, min_sep=1, max_sep=None):
    """Fixed ``(a, b, sep)`` layout of the in-window residue pairs of a length-``L`` fragment.

    Returns three int arrays of length ``P`` (the pair count), ordered by ``a`` then ``b`` so the
    layout is deterministic and shared by every fragment.  ``min_sep``/``max_sep`` clamp the
    sequence separation (``max_sep`` defaults to ``L - 1``).
    """
    hi = (L - 1) if max_sep is None else min(int(max_sep), L - 1)
    a_list, b_list, sep_list = [], [], []
    for a in range(L):
        for b in range(a + 1, L):
            sep = b - a
            if min_sep <= sep <= hi:
                a_list.append(a)
                b_list.append(b)
                sep_list.append(sep)
    return np.array(a_list), np.array(b_list), np.array(sep_list)


def assemble(index, rows, L, *, min_sep=1, max_sep=None):
    """Assemble fingerprints for fragment start ``rows`` of length ``L`` from ``index.dist``.

    ``rows`` should be *valid* length-``L`` start rows (forward context all in one segment; see
    :meth:`FragmentIndex._encoded_for_length`).  Returns ``(S, mask, layout)`` where ``S`` is
    ``(M, P, 9)`` distances in **nm**, ``mask`` is ``(M, P, 9)`` finite-entry flags, and
    ``layout`` is the ``(pair_a, pair_b, pair_sep)`` from :func:`in_window_pairs`.
    """
    rows = np.asarray(rows)
    pa, pb, ps = in_window_pairs(L, min_sep=min_sep, max_sep=max_sep)
    sep_pos = np.array([index._sep_index[int(s)] for s in ps])
    row_idx = rows[:, None] + pa[None, :]                       # (M, P)
    S = np.asarray(index.dist[row_idx, sep_pos[None, :], :], dtype=np.float32) * _NM_PER_A
    mask = np.isfinite(S)
    return S, mask, (pa, pb, ps)


def pair_weights(pair_sep, *, well_width=0.1):
    """Per-pair weight ``w = 1 / (well_width * sep**0.15)`` (the inverse well width)."""
    return 1.0 / (well_width * np.asarray(pair_sep, dtype=np.float64) ** 0.15)


def weighted_features(index, rows, L, *, well_width=0.1, min_sep=1, max_sep=None):
    """Flattened, well-width-weighted fingerprints ``F`` (``(M, P*9)`` float32) for ``rows``.

    ``F = w (S)`` with missing entries zero-filled, so plain L2 over ``F`` is :func:`d_struct`.
    This is the vector space the structural nearest-neighbor mining searches.
    """
    S, mask, (pa, pb, ps) = assemble(index, rows, L, min_sep=min_sep, max_sep=max_sep)
    w = pair_weights(ps, well_width=well_width).astype(np.float32)[None, :, None]
    F = np.where(mask, S, 0.0).astype(np.float32) * w
    return F.reshape(len(np.asarray(rows)), -1)


def d_struct(F_a, F_b):
    """Structural distance between weighted fingerprints (L2 over the last axis)."""
    return np.linalg.norm(np.asarray(F_a) - np.asarray(F_b), axis=-1)
