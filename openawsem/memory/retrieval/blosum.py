"""BLOSUM62 substitution matrix and an integer-indexed score table (vendored from
``fragdb.blosum``).

The score table is laid out over the 20-letter alphabet in :data:`aa.AA_ALPHABET` plus a
trailing row/column for the unknown symbol ``X`` (index 20), so a window encoded as ``int8``
indices is scored by a single vectorized gather.
"""
from __future__ import annotations

import numpy as np

from .aa import AA_ALPHABET, AA_INDEX, UNKNOWN

# Canonical NCBI BLOSUM62 over: A R N D C Q E G H I L K M F P S T W Y V B Z X *
_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYVBZX*"
_BLOSUM62_RAW = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
-2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
-1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
 0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
-4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
"""


def _parse_full_matrix() -> dict:
    rows = [r for r in _BLOSUM62_RAW.strip().splitlines()]
    values = [list(map(int, r.split())) for r in rows]
    matrix: dict = {}
    for a, row in zip(_BLOSUM62_ORDER, values):
        for b, v in zip(_BLOSUM62_ORDER, row):
            matrix[(a, b)] = v
    return matrix


BLOSUM62 = _parse_full_matrix()

# Index N = 20 standard letters + 1 unknown slot.
N_SYMBOLS = len(AA_ALPHABET) + 1
UNKNOWN_INDEX = len(AA_ALPHABET)


def score_table(matrix: dict | None = None) -> np.ndarray:
    """Build a (N_SYMBOLS x N_SYMBOLS) int score table aligned to AA_ALPHABET + X."""
    matrix = matrix or BLOSUM62
    table = np.zeros((N_SYMBOLS, N_SYMBOLS), dtype=np.int16)
    for a in AA_ALPHABET:
        for b in AA_ALPHABET:
            table[AA_INDEX[a], AA_INDEX[b]] = matrix[(a, b)]
    for a in AA_ALPHABET:
        table[AA_INDEX[a], UNKNOWN_INDEX] = matrix.get((a, "X"), -1)
        table[UNKNOWN_INDEX, AA_INDEX[a]] = matrix.get(("X", a), -1)
    table[UNKNOWN_INDEX, UNKNOWN_INDEX] = matrix.get(("X", "X"), -1)
    return table


def encode_sequence(seq: str) -> np.ndarray:
    """One-letter string -> int8 indices into the score table (X for unknown)."""
    out = np.empty(len(seq), dtype=np.int8)
    for i, ch in enumerate(seq):
        out[i] = AA_INDEX.get(ch, UNKNOWN_INDEX)
    return out


SCORE_TABLE = score_table()
