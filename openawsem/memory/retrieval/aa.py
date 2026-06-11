"""Amino-acid alphabet for fragment retrieval (vendored from ``fragdb.aa``).

Only the ordered one-letter alphabet and its index are needed for BLOSUM scoring; the
database-build parent-resolution helpers are not part of retrieval.
"""
from __future__ import annotations

# Ordered one-letter alphabet used by the substitution matrix and the word index.
AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {a: i for i, a in enumerate(AA_ALPHABET)}
UNKNOWN = "X"
