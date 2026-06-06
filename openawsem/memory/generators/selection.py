"""Deterministic selection of fragment hits (shared across structure backends).

A *hits* table is a DataFrame with at least ``query_start`` (the target window start,
1-based) and identity columns ``pdb_id``/``chain``/``subject_start``/``subject_end``;
optional ``query_end``, ``evalue``, ``bitscore``, ``weight``.

The point is reproducibility: repeated BLAST runs returned different hit *orders* (and the
legacy code relied on ``set`` iteration + thread completion order), so two runs could yield
different memories.  Here ranking is a fixed total order and "keep N per position" is
deterministic, so a given hit set always yields the same memory.  A backend that needs
method-specific filtering (e.g. BLAST homolog exclusion) adds it in its own module.
"""
from __future__ import annotations

import pandas as pd

#: identity columns that make a hit unique (used for de-duplication).
HIT_KEY = ["pdb_id", "chain", "subject_start", "subject_end", "query_start", "query_end"]

#: total-order ranking: (column, ascending). Lower evalue / higher bitscore first.
RANK_ORDER = [("query_start", True), ("evalue", True), ("bitscore", False),
              ("pdb_id", True), ("chain", True), ("subject_start", True)]


def deduplicate_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate hits (same pdb/chain/ranges), keeping the first occurrence.

    Rank first if you want "first" to mean "best": across PSI-BLAST iterations the same
    alignment recurs with different e-values, so :func:`select_n_per_position` ranks
    before de-duplicating to keep the best-scoring copy.
    """
    keys = [c for c in HIT_KEY if c in hits.columns]
    return hits.drop_duplicates(subset=keys).reset_index(drop=True)


def rank_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Sort hits into a deterministic total order (stable, independent of input order)."""
    cols = [(c, a) for c, a in RANK_ORDER if c in hits.columns]
    if not cols:
        return hits.reset_index(drop=True)
    by = [c for c, _ in cols]
    ascending = [a for _, a in cols]
    return hits.sort_values(by, ascending=ascending, kind="mergesort").reset_index(drop=True)


def select_n_per_position(hits: pd.DataFrame, n: int, *, position_col: str = "query_start",
                          report: bool = False):
    """Keep exactly ``n`` hits per target position where available, deterministically.

    Hits are de-duplicated and ranked first, then the top ``n`` per ``position_col`` are
    kept.  With ``report=True`` also returns a Series of positions that came up short
    (fewer than ``n`` hits).
    """
    # rank first so de-dup keeps the best-scoring copy of a hit recurring across iterations
    ranked = deduplicate_hits(rank_hits(hits))
    if position_col not in ranked.columns or len(ranked) == 0:
        kept = ranked
    else:
        kept = ranked.groupby(position_col, sort=True, group_keys=False).head(n).reset_index(drop=True)
    if report:
        if position_col in kept.columns and len(kept):
            counts = kept.groupby(position_col).size()
            shortfall = counts[counts < n]
        else:
            shortfall = pd.Series(dtype="int64")
        return kept, shortfall
    return kept
