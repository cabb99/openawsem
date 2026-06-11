"""BLAST-like single-sequence fragment retrieval over the residue-centric table (vendored
from ``fragdb.retrieval``).

Each residue row carries its local sequence context as per-position columns
(``ctx_p0..ctx_p{k}``) and the forward distances to its next neighbours.  Retrieval scans the
context columns for fragment starts; a hit's pairwise distances are reconstructed from L
consecutive rows.  No word index -- scoring is an exact vectorized BLOSUM62 scan.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .blosum import SCORE_TABLE, encode_sequence, score_table

# nine directional atom pairs (X on residue i, Y on residue i+sep), matching the build
ATOM_PAIRS = [(x, y) for x in ("CA", "CB", "O") for y in ("CA", "CB", "O")]


class FragmentDatabase:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        path = os.path.join(out_dir, "residues.csv")
        if not os.path.exists(path):
            path = os.path.join(out_dir, "residues.csv.gz")
        # context/identity columns are strings (letters or '-', never the pandas NA
        # tokens); distance columns stay numeric (empty -> NaN)
        ctx_cols = [c for c in pd.read_csv(path, nrows=0).columns if c.startswith("ctx_")]
        str_cols = {c: str for c in ctx_cols + ["parent1", "resname3"]}
        self.residues = pd.read_csv(path, dtype=str_cols, low_memory=False)
        self.context_offsets = self._parse_offsets(ctx_cols)
        self.max_forward = max(self.context_offsets)
        self._by_chain = None
        self._enc_cache: dict = {}
        self._calibration: dict = {}

    @classmethod
    def load(cls, out_dir: str) -> "FragmentDatabase":
        return cls(out_dir)

    @staticmethod
    def _parse_offsets(ctx_cols):
        offs = []
        for c in ctx_cols:
            tok = c[len("ctx_"):]
            offs.append(-int(tok[1:]) if tok.startswith("m") else int(tok[1:]))
        return sorted(offs)

    # -- coherent fragment access ------------------------------------------------
    def get_fragment(self, chain_uid: str, start: int, L: int) -> pd.DataFrame:
        """Reconstruct the C(L,2) pairwise distances of fragment [start, start+L-1].

        Returns one row per (i, j) with the nine directional atom-pair distances.
        """
        if self._by_chain is None:
            self._by_chain = {k: g.set_index("internal_index")
                              for k, g in self.residues.groupby("chain_uid")}
        rows = self._by_chain.get(chain_uid)
        if rows is None:
            return pd.DataFrame()
        out = []
        for a in range(L):
            i = start + a
            if i not in rows.index:
                continue
            ri = rows.loc[i]
            seg_i = ri["segment_id"]
            for b in range(a + 1, L):
                sep = b - a
                if sep > self.max_forward:
                    continue
                j = start + b
                if j not in rows.index or rows.loc[j]["segment_id"] != seg_i:
                    continue
                dist = {f"d_{X}_{Y}": ri.get(f"d_s{sep}_{X}_{Y}", "") for (X, Y) in ATOM_PAIRS}
                out.append({"chain_uid": chain_uid, "internal_i": i, "internal_j": j,
                            "sep": sep, "res_i": ri["parent1"], "res_j": rows.loc[j]["parent1"],
                            **dist})
        return pd.DataFrame(out)

    # -- BLAST-like search over context columns ---------------------------------
    def _encoded_for_length(self, L: int):
        if L not in self._enc_cache:
            cols = [f"ctx_p{k}" for k in range(L)]
            ctx = self.residues[cols].to_numpy()
            valid = (ctx != "-").all(axis=1)
            sub = self.residues[valid].reset_index(drop=True)
            seqs = ctx[valid]
            enc = np.stack([encode_sequence("".join(row)) for row in seqs]) \
                if len(seqs) else np.zeros((0, L), np.int8)
            self._enc_cache[L] = (sub, enc)
        return self._enc_cache[L]

    def search(self, query: str, L: int | None = None, top: int = 20,
               matrix=None) -> pd.DataFrame:
        L = L or len(query)
        if L - 1 > self.max_forward:
            raise ValueError(f"L={L} exceeds stored forward context {self.max_forward+1}")
        sub, enc = self._encoded_for_length(L)
        if len(sub) == 0:
            return pd.DataFrame(columns=["chain_uid", "start", "fragment", "raw_score",
                                         "identity", "positive_fraction", "E_value"])
        table = SCORE_TABLE if matrix is None else score_table(matrix)
        q = encode_sequence(query)
        per_pos = table[q[None, :], enc]
        scores = per_pos.sum(axis=1)
        identity = (enc == q[None, :]).sum(axis=1) / L
        positive = (per_pos > 0).sum(axis=1) / L
        order = np.argsort(-scores, kind="stable")[:top]
        ctx_cols = [f"ctx_p{k}" for k in range(L)]
        fragments = sub.iloc[order][ctx_cols].agg("".join, axis=1).to_numpy()
        cal = self._get_calibration(L, enc, len(sub))
        return pd.DataFrame({
            "chain_uid": sub["chain_uid"].to_numpy()[order],
            "start": sub["internal_index"].to_numpy()[order],
            "fragment": fragments,
            "raw_score": scores[order],
            "identity": np.round(identity[order], 4),
            "positive_fraction": np.round(positive[order], 4),
            "E_value": self._evalue(scores[order], cal),
        }).reset_index(drop=True)

    def _get_calibration(self, L, enc, n, n_samples=200_000, seed=0):
        if L in self._calibration:
            return self._calibration[L]
        rng = np.random.default_rng(seed)
        ia, ib = rng.integers(0, n, n_samples), rng.integers(0, n, n_samples)
        rand = SCORE_TABLE[enc[ia], enc[ib]].sum(axis=1)
        self._calibration[L] = {"sorted": np.sort(rand), "N": n}
        return self._calibration[L]

    def _evalue(self, scores, cal):
        sr = cal["sorted"]
        ge = len(sr) - np.searchsorted(sr, scores, side="left")
        return cal["N"] * np.maximum(ge, 1) / len(sr)

    def write_calibration(self, L: int, out_path: str | None = None) -> pd.DataFrame:
        sub, enc = self._encoded_for_length(L)
        cal = self._get_calibration(L, enc, len(sub))
        sr = cal["sorted"]
        raw = np.arange(int(sr.min()), int(sr.max()) + 1)
        ge = len(sr) - np.searchsorted(sr, raw, side="left")
        tail = np.maximum(ge, 1) / len(sr)
        df = pd.DataFrame({"raw_score": raw, "tail_prob": tail, "N": cal["N"],
                           "E_value": cal["N"] * tail})
        df.to_csv(out_path or os.path.join(self.out_dir,
                  f"score_calibration_L{L}_BLOSUM62.csv"), index=False)
        return df


def search(db: FragmentDatabase, query: str, **kw) -> pd.DataFrame:
    return db.search(query, **kw)


def get_fragment(db: FragmentDatabase, chain_uid: str, start: int, L: int) -> pd.DataFrame:
    return db.get_fragment(chain_uid, start, L)
