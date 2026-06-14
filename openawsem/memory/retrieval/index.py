"""Compact binary index for fast fragment retrieval.

The plain :class:`~openawsem.memory.retrieval.database.FragmentDatabase` loads the whole
``residues.csv[.gz]`` into pandas with object-dtype context columns -- ~50 GB / minutes for an
11.5M-residue table.  :class:`FragmentIndex` builds (once, streaming in chunks so the build
itself stays light) a small set of numpy arrays:

* ``ctx``        int8  (N, C)  -- forward context encoded to AA indices (``-`` -> 127)
* ``dist``       float32 (N, S, 9) -- the nine directional pair distances per separation
* ``chain_code`` / ``internal_index`` / ``segment_id`` int32 (N,)
* ``chain_offset`` int64 (n_chains+1,) -- CSR row ranges per chain

Loading memmaps ``dist`` and keeps only ``ctx`` (a few hundred MB) resident, so even the full
PDB DB loads in seconds.  :meth:`search` and :meth:`get_fragment` return the same columns as
``FragmentDatabase`` (the ``fragment`` string is decoded from indices, so rare non-standard
letters read back as ``X``; scores and distances are identical).

Build:  ``python -m openawsem.memory.retrieval.index <out_dir>``
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .aa import AA_ALPHABET, AA_INDEX
from .blosum import SCORE_TABLE, UNKNOWN_INDEX, encode_sequence

SENTINEL = 127                       # context code for '-' (segment edge)
ATOM_PAIRS = [(x, y) for x in ("CA", "CB", "O") for y in ("CA", "CB", "O")]
INDEX_DIRNAME = "fragment_index"
_DECODE = np.array(list(AA_ALPHABET) + ["X"])    # code -> letter (0..20)


def _encode_ctx(arr: np.ndarray) -> np.ndarray:
    """(n, C) object array of one-letter codes / '-' -> int8 codes."""
    flat = arr.ravel()
    codes = np.full(flat.shape, UNKNOWN_INDEX, dtype=np.int8)   # default unknown 'X'
    for ch, co in AA_INDEX.items():
        codes[flat == ch] = co
    codes[flat == "-"] = SENTINEL
    return codes.reshape(arr.shape)


class FragmentIndex:
    """Array-backed drop-in for ``FragmentDatabase.search`` / ``get_fragment``."""

    def __init__(self, ctx, dist, chain_code, internal_index, segment_id, chain_offset,
                 chain_names, seps, max_forward):
        self.ctx = ctx
        self.dist = dist
        self.chain_code = chain_code
        self.internal_index = internal_index
        self.segment_id = segment_id
        self.chain_offset = chain_offset
        self.chain_names = list(chain_names)
        self.seps = list(seps)
        self.max_forward = int(max_forward)
        self._sep_index = {s: i for i, s in enumerate(self.seps)}
        self._chain_index = {c: i for i, c in enumerate(self.chain_names)}
        self._enc_cache: dict = {}
        self._calibration: dict = {}
        self._seed_cache: dict = {}

    # ----- build / load -------------------------------------------------- #
    @staticmethod
    def index_dir(out_dir: str) -> str:
        return os.path.join(out_dir, INDEX_DIRNAME)

    @staticmethod
    def exists(out_dir: str) -> bool:
        return os.path.isfile(os.path.join(FragmentIndex.index_dir(out_dir), "ctx.npy"))

    @classmethod
    def build(cls, out_dir: str, dest: str | None = None, chunksize: int = 500_000) -> str:
        path = os.path.join(out_dir, "residues.csv")
        if not os.path.exists(path):
            path = os.path.join(out_dir, "residues.csv.gz")
        header = pd.read_csv(path, nrows=0).columns
        fwd = sorted((int(c[len("ctx_p"):]) for c in header if c.startswith("ctx_p")))
        max_forward = max(fwd)
        ctx_cols = [f"ctx_p{k}" for k in range(max_forward + 1)]
        seps = sorted({int(c.split("_")[1][1:]) for c in header
                       if c.startswith("d_s")} - {0})
        seps = [s for s in seps if s <= max_forward]
        dist_cols = [f"d_s{s}_{X}_{Y}" for s in seps for (X, Y) in ATOM_PAIRS]
        usecols = ["chain_uid", "internal_index", "segment_id"] + ctx_cols + dist_cols
        str_cols = {c: str for c in ctx_cols + ["chain_uid"]}

        ctx_parts, dist_parts, chain_parts, iidx_parts, seg_parts = [], [], [], [], []
        n = 0
        for chunk in pd.read_csv(path, usecols=usecols, dtype=str_cols, chunksize=chunksize):
            ctx_parts.append(_encode_ctx(chunk[ctx_cols].to_numpy()))
            d = chunk[dist_cols].to_numpy(dtype=np.float32)
            dist_parts.append(d.reshape(len(chunk), len(seps), len(ATOM_PAIRS)))
            chain_parts.append(chunk["chain_uid"].to_numpy(dtype=object))
            iidx_parts.append(chunk["internal_index"].to_numpy(dtype=np.int32))
            seg_parts.append(chunk["segment_id"].to_numpy(dtype=np.int32))
            n += len(chunk)
            print(f"  indexed {n:,} residues", end="\r")
        print()
        ctx = np.concatenate(ctx_parts)
        dist = np.concatenate(dist_parts)
        chain_arr = np.concatenate(chain_parts)
        internal_index = np.concatenate(iidx_parts)
        segment_id = np.concatenate(seg_parts)
        chain_code, chain_names = pd.factorize(chain_arr)         # first-appearance order
        chain_code = chain_code.astype(np.int32)
        # rows are in file order (chain-grouped); CSR offsets per chain
        bnd = np.flatnonzero(np.diff(chain_code) != 0) + 1
        chain_offset = np.concatenate([[0], bnd, [len(chain_code)]]).astype(np.int64)

        dest = dest or cls.index_dir(out_dir)
        os.makedirs(dest, exist_ok=True)
        np.save(os.path.join(dest, "ctx.npy"), ctx)
        np.save(os.path.join(dest, "dist.npy"), dist)
        np.save(os.path.join(dest, "chain_code.npy"), chain_code)
        np.save(os.path.join(dest, "internal_index.npy"), internal_index)
        np.save(os.path.join(dest, "segment_id.npy"), segment_id)
        np.save(os.path.join(dest, "chain_offset.npy"), chain_offset)
        with open(os.path.join(dest, "meta.json"), "w") as f:
            json.dump({"chain_names": list(map(str, chain_names)), "seps": seps,
                       "atom_pairs": ["-".join(p) for p in ATOM_PAIRS],
                       "max_forward": max_forward, "n": int(len(ctx))}, f)
        print(f"wrote index for {len(ctx):,} residues -> {dest}")
        return dest

    @classmethod
    def load(cls, out_dir: str, dest: str | None = None) -> "FragmentIndex":
        dest = dest or cls.index_dir(out_dir)
        with open(os.path.join(dest, "meta.json")) as f:
            meta = json.load(f)
        L = lambda name, mmap=None: np.load(os.path.join(dest, name), mmap_mode=mmap)  # noqa: E731
        return cls(ctx=L("ctx.npy"), dist=L("dist.npy", "r"), chain_code=L("chain_code.npy"),
                   internal_index=L("internal_index.npy"), segment_id=L("segment_id.npy"),
                   chain_offset=L("chain_offset.npy"), chain_names=meta["chain_names"],
                   seps=meta["seps"], max_forward=meta["max_forward"])

    # ----- search (same result columns as FragmentDatabase.search) ------- #
    def _encoded_for_length(self, L: int):
        if L not in self._enc_cache:
            codes = self.ctx[:, :L]
            valid = np.flatnonzero((codes != SENTINEL).all(axis=1))
            self._enc_cache[L] = (valid, codes[valid].astype(np.int8))
        return self._enc_cache[L]

    _SEARCH_COLUMNS = ["chain_uid", "start", "fragment", "raw_score", "identity",
                       "positive_fraction", "E_value"]

    def search(self, query: str, L: int | None = None, top: int = 20, *, seeded=False,
               seed_word=2, min_candidates=None) -> pd.DataFrame:
        """Top-``top`` hits by BLOSUM62.  With ``seeded`` only fragments sharing an exact
        ``seed_word``-mer with the query (at some position) are scored — a BLAST-like seed
        step — falling back to the full scan when too few candidates are found, so the result
        is the exact top-N whenever recall could be at risk."""
        L = L or len(query)
        if L - 1 > self.max_forward:
            raise ValueError(f"L={L} exceeds stored forward context {self.max_forward + 1}")
        rows, enc = self._encoded_for_length(L)
        if len(rows) == 0:
            return pd.DataFrame(columns=self._SEARCH_COLUMNS)
        q = encode_sequence(query)
        cal = self._get_calibration(L, enc, len(rows))
        if seeded and L - seed_word + 1 >= 1:
            cand = self._candidates(q, L, seed_word)
            if len(cand) >= max(min_candidates or top, top):
                return self._format(cand, self.ctx[cand, :L].astype(np.int8), q, L, top, cal)
        return self._format(rows, enc, q, L, top, cal)

    def _format(self, cand_rows, codes, q, L, top, cal) -> pd.DataFrame:
        per_pos = SCORE_TABLE[q[None, :], codes]
        scores = per_pos.sum(axis=1)
        identity = (codes == q[None, :]).sum(axis=1) / L
        positive = (per_pos > 0).sum(axis=1) / L
        order = np.argsort(-scores, kind="stable")[:top]
        sel = cand_rows[order]
        return pd.DataFrame({
            "chain_uid": [self.chain_names[c] for c in self.chain_code[sel]],
            "start": self.internal_index[sel],
            "fragment": ["".join(_DECODE[codes[o]]) for o in order],
            "raw_score": scores[order],
            "identity": np.round(identity[order], 4),
            "positive_fraction": np.round(positive[order], 4),
            "E_value": self._evalue(scores[order], cal),
        }).reset_index(drop=True)

    # ----- BLAST-like k-mer seeding ------------------------------------- #
    def _seeds_for_length(self, L, w):
        """Per seed position, the valid length-L fragment rows sorted by their exact w-mer key."""
        key = (L, w)
        if key not in self._seed_cache:
            rows, enc = self._encoded_for_length(L)
            base = len(_DECODE)                       # 21 symbols (codes 0..20)
            seeds = []
            for p in range(L - w + 1):
                wkey = np.zeros(len(enc), dtype=np.int64)
                for k in range(w):
                    wkey = wkey * base + enc[:, p + k].astype(np.int64)
                o = np.argsort(wkey, kind="stable")
                seeds.append((wkey[o], rows[o]))
            self._seed_cache[key] = seeds
        return self._seed_cache[key]

    def _candidates(self, q, L, w) -> np.ndarray:
        """Union of fragment rows sharing an exact w-mer with ``q`` at any position."""
        base = len(_DECODE)
        parts = []
        for p, (skeys, sids) in enumerate(self._seeds_for_length(L, w)):
            qkey = 0
            for k in range(w):
                qkey = qkey * base + int(q[p + k])
            lo = np.searchsorted(skeys, qkey, side="left")
            hi = np.searchsorted(skeys, qkey, side="right")
            if hi > lo:
                parts.append(sids[lo:hi])
        if not parts:
            return np.empty(0, dtype=self.internal_index.dtype)
        return np.unique(np.concatenate(parts))

    def topk_pool(self, q, L, top, *, seed_word=2, keep_rows=None):
        """The ``top`` length-``L`` fragment rows most similar to encoded query ``q``, as
        ``(rows, codes, scores)`` (BLOSUM62, stable order).  Seeded by exact ``seed_word``-mers
        with a full-scan fallback when too few candidates are found.  ``keep_rows`` is an optional
        boolean mask over all rows (e.g. a brain_damage exclusion) applied before ranking.

        Shared primitive for the soft (score-weighted) ``local_db`` memory and its mutation-ΔE
        engine: both anchor a candidate pool to a window and weight it by score."""
        cand = self._candidates(q, L, seed_word)
        if keep_rows is not None and len(cand):
            cand = cand[keep_rows[cand]]
        if len(cand) < top:
            allrows, _ = self._encoded_for_length(L)
            cand = allrows[keep_rows[allrows]] if keep_rows is not None else allrows
        codes = self.ctx[cand, :L]
        scores = SCORE_TABLE[q[None, :], codes].sum(axis=1)
        order = np.argsort(-scores, kind="stable")[:top]
        return cand[order], codes[order].astype(np.int16), scores[order].astype(np.float64)

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

    # ----- coherent fragment access (same columns as get_fragment) ------- #
    def get_fragment(self, chain_uid: str, start: int, L: int) -> pd.DataFrame:
        c = self._chain_index.get(chain_uid)
        if c is None:
            return pd.DataFrame()
        lo, hi = int(self.chain_offset[c]), int(self.chain_offset[c + 1])
        iidx = self.internal_index[lo:hi]

        def rowpos(i):
            p = np.searchsorted(iidx, i)
            return lo + int(p) if p < len(iidx) and iidx[p] == i else None

        out = []
        for a in range(L):
            ri = rowpos(start + a)
            if ri is None:
                continue
            seg_i = self.segment_id[ri]
            res_i = _DECODE[self.ctx[ri, 0]]
            for b in range(a + 1, L):
                sep = b - a
                if sep > self.max_forward:
                    continue
                rj = rowpos(start + b)
                if rj is None or self.segment_id[rj] != seg_i:
                    continue
                d = self.dist[ri, self._sep_index[sep]]              # (9,) float32
                row = {"chain_uid": chain_uid, "internal_i": start + a, "internal_j": start + b,
                       "sep": sep, "res_i": str(res_i), "res_j": str(_DECODE[self.ctx[rj, 0]])}
                for (X, Y), val in zip(ATOM_PAIRS, d):
                    row[f"d_{X}_{Y}"] = float(val)
                out.append(row)
        return pd.DataFrame(out)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="python -m openawsem.memory.retrieval.index",
                                description="Build a compact binary fragment-retrieval index.")
    p.add_argument("out_dir", help="fragdb output dir containing residues.csv[.gz]")
    p.add_argument("--dest", default=None, help="index directory (default <out_dir>/fragment_index)")
    p.add_argument("--chunksize", type=int, default=500_000)
    args = p.parse_args(argv)
    FragmentIndex.build(args.out_dir, dest=args.dest, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
