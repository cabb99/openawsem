"""Fast fragment-memory mutation energy on a fixed structure.

For a soft (score-weighted) ``local_db`` memory, :class:`MutationEnergy` computes

    ΔE(position, new_aa) = E_frag(structure; mutant memory) − E_frag(structure; WT memory)

on FIXED coordinates -- the memory changes with the sequence, the structure does not.  The
fragment-memory energy is ``-k_fm * sum_pairs sum_candidates w_c * exp(-(r_p - d_c)^2/2σ^2)``,
where ``r_p`` is the structure's CA/CB pair distance, ``d_c`` the candidate fragment's stored
distance, and ``w_c`` the candidate's softmax weight.  A point mutation only changes the windows
that cover ``position`` (the rest cancel in the difference), so ΔE is a small local update.

Three methods (selected via :meth:`LocalDB.mutation_engine`):

* ``"fulldb"`` (:class:`FullDBMutationEnergy`) -- the EXACT soft potential: evaluate every database
  fragment, not a pool.  A point mutation multiplies each fragment's softmax weight by a per-amino-acid
  factor, so ΔE reduces to an O(alphabet*L) dot product over per-position partial-sum tables built once
  in an O(N) precompute (needs a :class:`FragmentIndex`; ~minutes / GBs of RAM on a large DB).  Then
  ~tens of thousands of exact trial-ΔE/s; ``commit`` advances a walk in O(N) (~seconds/accept).  This is
  the accurate engine -- use it for a faithful Monte-Carlo.
* ``"anchored"`` (~thousands/s) -- the per-window candidate pool is fixed to the WT target; a
  mutation only reweights it via the one-position BLOSUM score delta.  No precompute, but APPROXIMATE:
  it cannot see fragments outside the pool, so it distorts the marginal (accept/reject-deciding) moves
  of a walk.  A fast fallback when the exact precompute/commit is unaffordable, not a faithful sampler.
* ``"faithful"`` (~few/s) -- re-search the covering windows for the mutant (the true mutant soft
  memory, pool re-selected); exact and needs no precompute, but slow.  Largely superseded by ``"fulldb"``.

``dE`` evaluates a single mutation against the engine's current sequence (initially the WT);
``commit`` applies a mutation permanently so a Monte-Carlo walk can accumulate moves.
"""
from __future__ import annotations

import logging

import numpy as np

from openawsem.memory.memory import FragmentMemory
from openawsem.memory.retrieval.blosum import SCORE_TABLE, encode_sequence
from openawsem.memory.retrieval.index import ATOM_PAIRS, FragmentIndex

logger = logging.getLogger(__name__)

_CACB = (("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB"))
GLY = int(encode_sequence("G")[0])
ALPHA = SCORE_TABLE.shape[0]                       # 21 (20 AAs + X)


class MutationEnergy:
    """Mutation ΔE of a soft ``local_db`` memory on a fixed ``structure`` (see module docstring).

    Build via :meth:`LocalDB.mutation_engine`; then ``engine.dE(position, new_aa)`` (1-based
    residue, one-letter amino acid).  ``method`` is ``"anchored"`` (default, fast) or
    ``"faithful"`` (re-search, exact to the regenerated mutant memory)."""

    def __init__(self, ldb, sequence, structure, *, method="anchored", k_fm=0.04184):
        if method not in ("anchored", "faithful"):
            raise ValueError(f"method must be 'anchored' or 'faithful', got {method!r}")
        self._init_common(ldb, sequence, structure, k_fm)
        self.method = method
        if method == "anchored":
            self.pools = [self._anchored_pool(s) for s in range(len(self.terms))]
            self.E_wt = [self._anchored_energy(s, self._window(self.seq, s))
                         for s in range(len(self.terms))]
        else:
            self.E_wt = [self._faithful_energy(s, self._window(self.seq, s))
                         for s in range(len(self.terms))]

    # ----- geometry: structure pair distances + masks, once -------------- #
    def _init_common(self, ldb, sequence, structure, k_fm):
        """Engine-wide setup shared by every method: copy the ``ldb`` parameters and build the fixed
        per-window structure geometry (``self.terms``)."""
        if not isinstance(sequence, str):
            raise ValueError("MutationEnergy supports a single chain (a sequence string)")
        idx = ldb.db
        if not isinstance(idx, FragmentIndex):
            raise TypeError("soft mutation ΔE needs a FragmentIndex; build one with "
                            "`python -m openawsem.memory.retrieval.index <database>`")
        self.idx = idx
        self.seq = sequence
        self.k_fm = float(k_fm)
        self.L = ldb.fragment_length
        self.minsep, self.maxsep = ldb.min_seq_sep, ldb.max_seq_sep
        self.well_width = ldb.well_width
        self.scale = ldb.n_mem
        self.temp = float(ldb.soft_temp)
        self.pool = int(ldb.soft_pool)
        self.seed_word = int(ldb.soft_seed_word)
        self.sep_index = idx._sep_index
        self.apidx = {p: ATOM_PAIRS.index(p) for p in _CACB}
        coords = FragmentMemory._structure_coords(structure)
        self.terms = [self._window_terms(s, coords) for s in range(len(sequence) - self.L + 1)]
    def _window(self, seq, s):
        return seq[s:s + self.L]

    def _window_terms(self, s, coords):
        """Per-pair structure geometry for window ``s``: which (atom_pair, separation) terms
        exist (both atoms present in the structure), their distance ``R`` and width ``SIG`` (nm),
        the residue offset ``A`` into the fragment, and the GLY-mask offsets/flags."""
        A, SEPI, API, R, SIG, AOFF, BOFF, NEEDA, NEEDB = ([] for _ in range(9))
        for a in range(self.L):
            for b in range(a + 1, self.L):
                sep = b - a
                if not (self.minsep <= sep <= self.maxsep) or sep not in self.sep_index:
                    continue
                ri, rj = s + a + 1, s + b + 1               # 1-based target resids (single chain)
                sig = self.well_width * sep ** 0.15
                for (X, Y) in _CACB:
                    ci, cj = coords.get((X, ri)), coords.get((Y, rj))
                    if ci is None or cj is None:
                        continue                              # structure lacks the atom (e.g. GLY CB)
                    A.append(a); SEPI.append(self.sep_index[sep]); API.append(self.apidx[(X, Y)])
                    R.append(float(np.linalg.norm(np.asarray(ci) - np.asarray(cj))))
                    SIG.append(sig); AOFF.append(a); BOFF.append(b)
                    NEEDA.append(X == "CB"); NEEDB.append(Y == "CB")
        return {"A": np.array(A, int), "SEPI": np.array(SEPI, int), "API": np.array(API, int),
                "R": np.array(R), "SIG": np.array(SIG), "AOFF": np.array(AOFF, int),
                "BOFF": np.array(BOFF, int), "NEEDA": np.array(NEEDA, bool),
                "NEEDB": np.array(NEEDB, bool)}

    def _gaussians(self, s, rows):
        """``G[candidate, term] = exp(-(R - d)^2 / 2σ^2)`` (float64) for window ``s``'s structure terms,
        with missing fragment distances (NaN) contributing zero.  ``FullDBMutationEnergy`` uses its own
        float32 variant for the whole-database gather."""
        t = self.terms[s]
        if len(t["A"]) == 0 or len(rows) == 0:
            return np.zeros((len(rows), len(t["A"])))
        d = self.idx.dist[rows[:, None] + t["A"][None, :], t["SEPI"][None, :], t["API"][None, :]] / 10.0
        valid = ~np.isnan(d)
        return np.where(valid, np.exp(-((t["R"][None, :] - d) ** 2) / (2.0 * t["SIG"][None, :] ** 2)), 0.0)

    def _gly_mask(self, s, window):
        t = self.terms[s]
        glyA = np.fromiter((window[o] == "G" for o in t["AOFF"]), bool, len(t["AOFF"]))
        glyB = np.fromiter((window[o] == "G" for o in t["BOFF"]), bool, len(t["BOFF"]))
        return ~((t["NEEDA"] & glyA) | (t["NEEDB"] & glyB))

    def _weights(self, scores):
        w = np.exp((scores - scores.max()) / self.temp)
        return self.scale * w / w.sum()

    # ----- anchored: fixed WT pool, reweight ----------------------------- #
    def _anchored_pool(self, s):
        q = encode_sequence(self._window(self.seq, s))
        rows, codes, scores = self.idx.topk_pool(q, self.L, self.pool, seed_word=self.seed_word)
        return [codes, scores, self._gaussians(s, rows)]

    def _anchored_energy(self, s, window):
        codes, scores, G = self.pools[s]
        if G.shape[1] == 0:
            return 0.0
        return -self.k_fm * float(((self._weights(scores) @ G) * self._gly_mask(s, window)).sum())

    # ----- faithful: re-search the window for the mutant ----------------- #
    def _faithful_energy(self, s, window):
        t = self.terms[s]
        if len(t["A"]) == 0:
            return 0.0
        q = encode_sequence(window)
        rows, _, scores = self.idx.topk_pool(q, self.L, self.pool, seed_word=self.seed_word)
        G = self._gaussians(s, rows)
        return -self.k_fm * float(((self._weights(scores) @ G) * self._gly_mask(s, window)).sum())

    # ----- public ΔE / commit -------------------------------------------- #
    def _covering_windows(self, position):
        kk, n = position - 1, len(self.seq)
        return kk, range(max(0, kk - self.L + 1), min(kk, n - self.L) + 1)

    def _mutant_window(self, s, kk, new_aa):
        off = kk - s
        win = self._window(self.seq, s)
        return off, win[:off] + new_aa + win[off + 1:]

    def _apply_reweight(self, s, off, new_code, old_code):
        """Add the one-position BLOSUM score delta into pool ``s``'s stored scores; return the originals
        (so a non-committing ``dE`` can restore them)."""
        codes, scores, _ = self.pools[s]
        col = codes[:, off]
        self.pools[s][1] = scores + (SCORE_TABLE[new_code, col] - SCORE_TABLE[old_code, col])
        return scores

    def dE(self, position, new_aa) -> float:
        """ΔE of mutating residue ``position`` (1-based) to ``new_aa`` against the current
        sequence.  Zero for a no-op (same residue) or a position outside every well's window."""
        kk, windows = self._covering_windows(position)
        new_code, old_code = int(encode_sequence(new_aa)[0]), int(encode_sequence(self.seq[kk])[0])
        de = 0.0
        for s in windows:
            off, mut = self._mutant_window(s, kk, new_aa)
            if self.method == "anchored":
                scores = self._apply_reweight(s, off, new_code, old_code)
                de += self._anchored_energy(s, mut) - self.E_wt[s]
                self.pools[s][1] = scores                 # restore (single-mutation eval)
            else:
                de += self._faithful_energy(s, mut) - self.E_wt[s]
        return de

    def commit(self, position, new_aa) -> None:
        """Apply a mutation permanently so subsequent ``dE`` calls are relative to it."""
        kk, windows = self._covering_windows(position)
        new_code, old_code = int(encode_sequence(new_aa)[0]), int(encode_sequence(self.seq[kk])[0])
        if self.method == "anchored":
            for s in windows:
                self._apply_reweight(s, kk - s, new_code, old_code)
        self.seq = self.seq[:kk] + new_aa + self.seq[kk + 1:]
        for s in windows:
            self.E_wt[s] = (self._anchored_energy(s, self._window(self.seq, s)) if self.method == "anchored"
                            else self._faithful_energy(s, self._window(self.seq, s)))


class FullDBMutationEnergy(MutationEnergy):
    """Exact soft mutation ΔE over the WHOLE database (no pool truncation) — see module docstring.

    A point mutation multiplies every fragment's weight by ``g[a]=exp((B[new,a]-B[old,a])/T)`` keyed by
    its amino acid at the mutated offset, so the per-window energy is a ratio of partial sums grouped by
    amino acid: ``E_s = -k_fm*scale*(Σ_a g[a]·S1[off,a]) / (Σ_a g[a]·S0[off,a])``.  The ``S0/S1`` tables
    (and a GLY ``Gother/Gtouch`` split) are built once over all N fragments (O(N) precompute).  ``dE`` is
    then O(alphabet*L); ``commit`` re-mixes the weights and rebuilds the tables in O(N) (with the
    structure-fixed Gaussians cached, so no memmap re-gather)."""

    def __init__(self, ldb, sequence, structure, *, k_fm=0.04184, cache_commit=True):
        self._init_common(ldb, sequence, structure, k_fm)
        self.method = "fulldb"
        self.cache_commit = bool(cache_commit)
        self.valid_rows = self.idx._encoded_for_length(self.L)[0]
        self.codes = np.asarray(self.idx.ctx[self.valid_rows, :self.L])     # (N, L) int8
        nwin = len(self.terms)
        logger.info("fulldb precompute over %d fragments x %d windows ...", len(self.valid_rows), nwin)
        self.W = []
        for s in range(nwin):
            self.W.append(self._precompute_window(s))
            if s % 8 == 0 or s == nwin - 1:
                logger.debug("  fulldb precompute window %d/%d", s + 1, nwin)
        self.E_wt = [self._wt_energy(s) for s in range(nwin)]

    # ----- per-window precompute (structure-fixed Gaussians + per-aa tables) ----- #
    def _pool_gaussians_f32(self, s, rows):
        """(M, T) float32 Gaussians of fragments ``rows`` for window ``s``'s terms; NaN distances -> 0.
        float32 (vs the base float64 ``_gaussians``) to halve the exp cost over the whole database."""
        t = self.terms[s]
        d = (self.idx.dist[rows[:, None] + t["A"][None, :],
                           t["SEPI"][None, :], t["API"][None, :]] * np.float32(0.1))   # (M, T) nm
        R = t["R"].astype(np.float32)[None, :]
        inv = (0.5 / t["SIG"].astype(np.float32) ** 2)[None, :]
        G = np.exp(-((R - d) ** 2) * inv, dtype=np.float32)
        G[np.isnan(d)] = 0.0
        return G

    def _selection_matrix(self, s, window):
        """(T, 2L) matrix M: column [2*off]=Gother(off) mask (no required CB at ``off``, WT-included),
        [2*off+1]=Gtouch(off) mask (a required CB at ``off``, pre-masked by the OTHER end's WT GLY-state).
        ``G @ M`` yields all per-offset Gother/Gtouch contributions at once."""
        t = self.terms[s]
        T = len(t["A"])
        include = self._gly_mask(s, window)
        AOFF, BOFF, NEEDA, NEEDB = t["AOFF"], t["BOFF"], t["NEEDA"], t["NEEDB"]
        glyA = np.fromiter((window[o] == "G" for o in AOFF), bool, T)
        glyB = np.fromiter((window[o] == "G" for o in BOFF), bool, T)
        M = np.zeros((T, 2 * self.L), dtype=np.float32)
        for off in range(self.L):
            a_touch = NEEDA & (AOFF == off)
            b_touch = NEEDB & (BOFF == off)
            touch = a_touch | b_touch
            other_ok = np.ones(T, bool)
            other_ok[a_touch] = ~(NEEDB[a_touch] & glyB[a_touch])
            other_ok[b_touch] = ~(NEEDA[b_touch] & glyA[b_touch])
            M[include & ~touch, 2 * off] = 1.0
            M[touch & other_ok, 2 * off + 1] = 1.0
        return M

    def _build_pool(self, s, rows, codes, window):
        """Structure-fixed part of a window: BLOSUM ``score`` vs ``window`` and the ``GM`` (M, 2L)
        Gother/Gtouch matrix.  Sequence-independent given (rows, window-GLY)."""
        q = encode_sequence(window)
        score = SCORE_TABLE[q[None, :], codes].sum(axis=1).astype(np.float64)
        if len(self.terms[s]["A"]) == 0:
            return {"score": score, "GM": np.zeros((len(rows), 2 * self.L))}
        GM = (self._pool_gaussians_f32(s, rows) @ self._selection_matrix(s, window)).astype(np.float64)
        return {"score": score, "GM": GM}

    def _tables_from(self, codes, e, GM):
        """Bincount the weighted Gother/Gtouch by amino acid at each offset (the S0/S1 tables)."""
        tables = {}
        for off in range(self.L):
            col = codes[:, off]
            S0 = np.bincount(col, weights=e, minlength=ALPHA)[:ALPHA]
            S1o = np.bincount(col, weights=e * GM[:, 2 * off], minlength=ALPHA)[:ALPHA]
            S1t = np.bincount(col, weights=e * GM[:, 2 * off + 1], minlength=ALPHA)[:ALPHA]
            tables[off] = (S0, S1o, S1t)
        return tables

    def _precompute_window(self, s):
        pool = self._build_pool(s, self.valid_rows, self.codes, self._window(self.seq, s))
        score, GM = pool["score"], pool["GM"]
        m = float(score.max()) if len(score) else 0.0
        e = np.exp((score - m) / self.temp)
        rec = {"e_sum": float(e.sum()), "tables": self._tables_from(self.codes, e, GM)}
        if self.cache_commit:                              # walk: keep scores (incremental reweight) +
            rec["score"], rec["GM"] = score, GM            # the structure-fixed Gaussians (no re-gather)
        return rec

    def _wt_energy(self, s):
        """WT window energy from the tables (any offset's split reproduces the full Gsum)."""
        w = self.W[s]
        S0, S1o, S1t = w["tables"][0]
        old_gly = self.seq[s] == "G"
        num = S1o.sum() + (0.0 if old_gly else S1t.sum())
        den = w["e_sum"]
        return 0.0 if den == 0 else -self.k_fm * self.scale * num / den

    # ----- ΔE ------------------------------------------------------------ #
    def dE(self, position, new_aa) -> float:
        """Exact ΔE of mutating ``position`` (1-based) to ``new_aa`` against the current sequence."""
        kk, windows = self._covering_windows(position)
        new_code, old_code = int(encode_sequence(new_aa)[0]), int(encode_sequence(self.seq[kk])[0])
        new_gly = new_code == GLY
        g = np.exp((SCORE_TABLE[new_code] - SCORE_TABLE[old_code]) / self.temp)[:ALPHA]
        de = 0.0
        for s in windows:
            off = kk - s
            S0, S1o, S1t = self.W[s]["tables"][off]
            den = float(g @ S0)
            if den == 0:
                continue
            num = float(g @ S1o) + (0.0 if new_gly else float(g @ S1t))
            de += (-self.k_fm * self.scale * num / den) - self.E_wt[s]
        return de

    def dE_naive(self, position, new_aa) -> float:
        """Brute O(N) reference (re-softmax over all fragments); test oracle for ``dE``.  Needs the
        per-window scores, i.e. ``cache_commit=True``."""
        if not self.cache_commit:
            raise RuntimeError("dE_naive needs cache_commit=True (per-window scores are not retained)")
        kk, windows = self._covering_windows(position)
        new_code, old_code = int(encode_sequence(new_aa)[0]), int(encode_sequence(self.seq[kk])[0])
        de = 0.0
        for s in windows:
            off = kk - s
            t = self.terms[s]
            if len(t["A"]) == 0:
                continue
            mut = self._mutant_window(s, kk, new_aa)[1]
            score_mut = self.W[s]["score"] + (SCORE_TABLE[new_code, self.codes[:, off]]
                                              - SCORE_TABLE[old_code, self.codes[:, off]])
            e = np.exp((score_mut - score_mut.max()) / self.temp)
            G = self._pool_gaussians_f32(s, self.valid_rows)
            include = self._gly_mask(s, mut)
            gsum = G[:, include].astype(np.float64).sum(axis=1) if include.any() else np.zeros(len(e))
            de += -self.k_fm * self.scale * float(e @ gsum) / float(e.sum()) - self.E_wt[s]
        return de

    # ----- commit (walk) ------------------------------------------------- #
    def commit(self, position, new_aa) -> None:
        """Apply a mutation permanently.  With ``cache_commit`` the structure-fixed Gaussians (GM) are
        reused: only the weights (via the one-position score delta) and the bincount tables are rebuilt
        -- no memmap re-gather.  GM is rebuilt only for a window whose WT GLY-state flips."""
        if not self.cache_commit:
            return self._commit_slow(position, new_aa)
        kk, windows = self._covering_windows(position)
        old_letter = self.seq[kk]
        new_code, old_code = int(encode_sequence(new_aa)[0]), int(encode_sequence(old_letter)[0])
        gly_flip = (new_aa == "G") != (old_letter == "G")
        self.seq = self.seq[:kk] + new_aa + self.seq[kk + 1:]
        for s in windows:
            off = kk - s
            w = self.W[s]
            w["score"] = w["score"] + (SCORE_TABLE[new_code, self.codes[:, off]]
                                       - SCORE_TABLE[old_code, self.codes[:, off]])
            if gly_flip:
                w["GM"] = self._build_pool(s, self.valid_rows, self.codes,
                                           self._window(self.seq, s))["GM"]
            e = np.exp((w["score"] - w["score"].max()) / self.temp)
            w["e_sum"] = float(e.sum())
            w["tables"] = self._tables_from(self.codes, e, w["GM"])
        for s in windows:
            self.E_wt[s] = self._wt_energy(s)

    def _commit_slow(self, position, new_aa) -> None:
        kk, windows = self._covering_windows(position)
        self.seq = self.seq[:kk] + new_aa + self.seq[kk + 1:]
        for s in windows:
            self.W[s] = self._precompute_window(s)
        for s in windows:
            self.E_wt[s] = self._wt_energy(s)
