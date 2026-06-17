"""``mc_fragments`` backend: precomputed energy table for fast Monte-Carlo mutation ΔE.

The key idea is to separate the structural geometry evaluation from the sequence weighting.
Given a fixed target structure, we precompute once:

    E_table[w, f] = Σ_{(k, sep, pair)} exp(-(r_struct_w[k,sep,pair] - d_db[f+k, sep, pair])² / 2σ²)

where ``w`` indexes structural windows of the target and ``f`` indexes database fragments.
This table is sequence-independent and computed from the structure coordinates (once, up-front).

For a soft (score-weighted) energy:

    E[w] = scale * Σ_f w_f * E_table[w, f]  where  w_f = softmax(BLOSUM_score_f / T)

During Monte-Carlo, a point mutation at position ``p`` only changes the softmax weights for
windows covering ``p`` (at most ``L`` windows).  The ΔE reduces to:

    ΔE[w] = scale * (Σ_f new_w_f * E_table[w, f]) − scale * (Σ_f old_w_f * E_table[w, f])

Because the new weight of fragment ``f`` is just the old weight multiplied by a per-AA factor
``g[aa_f_at_off] = exp((B[new, aa_f] - B[old, aa_f]) / T)``, this dot product collapses to a
21-way sum via bincount tables:

    S0[w, off, a] = Σ_{f: ctx[f,off]=a}  w_f           (sum of weights per AA at offset off)
    S1[w, off, a] = Σ_{f: ctx[f,off]=a}  w_f * E[w,f]  (weighted energy sum per AA at offset off)

Then:  ΔE[w] = scale * (g @ S1[w,off]) / (g @ S0[w,off]) − E_wt[w]

``dE`` is O(ALPHA × L) — essentially free.  ``commit`` rebuilds S0/S1 for affected windows
in O(N_db × L) (numpy bincount + E_table dot).  Phase 3 ports the bincount tables to CUDA for
parallel per-window commit (each window is independent).

Phase 1 (this file): CPU NumPy prototype.
Phase 2: validated against :class:`FullDBMutationEnergy` and folding tests.
Phase 3: PyTorch/CUDA acceleration for the commit step.
"""
from __future__ import annotations

import logging

import numpy as np

from openawsem.memory.generators.base import FragmentBackend, Registry
from openawsem.memory.retrieval.blosum import SCORE_TABLE, encode_sequence
from openawsem.memory.retrieval.index import FragmentIndex
from openawsem.memory.memory import FragmentMemory

logger = logging.getLogger(__name__)

_CACB = (("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB"))
ALPHA = SCORE_TABLE.shape[0]   # 21 (20 AAs + unknown X)


@Registry.register("mc_fragments")
class MCFragments(FragmentBackend):
    """Precomputed-table fragment backend for Monte-Carlo mutation ΔE.

    Parameters
    ----------
    database:
        Path to a ``fragdb`` output directory with a built ``fragment_index/``.
    fragment_length:
        Sliding-window length (default 9).
    well_width:
        Gaussian well width parameter (σ = well_width * sep^0.15, nm).
    soft_temp:
        Softmax temperature for BLOSUM score weighting.
    n_mem:
        Scale of the softmax weights (sum normalized to this value).
    min_seq_sep, max_seq_sep:
        Sequence separation range for structural pairs.
    k_fm:
        Fragment-memory force constant (kJ/mol).
    chunk_size:
        Number of DB fragments processed per chunk during table build.
    """

    def __init__(self, database, *, fragment_length=9, well_width=0.1, soft_temp=2.0,
                 n_mem=20, min_seq_sep=3, max_seq_sep=9, k_fm=0.04184, chunk_size=50_000,
                 soft_pool=80, soft_seed_word=2):
        self.database = str(database)
        self.fragment_length = int(fragment_length)
        self.well_width = float(well_width)
        self.soft_temp = float(soft_temp)
        self.n_mem = int(n_mem)
        self.min_seq_sep = int(min_seq_sep)
        self.max_seq_sep = int(max_seq_sep)
        self.k_fm = float(k_fm)
        self.chunk_size = int(chunk_size)
        self.soft_pool = int(soft_pool)
        self.soft_seed_word = int(soft_seed_word)
        self._db_cache: dict = {}

    @property
    def db(self) -> FragmentIndex:
        if self.database not in self._db_cache:
            if not FragmentIndex.exists(self.database):
                raise FileNotFoundError(
                    f"No fragment_index found at {self.database!r}. "
                    "Build one with: python -m openawsem.memory.retrieval.index <db_dir>")
            logger.info("loading fragment index from %s", self.database)
            self._db_cache[self.database] = FragmentIndex.load(self.database)
        return self._db_cache[self.database]

    def generate(self, sequence):
        """Delegate to LocalDB soft for the AWSEM force term."""
        from openawsem.memory.generators.local_db import LocalDB
        return LocalDB(self.database, weighting="soft", n_mem=self.n_mem,
                       fragment_length=self.fragment_length, well_width=self.well_width,
                       soft_temp=self.soft_temp, soft_pool=self.soft_pool,
                       soft_seed_word=self.soft_seed_word,
                       min_seq_sep=self.min_seq_sep,
                       max_seq_sep=self.max_seq_sep).generate(sequence)

    def mutation_engine(self, sequence, structure, *, k_fm=None) -> "MCMutationEngine":
        """Build a precomputed-table mutation engine for ``sequence`` on fixed ``structure``.

        The returned :class:`MCMutationEngine` exposes ``dE(position, new_aa)`` and
        ``commit(position, new_aa)`` with the same interface as
        :class:`~openawsem.memory.mutation.FullDBMutationEnergy`.
        """
        return MCMutationEngine(self, sequence, structure, k_fm=k_fm or self.k_fm)


class MCMutationEngine:
    """Precomputed-table mutation engine.

    Stored tables after ``__init__``:
    - ``E_table[N_win, N_db]`` float32 — structural Gaussian energy per (window, fragment)
    - ``scores[N_win, N_db]`` int16   — BLOSUM62 scores for the current sequence
    - ``S0[N_win, L, ALPHA]`` float64 — weight sum per (window, offset, AA)
    - ``S1[N_win, L, ALPHA]`` float64 — weighted energy sum per (window, offset, AA)
    - ``E_wt[N_win]`` float64         — current energy per window

    ``dE`` is O(ALPHA × L): one 21-element dot product per affected window (~9).
    ``commit`` is O(N_db × L): numpy bincount over E_table for each affected window.
    """

    def __init__(self, gen: MCFragments, sequence: str, structure, *, k_fm: float = 0.04184):
        if not isinstance(sequence, str):
            raise ValueError("MCMutationEngine supports a single chain (a sequence string)")
        idx = gen.db
        if not isinstance(idx, FragmentIndex):
            raise TypeError("MCMutationEngine needs a FragmentIndex; "
                            "build one with: python -m openawsem.memory.retrieval.index <db>")

        self.idx = idx
        self.seq = list(sequence)
        self.k_fm = float(k_fm)
        self.L = gen.fragment_length
        self.minsep = gen.min_seq_sep
        self.maxsep = gen.max_seq_sep
        self.well_width = gen.well_width
        self.scale = float(gen.n_mem)
        self.temp = gen.soft_temp
        self.chunk_size = gen.chunk_size

        coords = FragmentMemory._structure_coords(structure)
        n = len(sequence)
        self.terms = [self._window_terms(s, coords) for s in range(n - self.L + 1)]

        self.valid_rows, self.valid_ctx = idx._encoded_for_length(self.L)
        N_frags = len(self.valid_rows)
        N_windows = len(self.terms)

        # Transpose ctx to (L, N_frags) so ctx_T[off] is a contiguous row.
        # int32 is np.intp-compatible on 64-bit systems and halves memory vs int64.
        self._ctx_T = np.ascontiguousarray(self.valid_ctx.T).astype(np.int32)  # (L, N_frags)

        logger.info("building E_table: %d fragments × %d windows ...", N_frags, N_windows)
        self.E_table = self._build_table(N_frags, N_windows)
        logger.info("building score table and S0/S1 bincount tables ...")
        self.scores = self._build_scores(sequence, N_frags, N_windows)
        self.S0 = np.zeros((N_windows, self.L, ALPHA), dtype=np.float64)
        self.S1 = np.zeros((N_windows, self.L, ALPHA), dtype=np.float64)
        self.E_wt = np.zeros(N_windows, dtype=np.float64)
        for s in range(N_windows):
            self._rebuild_window_tables(s)
        logger.info("MCMutationEngine ready — E_table %.0f MB, scores %.0f MB, "
                    "S0/S1 %.0f KB",
                    self.E_table.nbytes / 1e6, self.scores.nbytes / 1e6,
                    (self.S0.nbytes + self.S1.nbytes) / 1e3)

    # ----- structural geometry ------------------------------------------- #

    def _window_terms(self, s, coords):
        """Per-pair structural geometry for window ``s``."""
        from openawsem.memory.retrieval.index import ATOM_PAIRS
        sep_index = self.idx._sep_index
        apidx = {p: ATOM_PAIRS.index(p) for p in _CACB}

        A, SEPI, API, R, SIG = [], [], [], [], []
        for a in range(self.L):
            for b in range(a + 1, self.L):
                sep = b - a
                if not (self.minsep <= sep <= self.maxsep) or sep not in sep_index:
                    continue
                ri, rj = s + a + 1, s + b + 1
                sig = self.well_width * sep ** 0.15
                for (X, Y) in _CACB:
                    ci = coords.get((X, ri))
                    cj = coords.get((Y, rj))
                    if ci is None or cj is None:
                        continue
                    A.append(a); SEPI.append(sep_index[sep]); API.append(apidx[(X, Y)])
                    R.append(float(np.linalg.norm(np.asarray(ci) - np.asarray(cj))))
                    SIG.append(sig)
        return {k: np.array(v, dtype=dt) for (k, v), dt in zip(
            [("A", A), ("SEPI", SEPI), ("API", API), ("R", R), ("SIG", SIG)],
            [np.int32, np.int32, np.int32, np.float32, np.float32])}

    # ----- E_table precompute -------------------------------------------- #

    def _pool_gaussians(self, s, rows):
        """(M,) float32 summed Gaussian for ``rows`` fragments and window ``s``."""
        t = self.terms[s]
        if len(t["A"]) == 0 or len(rows) == 0:
            return np.zeros(len(rows), dtype=np.float32)
        d = (self.idx.dist[rows[:, None] + t["A"][None, :],
                           t["SEPI"][None, :],
                           t["API"][None, :]] * np.float32(0.1))   # (M, T) nm
        R, inv2s = t["R"][None, :], 0.5 / t["SIG"][None, :] ** 2
        G = np.exp(-((R - d) ** 2) * inv2s, dtype=np.float32)
        G[np.isnan(d)] = 0.0
        return G.sum(axis=1)

    def _build_table(self, N_frags, N_windows):
        E = np.zeros((N_windows, N_frags), dtype=np.float32)
        cs = self.chunk_size
        for s in range(N_windows):
            for c0 in range(0, N_frags, cs):
                c1 = min(c0 + cs, N_frags)
                E[s, c0:c1] = self._pool_gaussians(s, self.valid_rows[c0:c1])
            if (s + 1) % 10 == 0 or s == N_windows - 1:
                logger.debug("  E_table window %d/%d", s + 1, N_windows)
        return E

    # ----- BLOSUM score table -------------------------------------------- #

    def _build_scores(self, sequence, N_frags, N_windows):
        scores = np.zeros((N_windows, N_frags), dtype=np.int16)
        for s in range(N_windows):
            q = encode_sequence(sequence[s:s + self.L])
            scores[s] = SCORE_TABLE[q[None, :], self.valid_ctx].sum(axis=1).astype(np.int16)
        return scores

    # ----- S0/S1 bincount tables ----------------------------------------- #

    def _softmax_weights_for(self, s):
        """Softmax weights over all fragments for window ``s``, scaled to self.scale.

        Accumulated in float64 so that ``dE`` (which reweights these S0/S1 bins) and the
        fresh ``E_wt`` rebuilt on commit agree to double precision rather than the ~1e-2
        kJ/mol gap that float32 reduction over ~2M terms would introduce.
        """
        sc = self.scores[s].astype(np.float64)
        sc -= sc.max()
        e = np.exp(sc / self.temp)
        return (self.scale / e.sum()) * e

    def _rebuild_window_tables(self, s):
        """Rebuild S0[s], S1[s], E_wt[s] from current scores and E_table."""
        wts = self._softmax_weights_for(s)        # (N_frags,) float64
        E_w = self.E_table[s].astype(np.float64)  # (N_frags,) float64 view of float32 table
        self.E_wt[s] = float(np.dot(wts, E_w))
        wE = wts * E_w                             # (N_frags,) float64
        for off in range(self.L):
            aa = self._ctx_T[off]                  # (N_frags,) int labels
            self.S0[s, off] = np.bincount(aa, weights=wts, minlength=ALPHA)[:ALPHA]
            self.S1[s, off] = np.bincount(aa, weights=wE,  minlength=ALPHA)[:ALPHA]

    # ----- covering windows ---------------------------------------------- #

    def _covering_windows(self, position):
        """1-based position -> (0-based index, range of covering window starts)."""
        kk = position - 1
        n = len(self.seq)
        return kk, range(max(0, kk - self.L + 1), min(kk, n - self.L) + 1)

    # ----- public API ----------------------------------------------------- #

    def dE(self, position, new_aa) -> float:
        """ΔE of mutating ``position`` (1-based) to ``new_aa`` on the current sequence.

        O(ALPHA × L) — one 21-element dot product per affected window.

        Note: GLY mutations are not corrected for the missing CB atom in this prototype.
        Non-GLY mutations agree with :class:`FullDBMutationEnergy` to float32 precision.
        """
        kk, windows = self._covering_windows(position)
        new_code = int(encode_sequence(new_aa)[0])
        old_code = int(encode_sequence(self.seq[kk])[0])
        if new_code == old_code:
            return 0.0
        g = np.exp((SCORE_TABLE[new_code] - SCORE_TABLE[old_code]).astype(np.float64)
                   / self.temp)[:ALPHA]
        de = 0.0
        for s in windows:
            off = kk - s
            denom = float(g @ self.S0[s, off])
            if denom == 0.0:
                continue
            new_E_w = self.scale * float(g @ self.S1[s, off]) / denom
            de += new_E_w - self.E_wt[s]
        return float(-self.k_fm * de)

    def commit(self, position, new_aa) -> None:
        """Apply a mutation permanently; rebuild S0/S1 tables for affected windows."""
        kk, windows = self._covering_windows(position)
        new_code = int(encode_sequence(new_aa)[0])
        old_code = int(encode_sequence(self.seq[kk])[0])
        if new_code == old_code:
            return
        for s in windows:
            off = kk - s
            delta_B = (SCORE_TABLE[new_code, self._ctx_T[off]]
                       - SCORE_TABLE[old_code, self._ctx_T[off]])
            self.scores[s] = np.clip(
                self.scores[s].astype(np.int32) + delta_B,
                -32768, 32767).astype(np.int16)
            self._rebuild_window_tables(s)
        self.seq[kk] = new_aa

    def total_energy(self) -> float:
        """Total fragment-memory energy for the current sequence on the fixed structure."""
        return float(-self.k_fm * self.E_wt.sum())
