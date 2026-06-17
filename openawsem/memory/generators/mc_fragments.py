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

``dE`` is O(ALPHA × L) — essentially free.  ``commit`` rebuilds S0/S1 for the affected windows,
which is O(N_db × L) (a softmax + bincount over every database fragment) and is the Monte-Carlo
throughput ceiling.  ``commit_topk=K`` rebuilds over only the K highest-weight fragments per window
(re-selected each commit), shrinking that cost; at low ``soft_temp`` the weight concentrates so a
small K is near-exact and the walk reaches ~thousands of moves/s.

GPU acceleration (``device='cuda'``).  The two costs that scale with the database — the one-time
``E_table`` precompute and the per-accepted-move ``commit`` rebuild — run on the GPU, while the
tiny ``dE`` (O(ALPHA × L), database-independent) stays on the CPU where kernel-launch latency
would otherwise dominate.  See ``docs/mc_fragments.md`` for the full walk-through (what "rebuild
the affected windows" means, why it is the expensive step, and the ``commit_topk`` / ``soft_temp``
fast-MC recipe).
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
        Number of DB fragments processed per chunk during the CPU table build.
    device:
        ``None`` (default) for the CPU NumPy engine, or a torch device string such as
        ``'cuda'`` to run the ``E_table`` precompute and ``commit`` rebuild on the GPU.
    gpu_chunk:
        Number of DB fragments per chunk when building ``E_table`` on the GPU.  Caps the
        transient ``(chunk, n_terms)`` index tensor; ~400k keeps it near 270 MB.
    commit_topk:
        If set, ``commit`` rebuilds the S0/S1 tables over only the ``commit_topk`` highest-weight
        fragments per window (re-selected each commit, dynamic), instead of all ``N_db``.  This
        shrinks the commit scatter and lifts Monte-Carlo throughput.  It is an approximation of
        the full soft energy controlled by ``commit_topk`` and ``soft_temp``: at low ``soft_temp``
        the softmax weight concentrates, so a small ``commit_topk`` (~10-50k) is near-exact
        (recommended ``soft_temp≈1.0`` + ``commit_topk≈10-50k`` → ~2k moves/s, dE≈full-DB and
        better folding); at the default ``soft_temp=2.0`` the weight has a fat tail, so a larger
        ``commit_topk`` (~200k) is needed to stay within ~kT.  ``None`` (default) is the exact
        full-DB rebuild.
    """

    def __init__(self, database, *, fragment_length=9, well_width=0.1, soft_temp=2.0,
                 n_mem=20, min_seq_sep=3, max_seq_sep=9, k_fm=0.04184, chunk_size=50_000,
                 soft_pool=80, soft_seed_word=2, device=None, gpu_chunk=400_000,
                 commit_topk=None):
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
        self.device = device
        self.gpu_chunk = int(gpu_chunk)
        self.commit_topk = None if commit_topk is None else int(commit_topk)
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

    def mutation_engine(self, sequence, structure, *, k_fm=None,
                        device="__default__") -> "MCMutationEngine":
        """Build a precomputed-table mutation engine for ``sequence`` on fixed ``structure``.

        The returned :class:`MCMutationEngine` exposes ``dE(position, new_aa)`` and
        ``commit(position, new_aa)`` with the same interface as
        :class:`~openawsem.memory.mutation.FullDBMutationEnergy`.  Pass ``device='cuda'``
        to run the precompute and commit rebuild on the GPU.
        """
        dev = self.device if device == "__default__" else device
        return MCMutationEngine(self, sequence, structure,
                                k_fm=k_fm or self.k_fm, device=dev)


class MCMutationEngine:
    """Precomputed-table mutation engine.

    Stored tables after ``__init__``:
    - ``E_table[N_win, N_db]`` float32 — structural Gaussian energy per (window, fragment)
    - ``scores[N_win, N_db]``  int     — BLOSUM62 scores for the current sequence
    - ``S0[N_win, L, ALPHA]`` float64  — weight sum per (window, offset, AA)
    - ``S1[N_win, L, ALPHA]`` float64  — weighted energy sum per (window, offset, AA)
    - ``E_wt[N_win]`` float64          — current energy per window

    ``E_table`` and ``scores`` are NumPy arrays in CPU mode and torch tensors on the GPU when
    ``device`` is set; ``S0``/``S1``/``E_wt`` are always small host arrays (consumed by ``dE``).

    ``dE`` is O(ALPHA × L): one 21-element dot product per affected window (~9), always on CPU.
    ``commit`` is O(N_db × L): a softmax + bincount over every database fragment for each
    affected window, on CPU (NumPy) or GPU (torch scatter) depending on ``device``.
    """

    def __init__(self, gen: MCFragments, sequence: str, structure, *,
                 k_fm: float = 0.04184, device=None):
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
        self.device = device
        self.gpu_chunk = gen.gpu_chunk
        self.commit_topk = gen.commit_topk

        coords = FragmentMemory._structure_coords(structure)
        n = len(sequence)
        self.terms = [self._window_terms(s, coords) for s in range(n - self.L + 1)]

        self.valid_rows, self.valid_ctx = idx._encoded_for_length(self.L)
        N_frags = len(self.valid_rows)
        N_windows = len(self.terms)

        # Transpose ctx to (L, N_frags) so ctx_T[off] is a contiguous row.
        # int32 is np.intp-compatible on 64-bit systems and halves memory vs int64.
        self._ctx_T = np.ascontiguousarray(self.valid_ctx.T).astype(np.int32)  # (L, N_frags)

        if device is None:
            self._gpu = None
            logger.info("building E_table (CPU): %d fragments × %d windows ...",
                        N_frags, N_windows)
            self.E_table = self._build_table(N_frags, N_windows)
            self.scores = self._build_scores(sequence, N_frags, N_windows)
        else:
            logger.info("building E_table (GPU %s): %d fragments × %d windows ...",
                        device, N_frags, N_windows)
            self._gpu = _TorchBackend(self, device, sequence)
            self.E_table = self._gpu.E          # torch tensor on GPU
            self.scores = self._gpu.scores      # torch tensor on GPU

        self.S0 = np.zeros((N_windows, self.L, ALPHA), dtype=np.float64)
        self.S1 = np.zeros((N_windows, self.L, ALPHA), dtype=np.float64)
        self.E_wt = np.zeros(N_windows, dtype=np.float64)
        for s in range(N_windows):
            self._rebuild_window_tables(s)

        e_mb = self._nbytes(self.E_table) / 1e6
        s_mb = self._nbytes(self.scores) / 1e6
        logger.info("MCMutationEngine ready (%s) — E_table %.0f MB, scores %.0f MB",
                    device or "cpu", e_mb, s_mb)

    @staticmethod
    def _nbytes(arr) -> int:
        if isinstance(arr, np.ndarray):
            return arr.nbytes
        return arr.element_size() * arr.nelement()   # torch tensor

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

    # ----- E_table precompute (CPU) -------------------------------------- #

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

    # ----- BLOSUM score table (CPU) -------------------------------------- #

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
        """Rebuild S0[s], S1[s], E_wt[s] from the current scores and E_table.

        This is the per-affected-window cost during ``commit``: it forms the softmax weight
        of every database fragment for window ``s`` and bins those weights (and weight·energy)
        by the amino acid each fragment carries at each offset.  O(N_db × L).
        """
        if self._gpu is not None:
            self.S0[s], self.S1[s], self.E_wt[s] = self._gpu.rebuild(s, self.scale, self.temp)
            return
        wts = self._softmax_weights_for(s)        # (N_frags,) float64
        E_w = self.E_table[s].astype(np.float64)  # (N_frags,) float64 view of float32 table
        wE = wts * E_w                             # (N_frags,) float64
        ctx_T = self._ctx_T
        if self.commit_topk is not None and self.commit_topk < wts.shape[0]:
            top = np.argpartition(wts, -self.commit_topk)[-self.commit_topk:]
            wts, wE, ctx_T = wts[top], wE[top], ctx_T[:, top]
        for off in range(self.L):
            aa = ctx_T[off]                        # (M,) int labels
            self.S0[s, off] = np.bincount(aa, weights=wts, minlength=ALPHA)[:ALPHA]
            self.S1[s, off] = np.bincount(aa, weights=wE,  minlength=ALPHA)[:ALPHA]
        # E_wt = Σ_f w_f·E_f = Σ_a S1[s,0,a]; reuse S1 instead of a second O(N_db) dot.
        self.E_wt[s] = float(self.S1[s, 0].sum())

    # ----- covering windows ---------------------------------------------- #

    def _covering_windows(self, position):
        """1-based position -> (0-based index, range of covering window starts)."""
        kk = position - 1
        n = len(self.seq)
        return kk, range(max(0, kk - self.L + 1), min(kk, n - self.L) + 1)

    # ----- public API ----------------------------------------------------- #

    def dE(self, position, new_aa) -> float:
        """ΔE of mutating ``position`` (1-based) to ``new_aa`` on the current sequence.

        O(ALPHA × L) — one 21-element dot product per affected window.  Always runs on the
        host (the per-call work is too small to benefit from a GPU launch).

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
        """Apply a mutation permanently; rebuild S0/S1 tables for the affected windows."""
        kk, windows = self._covering_windows(position)
        new_code = int(encode_sequence(new_aa)[0])
        old_code = int(encode_sequence(self.seq[kk])[0])
        if new_code == old_code:
            return
        for s in windows:
            off = kk - s
            if self._gpu is not None:
                self._gpu.apply_score_delta(s, off, new_code, old_code)
            else:
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


class _TorchBackend:
    """GPU-resident E_table/scores + per-window S0/S1 rebuild (torch).

    Holds the two O(N_db) tensors on the GPU and rebuilds the small S0/S1/E_wt tables for one
    window at a time, copying the results back to the host so the hot ``dE`` loop stays on CPU.
    The softmax weights and the scatter accumulate in float64 (the per-window scatter is
    memory-bound, only ~2.4× the float32 cost, so it stays ~20× faster than CPU while keeping
    ``dE == E_after - E_before`` consistent to double precision).  ``E_table`` itself is held in
    float32; GPU and CPU engines then agree to ~1e-6 kJ/mol on total energy and dE.
    """

    def __init__(self, eng: MCMutationEngine, device, sequence: str):
        import torch
        self.torch = torch
        self.eng = eng
        self.device = torch.device(device)
        self.ctx = torch.as_tensor(eng._ctx_T, device=self.device, dtype=torch.long)  # (L, N)
        self.B = torch.as_tensor(np.asarray(SCORE_TABLE), device=self.device, dtype=torch.int32)
        self.E = self._build_table()                # (W, N) float32 on GPU
        self.scores = self._build_scores(sequence)  # (W, N) int32 on GPU

    def _build_table(self):
        torch, eng, dev = self.torch, self.eng, self.device
        dist = eng.idx.dist                          # memmap (Nfull, n_seps, npair) Angstrom
        n_seps, npair = dist.shape[1], dist.shape[2]
        stride = n_seps * npair
        valid_rows = np.asarray(eng.valid_rows)
        N, W = len(valid_rows), len(eng.terms)
        E = torch.empty((W, N), device=dev, dtype=torch.float32)
        gterms = []
        for s in range(W):
            t = eng.terms[s]
            if len(t["A"]) == 0:
                gterms.append(None); continue
            gterms.append((
                torch.as_tensor(t["A"], device=dev, dtype=torch.long),
                torch.as_tensor(t["SEPI"] * npair + t["API"], device=dev, dtype=torch.long),
                torch.as_tensor(t["R"], device=dev, dtype=torch.float32),
                torch.as_tensor(0.5 / (t["SIG"] ** 2), device=dev, dtype=torch.float32)))
        zero = torch.zeros((), device=dev)
        chunk = eng.gpu_chunk
        for c0 in range(0, N, chunk):
            c1 = min(c0 + chunk, N)
            rows = valid_rows[c0:c1]
            lo, hi = int(rows.min()), int(rows.max()) + eng.L
            dspan = torch.as_tensor(np.ascontiguousarray(dist[lo:hi]).reshape(-1), device=dev)
            local = torch.as_tensor(rows - lo, device=dev, dtype=torch.long)
            for s in range(W):
                g = gterms[s]
                if g is None:
                    E[s, c0:c1] = 0.0; continue
                A, base, R, inv2s = g
                lin = (local[:, None] + A[None, :]) * stride + base[None, :]
                d = dspan[lin] * 0.1
                G = torch.exp(-((R[None, :] - d) ** 2) * inv2s[None, :])
                E[s, c0:c1] = torch.where(torch.isnan(d), zero, G).sum(dim=1)
        torch.cuda.synchronize()
        return E

    def _build_scores(self, sequence):
        torch, eng, dev = self.torch, self.eng, self.device
        W, N, L = len(eng.terms), self.ctx.shape[1], eng.L
        scores = torch.empty((W, N), device=dev, dtype=torch.int32)
        for s in range(W):
            q = encode_sequence(sequence[s:s + L])
            acc = torch.zeros(N, device=dev, dtype=torch.int32)
            for off in range(L):
                acc += self.B[int(q[off]), self.ctx[off]]
            scores[s] = acc
        torch.cuda.synchronize()
        return scores

    def rebuild(self, s, scale, temp):
        """Rebuild S0[s], S1[s], E_wt[s] on the GPU; return host (S0_np, S1_np, E_wt_float)."""
        torch, dev, L = self.torch, self.device, self.eng.L
        sc = self.scores[s].double()
        sc = sc - sc.max()
        e = torch.exp(sc / temp)
        w = (scale / e.sum()) * e                    # (N,) float64 softmax weights
        Es = self.E[s].double()                       # (N,) structural energies
        wE = w * Es
        ctx = self.ctx
        topk = self.eng.commit_topk
        if topk is not None and topk < w.shape[0]:
            idx = torch.topk(w, topk, sorted=False).indices   # highest-weight fragments
            w, wE, ctx = w[idx], wE[idx], self.ctx[:, idx]
        # One scatter over all L offsets at once: ``expand`` is a zero-copy view, so w is read
        # through cache instead of re-streamed per offset (fewer launches, ~17% faster rebin).
        S0 = torch.zeros((L, ALPHA), device=dev, dtype=torch.float64)
        S1 = torch.zeros((L, ALPHA), device=dev, dtype=torch.float64)
        S0.scatter_add_(1, ctx, w.unsqueeze(0).expand(L, -1))
        S1.scatter_add_(1, ctx, wE.unsqueeze(0).expand(L, -1))
        E_wt = float(S1[0].sum())                     # = Σ_f w_f·E_f, reuse S1 (no extra dot)
        return (S0.cpu().numpy(), S1.cpu().numpy(), E_wt)

    def apply_score_delta(self, s, off, new_code, old_code):
        """In-place GPU update of scores[s] for a mutation at offset ``off``."""
        ctx_row = self.ctx[off]
        self.scores[s] += self.B[new_code, ctx_row] - self.B[old_code, ctx_row]
