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
    triton_commit:
        Use the Triton block-partial histogram for the exact full-DB ``commit`` rebuild
        (``commit_topk=None``) on a CUDA/ROCm device.  Each program reduces its tile of fragments
        by amino acid without atomics and writes its own partial, replacing the contention-bound
        ``scatter_add`` (~30-36× faster end-to-end commit on a 3080 Ti).  Default ``True``; falls
        back to the ``scatter_add`` when triton is unavailable, on CPU, or when ``commit_topk`` is
        set.
    hist_dtype:
        Accumulation precision for the Triton histogram.  ``'f32'`` (default) sums each tile in
        float32 and reduces into float64 (``dE`` within ~1e-5 kJ/mol of the float64 scatter, well
        inside the mutation tolerance); ``'f64'`` accumulates entirely in float64 (bit-exact
        ``dE == E_after − E_before``, ~6× slower than ``'f32'`` but still faster than the scatter).
    gly_masking:
        How glycine CB wells behave under sequence mutation.  Either way a virtual CB (built in Angstrom
        from the backbone) is placed at every structural glycine, so every position has a CB and mutating
        into or out of glycine is smooth (no on/off discontinuity in the geometry).

        ``False`` (default): the lean, fast path.  Glycine is treated as having a CB and its wells stay
        active (the database's virtual-CB convention); this is self-consistent and smooth, but differs
        from the conventional masked energy at glycine positions (the glycine CB wells, ~7.5% of the
        total on 1r69).

        ``True``: additionally mask the CB-pair wells by the *current* residue (the ``Gother``/``Gtouch``
        split), so they switch off where a position is glycine and on where it is not -- the exact
        conventional energy, agreeing with :class:`FullDBMutationEnergy`.  Works on CPU (``device=None``)
        and GPU (``device='cuda'``); it stores the ``Gother``/``Gtouch`` geometry resident (~9 GB float32
        for pc30/1r69 -- the GPU path needs a card with enough memory and raises a clear error
        otherwise).  A commit that does not change a position's glycine state is fast; one that flips a
        position into or out of glycine rebuilds that geometry from the database (~0.5 s on the GPU).

        Neither is strictly more correct (masking reproduces conventional AWSEM; the default is the
        database's virtual-CB convention).  For Monte-Carlo sequence design the default is usually
        preferable: it needs no large table (fits any GPU, faster) and treats glycine as an ordinary
        residue, whereas masking makes leaving glycine artificially favorable (it regains the masked-off
        CB contacts), biasing the walk to flee glycine.  Use ``True`` for an exact conventional energy.
    """

    def __init__(self, database, *, fragment_length=9, well_width=0.1, soft_temp=2.0,
                 n_mem=20, min_seq_sep=3, max_seq_sep=9, k_fm=0.04184, chunk_size=50_000,
                 soft_pool=80, soft_seed_word=2, device=None, gpu_chunk=400_000,
                 commit_topk=None, triton_commit=True, hist_dtype="f32",
                 gly_masking=False):
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
        self.triton_commit = bool(triton_commit)
        self.hist_dtype = str(hist_dtype)
        self.gly_masking = bool(gly_masking)
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
        self.triton_commit = gen.triton_commit
        self.hist_dtype = gen.hist_dtype
        self.gly_masking = gen.gly_masking

        # Always place a virtual CB at structural glycines (built in Angstrom) so every position has a
        # CB and mutating into or out of glycine is smooth -- no on/off discontinuity in the geometry.
        # The default then treats glycine as having a CB (wells always active, the database's virtual-CB
        # convention); ``gly_masking`` additionally drops those CB wells wherever the current residue is
        # glycine, for the exact conventional energy.
        coords = FragmentMemory._structure_coords(structure, virtual_cb=True)
        n = len(sequence)
        self.terms = [self._window_terms(s, coords) for s in range(n - self.L + 1)]

        self.valid_rows, self.valid_ctx = idx._encoded_for_length(self.L)
        N_frags = len(self.valid_rows)
        N_windows = len(self.terms)

        # Transpose ctx to (L, N_frags) so ctx_T[off] is a contiguous row.
        # int32 is np.intp-compatible on 64-bit systems and halves memory vs int64.
        self._ctx_T = np.ascontiguousarray(self.valid_ctx.T).astype(np.int32)  # (L, N_frags)

        if self.gly_masking:
            self.S0 = np.zeros((N_windows, self.L, ALPHA), dtype=np.float64)
            self.S1o = np.zeros((N_windows, self.L, ALPHA), dtype=np.float64)
            self.S1t = np.zeros((N_windows, self.L, ALPHA), dtype=np.float64)
            self.E_wt = np.zeros(N_windows, dtype=np.float64)
            if device is None:
                self._gpu = None
                self.scores = self._build_scores(sequence, N_frags, N_windows)
                self._build_split_tables(N_windows)            # builds self._GM + fills tables
                gm_mb = sum(gm.nbytes for gm in self._GM) / 1e6
                logger.info("MCMutationEngine ready (gly_masking, cpu) — GM %.0f MB", gm_mb)
            else:
                self._gpu = _TorchBackend(self, device, sequence)   # GM + scores on GPU
                self.scores = self._gpu.scores                 # GPU scores (commit updates in place)
                self._rebuild_split_windows_gpu(range(N_windows))
                logger.info("MCMutationEngine ready (gly_masking, %s) — GM %.0f MB", device,
                            self._nbytes(self._gpu.GM) / 1e6)
            return

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
        if self._gpu is not None:
            self._rebuild_windows_gpu(range(N_windows))
        else:
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

        A, SEPI, API, R, SIG, BOFF, NEEDA, NEEDB = [], [], [], [], [], [], [], []
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
                    BOFF.append(b); NEEDA.append(X == "CB"); NEEDB.append(Y == "CB")
        # A is the per-term A-end offset (== AOFF); BOFF the B-end offset; NEEDA/NEEDB flag a CB end
        # (used by glycine masking to drop CB-touching terms where the current residue is glycine).
        return {k: np.array(v, dtype=dt) for (k, v), dt in zip(
            [("A", A), ("SEPI", SEPI), ("API", API), ("R", R), ("SIG", SIG),
             ("BOFF", BOFF), ("NEEDA", NEEDA), ("NEEDB", NEEDB)],
            [np.int32, np.int32, np.int32, np.float32, np.float32, np.int32, np.bool_, np.bool_])}

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
        """Rebuild S0[s], S1[s], E_wt[s] for window ``s`` on the CPU (NumPy path).

        Forms the softmax weight of each database fragment and bins the weights (and
        weight·energy) by amino acid at each offset.  With ``commit_topk`` set, only the top-K
        fragments by score are kept (selected on the raw score — monotonic with the weight — so
        the O(N_db) ``exp`` is replaced by an ``exp`` over only K; the K survivors are
        renormalized to ``scale``).
        """
        sc = self.scores[s].astype(np.float64)
        E_w = self.E_table[s].astype(np.float64)
        ctx_T = self._ctx_T
        if self.commit_topk is not None and self.commit_topk < sc.shape[0]:
            top = np.argpartition(sc, -self.commit_topk)[-self.commit_topk:]
            sc, E_w, ctx_T = sc[top], E_w[top], ctx_T[:, top]
        sc = sc - sc.max()
        e = np.exp(sc / self.temp)
        wts = (self.scale / e.sum()) * e
        wE = wts * E_w
        for off in range(self.L):
            aa = ctx_T[off]
            self.S0[s, off] = np.bincount(aa, weights=wts, minlength=ALPHA)[:ALPHA]
            self.S1[s, off] = np.bincount(aa, weights=wE,  minlength=ALPHA)[:ALPHA]
        # E_wt = Σ_f w_f·E_f = Σ_a S1[s,0,a]; reuse S1 instead of a second O(N_db) dot.
        self.E_wt[s] = float(self.S1[s, 0].sum())

    # ----- glycine masking: Gother/Gtouch split (CPU) -------------------- #

    def _pool_terms(self, s, rows):
        """(M, T) float32 per-term Gaussians for window ``s`` (not summed over terms)."""
        t = self.terms[s]
        if len(t["A"]) == 0 or len(rows) == 0:
            return np.zeros((len(rows), len(t["A"])), dtype=np.float32)
        d = (self.idx.dist[rows[:, None] + t["A"][None, :],
                           t["SEPI"][None, :], t["API"][None, :]] * np.float32(0.1))
        R, inv2s = t["R"][None, :], 0.5 / t["SIG"][None, :] ** 2
        G = np.exp(-((R - d) ** 2) * inv2s, dtype=np.float32)
        G[np.isnan(d)] = 0.0
        return G

    def _selection_matrix(self, s):
        """(T, 2L) Gother/Gtouch columns per offset from the CURRENT sequence's glycine pattern.

        Column ``2*off`` selects terms not needing a CB at offset ``off`` (kept unless the term's other
        end is a current glycine); column ``2*off+1`` selects terms needing a CB at ``off`` (pre-masked
        by the other end's glycine state).  ``G @ M`` gives the per-offset Gother/Gtouch per fragment.
        """
        t = self.terms[s]; T = len(t["A"]); L = self.L
        if T == 0:
            return np.zeros((0, 2 * L), np.float32)
        AOFF, BOFF, NEEDA, NEEDB = t["A"], t["BOFF"], t["NEEDA"], t["NEEDB"]
        glyA = np.fromiter((self.seq[s + o] == "G" for o in AOFF), bool, T)
        glyB = np.fromiter((self.seq[s + o] == "G" for o in BOFF), bool, T)
        include = ~((NEEDA & glyA) | (NEEDB & glyB))
        M = np.zeros((T, 2 * L), np.float32)
        for off in range(L):
            at = NEEDA & (AOFF == off); bt = NEEDB & (BOFF == off); touch = at | bt
            other_ok = np.ones(T, bool)
            other_ok[at] = ~(NEEDB[at] & glyB[at]); other_ok[bt] = ~(NEEDA[bt] & glyA[bt])
            M[include & ~touch, 2 * off] = 1.0
            M[touch & other_ok, 2 * off + 1] = 1.0
        return M

    def _build_GM(self, s):
        G = self._pool_terms(s, self.valid_rows)
        if G.shape[1] == 0:
            return np.zeros((len(self.valid_rows), 2 * self.L), np.float32)
        return (G @ self._selection_matrix(s)).astype(np.float32)            # (N, 2L)

    def _build_split_tables(self, N_windows):
        self._GM = [self._build_GM(s) for s in range(N_windows)]   # CPU per-window (N, 2L) float32
        for s in range(N_windows):
            self._rebuild_split_window(s)

    def _rebuild_split_windows_gpu(self, windows):
        """Rebuild S0/S1o/S1t/E_wt for ``windows`` via the GPU split kernel (one host transfer)."""
        windows = list(windows)
        S0h, S1oh, S1th = self._gpu.rebuild_windows_split(windows, self.scale, self.temp)
        for i, s in enumerate(windows):
            self.S0[s], self.S1o[s], self.S1t[s] = S0h[i], S1oh[i], S1th[i]
            old_gly = self.seq[s] == "G"
            self.E_wt[s] = float(S1oh[i, 0].sum() + (0.0 if old_gly else S1th[i, 0].sum()))

    def _rebuild_split_window(self, s):
        sc = self.scores[s].astype(np.float64); sc = sc - sc.max()
        e = np.exp(sc / self.temp)
        w = (self.scale / e.sum()) * e                                       # weights sum to scale
        GM = self._GM[s]
        for off in range(self.L):
            aa = self._ctx_T[off]
            self.S0[s, off] = np.bincount(aa, weights=w, minlength=ALPHA)[:ALPHA]
            self.S1o[s, off] = np.bincount(aa, weights=w * GM[:, 2 * off], minlength=ALPHA)[:ALPHA]
            self.S1t[s, off] = np.bincount(aa, weights=w * GM[:, 2 * off + 1], minlength=ALPHA)[:ALPHA]
        old_gly = self.seq[s] == "G"
        self.E_wt[s] = float(self.S1o[s, 0].sum() + (0.0 if old_gly else self.S1t[s, 0].sum()))

    def _rebuild_windows_gpu(self, windows):
        """Rebuild a list of windows on the GPU with a single host transfer, then store them."""
        windows = list(windows)
        S0h, S1h, Ewh = self._gpu.rebuild_windows(windows, self.scale, self.temp)
        for i, s in enumerate(windows):
            self.S0[s] = S0h[i]
            self.S1[s] = S1h[i]
            self.E_wt[s] = Ewh[i]

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

        Glycine (see ``MCFragments.gly_masking``): both modes give every position a virtual CB so
        mutations into/out of glycine are smooth.  The default treats glycine as having a CB (wells
        active); ``gly_masking=True`` masks those wells by the current residue for the exact conventional
        energy (agreeing with :class:`FullDBMutationEnergy`).
        """
        kk, windows = self._covering_windows(position)
        new_code = int(encode_sequence(new_aa)[0])
        old_code = int(encode_sequence(self.seq[kk])[0])
        if new_code == old_code:
            return 0.0
        g = np.exp((SCORE_TABLE[new_code] - SCORE_TABLE[old_code]).astype(np.float64)
                   / self.temp)[:ALPHA]
        de = 0.0
        if self.gly_masking:
            new_gly = new_aa == "G"
            for s in windows:
                off = kk - s
                denom = float(g @ self.S0[s, off])
                if denom == 0.0:
                    continue
                num = float(g @ self.S1o[s, off]) + (0.0 if new_gly else float(g @ self.S1t[s, off]))
                de += self.scale * num / denom - self.E_wt[s]
            return float(-self.k_fm * de)
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
        windows = list(windows)
        if self.gly_masking:
            gly_flip = (new_aa == "G") != (self.seq[kk] == "G")
            if self._gpu is not None:
                for s in windows:
                    self._gpu.apply_score_delta(s, kk - s, new_code, old_code)
                self.seq[kk] = new_aa                     # before rebuilding the (seq-dependent) split
                if gly_flip:
                    self._gpu.rebuild_GM_windows(windows)
                self._rebuild_split_windows_gpu(windows)
                return
            for s in windows:
                off = kk - s
                delta_B = (SCORE_TABLE[new_code, self._ctx_T[off]]
                           - SCORE_TABLE[old_code, self._ctx_T[off]])
                self.scores[s] = np.clip(self.scores[s].astype(np.int32) + delta_B,
                                         -32768, 32767).astype(np.int16)
            self.seq[kk] = new_aa                         # update before rebuilding the glycine split
            for s in windows:
                if gly_flip:
                    self._GM[s] = self._build_GM(s)       # glycine pattern changed -> rebuild the split
                self._rebuild_split_window(s)
            return
        if self._gpu is not None:
            for s in windows:
                self._gpu.apply_score_delta(s, kk - s, new_code, old_code)
            self._rebuild_windows_gpu(windows)          # all windows, one host transfer
        else:
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
        self.gly_masking = eng.gly_masking
        self.hist_dtype = eng.hist_dtype
        from openawsem.memory.generators import _mc_triton
        if self.gly_masking and not (_mc_triton.HAVE_TRITON and self.device.type == "cuda"):
            raise RuntimeError("GPU gly_masking needs triton on a CUDA/ROCm device")
        self._triton = _mc_triton if (eng.triton_commit and _mc_triton.HAVE_TRITON
                                      and self.device.type == "cuda") else None
        self.ctx_i8 = self.ctx.to(torch.int8).contiguous() if (self._triton is not None
                                                               or self.gly_masking) else None
        self.scores = self._build_scores(sequence)  # (W, N) int32 on GPU
        if self.gly_masking:
            # GM[s] = (per-term Gaussians) @ (Gother/Gtouch selection), per window, resident on the GPU.
            # Size W*N*2L float32 (~9 GB for pc30/1r69) -- needs a GPU with enough memory.
            try:
                self.GM = self._build_GM()          # (W, N, 2L) float32 on GPU
            except torch.cuda.OutOfMemoryError as exc:
                W, N = len(eng.terms), len(eng.valid_rows)
                need = W * N * 2 * eng.L * 4 / 1e9
                raise torch.cuda.OutOfMemoryError(
                    f"GPU gly_masking needs the (W, N, 2L) Gother/Gtouch geometry resident on the GPU "
                    f"(~{need:.1f} GB for this structure/database). Use a GPU with more memory, a "
                    f"smaller/curated database, or run gly_masking on the CPU (device=None).") from exc
        else:
            self.E = self._build_table()            # (W, N) float32 on GPU

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
            dspan = torch.as_tensor(np.array(dist[lo:hi]).reshape(-1), device=dev)  # copy: dist is a read-only memmap
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

    # ----- glycine masking: GM (Gother/Gtouch) geometry on the GPU -------- #

    def _gterms(self, s):
        """(A, base, R, inv2s) GPU tensors for window ``s``'s structure terms, or None if empty."""
        torch, eng, dev = self.torch, self.eng, self.device
        n_seps, npair = eng.idx.dist.shape[1], eng.idx.dist.shape[2]
        t = eng.terms[s]
        if len(t["A"]) == 0:
            return None
        return (torch.as_tensor(t["A"], device=dev, dtype=torch.long),
                torch.as_tensor(t["SEPI"] * npair + t["API"], device=dev, dtype=torch.long),
                torch.as_tensor(t["R"], device=dev, dtype=torch.float32),
                torch.as_tensor(0.5 / (t["SIG"] ** 2), device=dev, dtype=torch.float32))

    def _fill_GM(self, windows):
        """Fill ``self.GM[s]`` (2L, N) = ((per-term Gaussians) @ selection-matrix(s)).T for each window
        in ``windows``.  Each distance chunk is read from the memmap ONCE and reused across all the
        windows (mirroring ``_build_table``), so init and a glycine-flip rebuild both read the database
        only once rather than once per window."""
        torch, eng, dev = self.torch, self.eng, self.device
        n_seps, npair = eng.idx.dist.shape[1], eng.idx.dist.shape[2]
        stride = n_seps * npair
        valid_rows = np.asarray(eng.valid_rows)
        N = len(valid_rows)
        windows = list(windows)
        gt = {s: self._gterms(s) for s in windows}
        Mts = {s: torch.as_tensor(eng._selection_matrix(s), device=dev, dtype=torch.float32)
               for s in windows if gt[s] is not None}
        zero = torch.zeros((), device=dev)
        for c0 in range(0, N, eng.gpu_chunk):
            c1 = min(c0 + eng.gpu_chunk, N)
            rows = valid_rows[c0:c1]
            lo, hi = int(rows.min()), int(rows.max()) + eng.L
            dspan = torch.as_tensor(np.array(eng.idx.dist[lo:hi]).reshape(-1), device=dev)
            local = torch.as_tensor(rows - lo, device=dev, dtype=torch.long)
            for s in windows:
                g = gt[s]
                if g is None:
                    self.GM[s][:, c0:c1] = 0.0; continue
                A, base, R, inv2s = g
                lin = (local[:, None] + A[None, :]) * stride + base[None, :]
                d = dspan[lin] * 0.1
                G = torch.where(torch.isnan(d), zero,
                                torch.exp(-((R[None, :] - d) ** 2) * inv2s[None, :]))
                self.GM[s][:, c0:c1] = (G @ Mts[s]).T       # (2L, chunk)
        torch.cuda.synchronize()

    def _build_GM(self):
        N, W = len(self.eng.valid_rows), len(self.eng.terms)
        self.GM = self.torch.empty((W, 2 * self.eng.L, N), device=self.device, dtype=self.torch.float32)
        self._fill_GM(range(W))
        return self.GM

    def rebuild_GM_windows(self, windows):
        """Rebuild GM for windows whose glycine pattern flipped (the geometry is sequence-independent
        per glycine pattern, so only a flip needs this)."""
        self._fill_GM(windows)

    def rebuild_windows_split(self, windows, scale, temp):
        """S0/S1o/S1t host arrays for ``windows`` via the split Triton histogram, ONE host transfer."""
        return self._triton.rebuild_windows_split(self.scores, self.GM, self.ctx_i8,
                                                  list(windows), temp, scale, hist_dtype=self.hist_dtype)

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

    def rebuild_windows(self, windows, scale, temp):
        """Rebuild S0/S1/E_wt for a list of windows on the GPU; return host arrays in ONE transfer.

        Batching all affected windows into a single ``.cpu()`` avoids per-window GPU→CPU sync
        latency (each sync ~0.4 ms; a commit touches ~L windows).  With ``commit_topk`` set the
        top-K fragments are picked on the raw int score (monotonic with the weight), so the
        softmax ``exp`` runs over only K instead of all N_db, and the K survivors are
        renormalized to ``scale`` (at low ``soft_temp`` the dropped tail's weight is negligible,
        so this matches the full-DB energy).
        """
        torch, dev, L = self.torch, self.device, self.eng.L
        topk = self.eng.commit_topk
        if self._triton is not None and topk is None:
            return self._triton.rebuild_windows(
                self.scores, self.E, self.ctx_i8, list(windows), temp, scale,
                hist_dtype=self.hist_dtype)
        nw = len(windows)
        S0 = torch.zeros((nw, L, ALPHA), device=dev, dtype=torch.float64)
        S1 = torch.zeros((nw, L, ALPHA), device=dev, dtype=torch.float64)
        for i, s in enumerate(windows):
            sc = self.scores[s]
            if topk is not None and topk < sc.shape[0]:
                idx = torch.topk(sc, topk, sorted=False).indices   # top-K by score (no O(N) exp)
                scd, Es, ctx = sc[idx].double(), self.E[s][idx].double(), self.ctx[:, idx]
            else:
                scd, Es, ctx = sc.double(), self.E[s].double(), self.ctx
            scd = scd - scd.max()
            e = torch.exp(scd / temp)
            w = (scale / e.sum()) * e
            wE = w * Es
            # One scatter over all L offsets at once: ``expand`` is a zero-copy view of the weights.
            S0[i].scatter_add_(1, ctx, w.unsqueeze(0).expand(L, -1))
            S1[i].scatter_add_(1, ctx, wE.unsqueeze(0).expand(L, -1))
        S0h, S1h = S0.cpu().numpy(), S1.cpu().numpy()              # one host transfer each
        E_wt = S1h[:, 0, :].sum(axis=1)                            # Σ_f w_f·E_f per window
        return S0h, S1h, E_wt

    def apply_score_delta(self, s, off, new_code, old_code):
        """In-place GPU update of scores[s] for a mutation at offset ``off``."""
        ctx_row = self.ctx[off]
        self.scores[s] += self.B[new_code, ctx_row] - self.B[old_code, ctx_row]
