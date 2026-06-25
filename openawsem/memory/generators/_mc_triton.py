"""Triton block-partial histogram for the exact full-DB ``S0``/``S1`` rebuild (``commit``).

The float64 ``scatter_add`` over the database (``N`` fragments per window) is the throughput
ceiling of ``mc_fragments.commit``: ``N*L`` atomic adds into ``ALPHA`` bins serialize, running at
~2% of peak memory bandwidth.  This module replaces it with a contention-free block-partial
histogram: each Triton program reduces its tile of fragments by amino acid using shared-memory
tree reductions (no atomics) and writes its own ``(L, ALPHA)`` partial; ``torch`` sums the
partials.  On a single RTX 3080 Ti this is ~30-36x faster end-to-end commit (pc30 215 -> 7 ms,
pc80res3 1043 -> 29 ms).

The kernel accumulates the *unnormalized* softmax weights ``exp((score - max)/T)``; the softmax
denominator is recovered as the sum over the bins (``sum_a S0[0, a]``), so no separate sum-exp
pass over ``N`` is needed.  ``ctx`` is read as ``int8`` (was ``int64``), and the per-window
weights are never materialized as length-``N`` tensors.

``hist_dtype``:
  ``'f32'`` tile sums in float32, reduced into float64 -- recommended; ``dE`` agrees with the
            float64 scatter to ~1e-5 kJ/mol (well within the ~1e-4 mutation tolerance).
  ``'f64'`` everything in float64 -- bit-exact ``dE == E_after - E_before`` (~1e-11), ~6x slower.

Falls back gracefully: import this module and use :func:`rebuild_windows` only when
:data:`HAVE_TRITON` is true; the caller keeps the ``scatter_add`` path otherwise.
"""
from __future__ import annotations

from openawsem.memory.retrieval.blosum import SCORE_TABLE

ALPHA = SCORE_TABLE.shape[0]            # 20 AAs + unknown (matches mc_fragments.ALPHA)

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                       # pragma: no cover - optional dependency
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _whist_block(scores_ptr, E_ptr, ctx_ptr, S0_ptr, S1_ptr,
                     N, max_score, inv_temp,
                     L: tl.constexpr, A: tl.constexpr, BLOCK: tl.constexpr, CDT: tl.constexpr):
        pid = tl.program_id(0)
        offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        mask = offs < N
        sc = tl.load(scores_ptr + offs, mask=mask, other=0).to(CDT)
        e = tl.where(mask, tl.exp((sc - max_score) * inv_temp), tl.zeros((BLOCK,), CDT))
        Ev = tl.load(E_ptr + offs, mask=mask, other=0.0).to(CDT)
        eE = e * Ev
        base = pid * L * A
        for off in range(L):
            c = tl.load(ctx_ptr + off * N + offs, mask=mask, other=255).to(tl.int32)
            for a in tl.static_range(A):
                m = c == a
                tl.store(S0_ptr + base + off * A + a, tl.sum(tl.where(m, e, 0.0)))
                tl.store(S1_ptr + base + off * A + a, tl.sum(tl.where(m, eE, 0.0)))

    @triton.jit
    def _whist_split(scores_ptr, GM_ptr, ctx_ptr, S0_ptr, S1o_ptr, S1t_ptr,
                     N, max_score, inv_temp,
                     L: tl.constexpr, A: tl.constexpr, BLOCK: tl.constexpr, CDT: tl.constexpr):
        # Glycine masking: GM is (2L, N) -- row 2*off = Gother(off), row 2*off+1 = Gtouch(off), so the
        # per-offset reads over consecutive fragments are coalesced.
        pid = tl.program_id(0)
        offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        mask = offs < N
        sc = tl.load(scores_ptr + offs, mask=mask, other=0).to(CDT)
        e = tl.where(mask, tl.exp((sc - max_score) * inv_temp), tl.zeros((BLOCK,), CDT))
        base = pid * L * A
        for off in range(L):
            c = tl.load(ctx_ptr + off * N + offs, mask=mask, other=255).to(tl.int32)
            go = tl.load(GM_ptr + (2 * off) * N + offs, mask=mask, other=0.0).to(CDT)
            gt = tl.load(GM_ptr + (2 * off + 1) * N + offs, mask=mask, other=0.0).to(CDT)
            eo, et = e * go, e * gt
            for a in tl.static_range(A):
                m = c == a
                tl.store(S0_ptr + base + off * A + a, tl.sum(tl.where(m, e, 0.0)))
                tl.store(S1o_ptr + base + off * A + a, tl.sum(tl.where(m, eo, 0.0)))
                tl.store(S1t_ptr + base + off * A + a, tl.sum(tl.where(m, et, 0.0)))


def rebuild_windows_split(scores, GM, ctx_i8, windows, temp, scale, *, hist_dtype="f32", BLOCK=2048):
    """Glycine-masked ``S0``/``S1o``/``S1t`` for ``windows`` via the split Triton histogram.

    ``scores`` (W, N) int, ``GM`` (W, N, 2L) float32 (Gother/Gtouch per offset), ``ctx_i8`` (L, N) int8.
    Returns numpy ``(S0h, S1oh, S1th)`` (each ``(len(windows), L, ALPHA)``) in one host transfer.
    """
    import torch
    dev = scores.device
    L, N = ctx_i8.shape
    A = ALPHA
    cdt = tl.float32 if hist_dtype == "f32" else tl.float64
    acc_t = torch.float32 if hist_dtype == "f32" else torch.float64
    inv_temp = 1.0 / temp
    grid = (triton.cdiv(N, BLOCK),)
    nb = grid[0]
    out = [torch.empty((len(windows), L, A), device=dev, dtype=torch.float64) for _ in range(3)]
    u = [torch.empty((nb, L, A), device=dev, dtype=acc_t) for _ in range(3)]
    for i, s in enumerate(windows):
        sc = scores[s].contiguous()
        gm = GM[s].contiguous()
        max_score = float(sc.max().item())
        _whist_split[grid](sc, gm, ctx_i8, u[0], u[1], u[2], N, max_score, inv_temp,
                           L=L, A=A, BLOCK=BLOCK, CDT=cdt)
        S0 = u[0].double().sum(0)
        norm = scale / S0[0].sum()
        out[0][i] = norm * S0
        out[1][i] = norm * u[1].double().sum(0)
        out[2][i] = norm * u[2].double().sum(0)
    return tuple(o.cpu().numpy() for o in out)


def rebuild_windows(scores, E, ctx_i8, windows, temp, scale, *, hist_dtype="f32", BLOCK=2048):
    """Exact full-DB ``S0``/``S1``/``E_wt`` for ``windows`` via the Triton histogram.

    Parameters mirror the tensors held by ``_TorchBackend``:
    ``scores`` (W, N) int, ``E`` (W, N) float32, ``ctx_i8`` (L, N) int8.  Returns numpy
    ``(S0h, S1h, E_wt)`` with one host transfer (S0h/S1h shaped ``(len(windows), L, ALPHA)``).
    """
    import torch
    dev = scores.device
    L, N = ctx_i8.shape
    A = ALPHA
    cdt = tl.float32 if hist_dtype == "f32" else tl.float64
    acc_t = torch.float32 if hist_dtype == "f32" else torch.float64
    inv_temp = 1.0 / temp
    grid = (triton.cdiv(N, BLOCK),)
    nb = grid[0]
    S0g = torch.empty((len(windows), L, A), device=dev, dtype=torch.float64)
    S1g = torch.empty((len(windows), L, A), device=dev, dtype=torch.float64)
    S0u = torch.empty((nb, L, A), device=dev, dtype=acc_t)
    S1u = torch.empty((nb, L, A), device=dev, dtype=acc_t)
    for i, s in enumerate(windows):
        sc = scores[s].contiguous()
        Es = E[s].contiguous()
        max_score = float(sc.max().item())
        _whist_block[grid](sc, Es, ctx_i8, S0u, S1u, N, max_score, inv_temp,
                           L=L, A=A, BLOCK=BLOCK, CDT=cdt)
        S0 = S0u.double().sum(0)
        S1 = S1u.double().sum(0)
        sum_e = S0[0].sum()
        norm = scale / sum_e
        S0g[i] = norm * S0
        S1g[i] = norm * S1
    S0h, S1h = S0g.cpu().numpy(), S1g.cpu().numpy()
    return S0h, S1h, S1h[:, 0, :].sum(axis=1)
