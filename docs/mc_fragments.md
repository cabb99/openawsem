# `mc_fragments`: fast Monte-Carlo mutation energies

This document explains how the `mc_fragments` backend computes fragment-memory mutation
energies fast enough for a Monte-Carlo (MC) sequence-design walk, and — the part that usually
confuses people — **what "rebuild the affected windows" means during `commit` and why it is the
expensive step**.

Code: [`openawsem/memory/generators/mc_fragments.py`](../openawsem/memory/generators/mc_fragments.py)

---

## 1. The problem

We hold a protein **structure** fixed (e.g. 1r69) and walk over its **sequence**, proposing
point mutations. For each proposed mutation we need the change in fragment-memory energy, ΔE,
and we need it millions of times. The fragment-memory energy compares the geometry of the fixed
structure against a large **database** of real protein fragments (pc30 ≈ 2.3M fragments, pc80 ≈
11M). Recomputing the full energy per mutation is hopeless. The trick is to precompute the
geometry once and make each mutation a tiny update.

### Notation

| symbol | meaning | typical size |
|--------|---------|--------------|
| `L`    | fragment window length | 9 |
| `W`    | number of structural windows = `n_res - L + 1` | 55 (1r69) |
| `N_db` | number of database fragments | 2.3M (pc30) / 11M (pc80) |
| `ALPHA`| amino-acid alphabet (20 + unknown) | 21 |
| `T`    | softmax temperature (`soft_temp`) | 2.0 |
| `scale`| weight normalization (`n_mem`) | 20 |

A **window** `w` is a length-`L` slice of the target structure (residues `w … w+L-1`). A
**fragment** `f` is a length-`L` slice of some database protein. We compare windows to fragments.

---

## 2. The energy, split into geometry × sequence

The soft fragment-memory energy of the target is a sum over windows:

```
E = -k_fm · Σ_w  E[w]
E[w] = scale · Σ_f  w_f · E_table[w, f]
```

where the sum over `f` runs over **every** database fragment. There are two distinct pieces:

### 2a. `E_table[w, f]` — pure geometry, sequence-independent

```
E_table[w, f] = Σ_(k, sep, pair)  exp( -(r_struct_w[k,sep,pair] - d_db[f+k, sep, pair])² / 2σ² )
```

For each pair of positions inside the window (separated by `sep`, atom pair CA/CB), we take the
distance in the **target structure** (`r_struct`) and in the **database fragment** (`d_db`) and
score their agreement with a Gaussian well. `E_table[w, f]` is high when fragment `f` is
geometrically similar to window `w`. **This depends only on coordinates, never on the
identity of the residues** — so it is computed once, up front, and never changes during the MC
walk. It is the big table: `W × N_db` floats (0.5 GB for pc30, 2.4 GB for pc80).

### 2b. `w_f` — the sequence weighting (softmax over BLOSUM)

```
w_f = scale · exp(score_f / T) / Σ_g exp(score_g / T)
score_f = Σ_(off=0..L-1)  BLOSUM62[ target_seq[w+off], fragment_aa[f, off] ]
```

`score_f` measures how similar fragment `f`'s **sequence** is to the current target window
sequence. The softmax turns these scores into weights that emphasize the most sequence-similar
fragments. **This is the only part that changes when we mutate the sequence.**

So: geometry (`E_table`) is frozen; a mutation only reshuffles the weights `w_f`.

---

## 3. Why a mutation is (almost) free: the S0/S1 bincount trick

A point mutation at sequence position `p` changes `target_seq[p]`. That changes `score_f` only
for windows that **cover** `p`, and within each such window only by a per-fragment factor that
depends on **which amino acid fragment `f` carries at the one affected offset**:

```
new_score_f = score_f + ( BLOSUM[new, aa_f] - BLOSUM[old, aa_f] )
⇒ new_w_f ∝ w_f · g[aa_f]      where  g[a] = exp( (BLOSUM[new,a] - BLOSUM[old,a]) / T )
```

`g` is just a 21-vector (one factor per amino-acid type). The new per-window energy is

```
new E[w] = scale · Σ_f (w_f · g[aa_f]) · E_table[w,f] / Σ_f (w_f · g[aa_f])
```

The sums over millions of fragments collapse if we pre-aggregate the fragments **by the amino
acid they carry at each offset**. Define, for each window `w` and offset `off`:

```
S0[w, off, a] = Σ_{f : fragment_aa[f, off] = a}  w_f
S1[w, off, a] = Σ_{f : fragment_aa[f, off] = a}  w_f · E_table[w, f]
```

These are `(W, L, 21)` tables — tiny. With them, the mutation energy is just two 21-element dot
products:

```
ΔE[w] = scale · (g · S1[w, off]) / (g · S0[w, off])  −  E_wt[w]
```

So **`dE` is O(ALPHA × L) ≈ 189 multiply-adds, independent of database size.** This is why CPU
`dE` is ~35 µs on pc30 *and* ~40 µs on pc80 despite 4.7× more fragments — it never touches the
millions of fragments, only the binned tables.

---

## 4. `commit`: "rebuild the affected windows" — what and why

`dE` only *evaluates* a proposed move. When the MC walk **accepts** a move, we must make it
permanent: update the stored sequence, and update the data structures so the *next* `dE` is
computed against the new sequence. That is `commit`, and it is where the cost lives.

### 4a. Which windows are "affected"?

A mutation at residue `p` changes only windows whose length-`L` span includes `p`. For `L = 9`
there are at most `L = 9` such windows (those starting at `p-8 … p`, clipped at the chain ends).
Every other window's tables are untouched. So `commit` rerebuilds **≈ 9 windows**, not all `W`.

```
residues:      … p-8  p-7  …  p  …  p+? …
windows that      [────────── 9 wide ──────────]   ← each of these 9 windows covers p
cover p:       (start = p-8) … (start = p)
```

### 4b. What "rebuild a window" actually does

For each affected window `s`, `_rebuild_window_tables(s)` recomputes `S0[s]`, `S1[s]`, `E_wt[s]`
**from scratch** for the new sequence:

1. Recompute `score_f` for **all `N_db` fragments** in window `s` (incrementally: add
   `BLOSUM[new, aa_f] − BLOSUM[old, aa_f]` to the stored score of every fragment).
2. Form the softmax weight `w_f` of **all `N_db` fragments** — an `exp` and a normalization over
   millions of values.
3. **Bin** those `N_db` weights into the 21 amino-acid buckets at each of the `L` offsets
   (`np.bincount` on CPU, `scatter_add` on GPU) to refill `S0[s]` and `S1[s]`.
4. Recompute `E_wt[s] = Σ_f w_f · E_table[s, f]` — a dot product over millions of values.

### 4c. Why it is slow

Steps 1–4 each sweep **every one of the `N_db` fragments**, for each of the ~9 affected windows:

```
cost(commit) ≈ 9 windows × N_db × (exp + L bincounts + a dot)  =  O(N_db · L)
```

That is the asymmetry at the heart of this engine:

| operation | complexity | pc30 (2.3M) | pc80 (11M) |
|-----------|-----------|-------------|------------|
| `dE` (evaluate a move) | **O(ALPHA·L)** | 35 µs | 40 µs |
| `commit` (accept a move) | **O(N_db·L)** | 1.6 s (CPU) | 14.7 s (CPU) |

`dE` is database-independent and instant. `commit` re-aggregates millions of fragments and
scales linearly with the database. In a Metropolis walk you call `dE` on every proposal but
`commit` only on accepted moves — still, `commit` dominates wall-clock, and it is the natural
target for the GPU.

#### Where the rebuild time actually goes

Profiling a single window rebuild (pc30, N=2.3M) splits the cost cleanly:

| step | CPU % | GPU % |
|------|------:|------:|
| score update (BLOSUM delta) | 6.6% | 2.2% |
| normalization (`exp` + `max` + `sum`) | 11.7% | 3.1% |
| the dot (`E_wt`) | 4.5% | 2.3% |
| **rebinning (`bincount` / `scatter_add`)** | **73.7%** | **92.3%** |

So the rebinning *is* the cost — not the score update (a per-amino-acid `BLOSUM[·]` precompute
would only touch that ~3–7%), not the softmax normalization, not the dot. The dot is in fact
**redundant**: `E_wt = Σ_a S1[w, off, a]`, so it is read straight off `S1` rather than recomputed.

The rebin runs far below memory-bandwidth peak — its cost is float64 atomic-adds into few bins
plus re-streaming the weight vector once per offset. Two free improvements are applied: deriving
`E_wt` from `S1`, and a single 2D `scatter_add` over all `L` offsets (a zero-copy `expand` view of
the weights — ~17% faster than the per-offset loop). The remaining big lever is precision: a
float32 scatter is ~2.3× faster than float64 but reintroduces a ~0.02 kJ/mol gap in the
`dE == E_after − E_before` identity (§6), so it is left as an opt-in rather than the default.

> **Why rebuild from scratch instead of patching `S0`/`S1` incrementally?** A mutation changes
> the softmax **denominator** `Σ_g exp(score_g/T)`, which couples *all* fragments' weights, not
> just those carrying the mutated amino acid. There is no cheap local patch — every fragment's
> weight shifts. Rebuilding from the (incrementally updated) scores is the clean, correct option.
> A bonus: because `E_wt` is rebuilt exactly each time, the running total energy never drifts.

---

## 5. CPU + GPU split

Profiling makes the design obvious:

- **`dE` stays on the CPU.** It is O(ALPHA·L) ≈ 189 ops. A GPU kernel launch (~10–1000 µs of
  latency) is *slower* than just doing it on the host (35 µs). Measured: GPU single `dE` ≈ 1.3 ms,
  ~35× slower than CPU. **There is nothing to parallelize.**
- **`E_table` precompute and `commit` rebuild go to the GPU.** Both are O(N_db) sweeps over
  millions of fragments — exactly what a GPU eats for breakfast.

Implementation: with `device='cuda'`, `E_table` and `scores` live as torch tensors on the GPU.
`commit` updates the GPU `scores` in place and runs the softmax + `scatter_add` rebuild on the
GPU, then copies the **tiny** `S0[s]`/`S1[s]`/`E_wt[s]` (a `9×21` block) back to host RAM. The
hot `dE` loop then runs on the CPU against those host tables. Best of both worlds.

### Measured timings (1r69, RTX 3080 Ti vs CPU)

| operation | pc30 CPU | pc30 GPU | pc80 CPU | pc80 GPU |
|-----------|---------:|---------:|---------:|---------:|
| precompute `E_table` | 122 s | **2.6 s** (48×) | 648 s | **11 s** (58×) |
| full energy `E` (all windows) | 10.0 s | **1.1 s** (9×) | 99.5 s | **4.9 s** (20×) |
| `commit` (~9 windows) | 1.6 s | **~0.2 s** | 14.7 s | **~0.8 s** |
| `dE` (single move) | **35 µs** | 1.3 ms | **40 µs** | 1.3 ms |

GPU speedup grows with database size, because the fixed overhead amortizes over more work.

---

## 6. Precision

`dE` is *algebraically identical* to `E_after − E_before`. In finite precision they differ only
by floating-point rounding from summing millions of terms in two different orders.

- **CPU**: softmax weights, `S0`/`S1` bincounts, and `E_wt` accumulate in **float64**. `dE` then
  matches `E_after − E_before` to ~1.6e-11 kJ/mol (machine precision).
- **GPU**: same float64 accumulation in the scatter. The per-window scatter is memory-bound, so
  float64 costs only ~2.4× the float32 version (80 vs 33 ms/window at pc80 scale) — still ~20×
  faster than the CPU. GPU and CPU agree to ~1e-6 kJ/mol on total energy and ~7e-3 kJ/mol on
  individual `dE` values (the residual is `E_table` being stored in float32).

`E_table` is deliberately kept in **float32** (halves the 0.5–2.4 GB table and the PCIe/VRAM
traffic); the float64 accumulation happens in the reductions, where it matters.

---

## 7. Known limitation: GLY masking

The conventional fragment memory drops CB-containing pairs at glycine positions, applying the
mask dynamically as the sequence changes. `mc_fragments` excludes **structural** glycines (the
fixed target has no CB there) but does **not** re-mask when a *mutation* turns a position into or
out of GLY. Non-GLY mutations agree with the reference `FullDBMutationEnergy` to float32
precision; GLY mutations carry a ~1 kJ/mol approximation. For most design walks this is
negligible, but it is the one place the energy is not exact.

---

## 8. Usage

```python
from openawsem.memory.generators.mc_fragments import MCFragments
from openawsem.memory.structures import fetch_structure

structure = fetch_structure("1r69")
gen = MCFragments("~/Databases/fragment_db/output_pc30")

# CPU engine
eng = gen.mutation_engine(sequence, structure)

# GPU engine (precompute + commit on CUDA, dE on host)
eng = gen.mutation_engine(sequence, structure, device="cuda")

eng.total_energy()        # current fragment-memory energy (kJ/mol)
eng.dE(position, "A")     # ΔE of mutating 1-based `position` to alanine  (no state change)
eng.commit(position, "A") # accept the mutation; rebuild affected windows
```

`generate(sequence)` still produces standard AWSEM `MemoryWells` (delegated to `LocalDB` soft),
so the same backend serves both force-field generation and MC ΔE.
