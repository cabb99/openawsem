# Fast Fragment-memory energy calculation (mc_fragments)

`mc_fragments` is a fragment-memory backend that computes the AWSEM fragment-memory energy of a
fixed structure, and the change in that energy under a point mutation of its sequence, from a
fragment database precomputed once. The mutation energy is obtained in time proportional to the
fragment length, independent of the database size, which makes the backend suitable for repeated
evaluation such as Monte-Carlo sequence design.

Code: [`openawsem/memory/generators/mc_fragments.py`](../openawsem/memory/generators/mc_fragments.py)

The backend implements the standard `FragmentBackend` interface:

* `generate(sequence)` returns AWSEM `MemoryWells`, so the same object serves as a force-field
  memory generator (delegated to `LocalDB` soft).
* `mutation_engine(sequence, structure)` returns an `MCMutationEngine` exposing `total_energy()`,
  `dE(position, new_aa)`, and `commit(position, new_aa)`, matching `FullDBMutationEnergy`.

## Overview

The AWSEM fragment-memory potential biases a structure toward the local geometry of homologous
fragments. For each sliding window of the target sequence, a set of database fragments is selected
by sequence similarity, and the pairwise distances they carry become Gaussian wells acting on the
corresponding atoms of the target. Building this potential requires a database search per window,
which is repeated whenever the sequence changes. In a sequence-design walk the sequence changes at
every step, so the search cost dominates.

`mc_fragments` removes that cost by separating the two ingredients of the energy. The geometry, how
well a database fragment matches the distances of a structural window, depends only on the fixed
coordinates and is computed once for the whole database. Concretely, the target structure is divided
into windows, and the fragment-memory energy of every database fragment placed on every window is
evaluated and stored. This per-fragment energy table is the quantity $G_{wf}$ introduced below; it
is sequence-independent, so it is built a single time. The sequence enters only afterward, through a
Boltzmann weighting of fragments by their BLOSUM62 similarity to the current window. A point mutation
therefore leaves the stored energies untouched and only reweights the fragments. By aggregating the
fragments by the amino acid they carry at each window position, the energy change of a mutation
collapses to a small dot product whose size is set by the fragment length and the number of amino
acid types, not by the number of fragments in the database.

The result is a single precomputed model that answers three questions cheaply:

* the total fragment-memory energy of the current sequence on the fixed structure,
* the energy change $\Delta E$ of a trial mutation, without altering any state,
* the update that advances the model to an accepted mutation.

On 1r69 (a 63 residue protein, giving 55 windows of length 9) against a database of 2.3 million
fragments, a trial $\Delta E$ takes about 35 microseconds and is independent of database size, while
the one-time precompute and the per-acceptance update scale with the database and are accelerated on
a GPU (see [Parallelization](#parallelization)).

### Notation

| symbol | meaning |
|--------|---------|
| $L$ | fragment length (residues per window), default 9 |
| $N$ | number of fragments in the database |
| $N_w$ | number of windows of the target, $n_\text{res}-L+1$ |
| $N_\text{mem}$ | number of memories per window (`n_mem`), the weight normalization, default 20 |
| $S_f$ | BLOSUM62 score of database fragment $f$ against a window |
| $T$ | scoring temperature (`soft_temp`) |
| $G_{wf}$ | stored fragment-memory energy of fragment $f$ on window $w$ (the geometry table) |
| $r_{ij},\ r_{ij}^{f}$ | atom-pair distance in the target and in fragment $f$ |
| $\sigma_{ij}$ | Gaussian well width, $\sigma_{ij}=\sigma_0\,|i-j|^{0.15}$ |
| $k_{fm}$ | fragment-memory force constant, default 0.04184 kJ/mol |

A window is a length $L$ slice of the target structure. A fragment is a length $L$ slice of a
database protein. A single chain (a sequence string) is supported per engine.

## Fragment-memory energy and soft scoring

### The conventional potential

The AWSEM fragment-memory energy ([`templateTerms.py`](../openawsem/functionTerms/templateTerms.py))
sums Gaussian wells over selected memory fragments $m$ and over the residue pairs $(i,j)$ inside
each window,

$$
E_\text{FM} \;=\; -\,k_{fm}\sum_{m}\sum_{(i,j)} w_m\,\gamma_{ij}\,
\exp\!\left(-\frac{\bigl(r_{ij}-r_{ij}^{m}\bigr)^2}{2\,\sigma_{ij}^2}\right),
$$

with $\sigma_{ij}=\sigma_0\,|i-j|^{0.15}$ and $w_m$ the per-memory weight. The pair sum runs over
residues with sequence separation between `min_seq_sep` and `max_seq_sep`. The fragment length and
the maximum separation are distinct parameters: a window of length $L$ contains pairs of separation
up to $L-1$, so `max_seq_sep` is effectively capped at $L-1$ (with the defaults $L=9$ and
`max_seq_sep=9`, the largest separation actually used is 8).

### The stored per-fragment energy

`mc_fragments` keeps this energy but replaces the discrete selection of memories by a continuous
weighting of all database fragments. Collect the geometric part of a window $w$ against a database
fragment $f$ into a single quantity,

$$
G_{wf} \;=\; \sum_{(i,j)\in w}\gamma_{ij}\,
\exp\!\left(-\frac{\bigl(r_{ij}-r_{ij}^{f}\bigr)^2}{2\,\sigma_{ij}^2}\right),
$$

where $r_{ij}$ is read from the fixed target structure and $r_{ij}^{f}$ from fragment $f$. The
quantity $G_{wf}$ is the fragment-memory energy, apart from the constant $k_{fm}$ and the sequence
weight, of using fragment $f$ as the single memory of window $w$. It depends only on coordinates, so
the array $G$ over all windows and fragments is the stored table of per-fragment energies, computed
once. Its size is $N_w\times N$ in float32, about 0.5 GB for a 2.3 million fragment database on 1r69
and about 2.4 GB for an 11 million fragment database. Atom pairs are restricted to CA and CB;
structural glycines, which lack a CB in the fixed target, are dropped at this stage.

The full energy is then the weighted sum over the stored table,

$$
E_\text{FM} \;=\; -\,k_{fm}\sum_{w}\,\Phi_w,\qquad \Phi_w=\sum_{f} w_f\,G_{wf},
$$

where $\Phi_w$ is the weighted energy of window $w$ and $w_f$ are the soft weights below.

### Soft Boltzmann scoring

Each fragment is scored against the window by summed BLOSUM62
([`retrieval/blosum.py`](../openawsem/memory/retrieval/blosum.py)),

$$
S_f \;=\; \sum_{o=0}^{L-1} B\bigl[q_{w,o},\,c_{f,o}\bigr],
$$

where $B$ is the BLOSUM62 matrix, $q_{w,o}$ is the query residue at offset $o$ of window $w$, and
$c_{f,o}$ is the residue of fragment $f$ at offset $o$. The fragments are weighted by a Boltzmann
factor of their score, normalized so the weights of a window sum to $N_\text{mem}$,

$$
w_f \;=\; N_\text{mem}\,\frac{e^{\,S_f/T}}{\sum_g e^{\,S_g/T}},
\qquad \sum_f w_f = N_\text{mem}.
$$

The normalization makes the soft memory carry the same total weight as a conventional set of
$N_\text{mem}$ unit-weight fragments, so a window contributes the depth of roughly $N_\text{mem}$
memories. The temperature $T$ sets how that weight is distributed: as $T\to 0$ the weight collapses
onto the single most similar fragment, and as $T$ grows the weight spreads toward a database
average. [Choosing the scoring temperature](#choosing-the-scoring-temperature) discusses the value.

## Database architecture

The database is a sequence-identity-culled set of experimental structures (for example a PISCES
`cullpdb` set at 30 percent identity for a non-redundant database, or 80 percent for a denser one),
reduced to a per-residue table and then to a binary index.

### Forward distance storage and triangular retrieval

Each residue is stored once, with its identity, its forward sequence context (the residues ahead of
it), and its forward distances: the distances from its CA, CB, and O atoms to the CA, CB, and O atoms
of residues up to a maximum separation ahead. Storing distances only in the forward direction avoids
holding each pair twice. The distance between residues $i$ and $j$ with $i<j$ lives in row $i$ at
separation $j-i$. The per-residue table (`residues.csv`) therefore groups its columns as provenance,
identity, forward context, and forward distances:

| chain_uid | internal_index | segment_id | parent1 | ctx_p0 … ctx_p8 | d_s1_CA_CA | d_s1_CA_CB | d_s1_CB_CB | … | d_s8_CB_CB |
|-----------|---------------:|-----------:|:-------:|:---------------:|-----------:|-----------:|-----------:|:-:|-----------:|
| 1r69_A | 1 | 0 | S | S Q E S V K E F L | 3.80 | 4.93 | 6.41 | … | 10.7 |
| 1r69_A | 2 | 0 | Q | Q E S V K E F L A | 3.81 | 5.02 | 6.55 | … | NaN |
| 1r69_A | 3 | 0 | E | E S V K E F L A G | 3.79 | 4.88 | 6.38 | … | NaN |

The `ctx_p0 … ctx_p8` columns hold the forward sequence (the residue and the eight ahead of it), and
the `d_s{sep}_{X}_{Y}` columns hold one distance per separation `sep` and atom pair `(X, Y)`, for the
nine directional pairs $\{$CA, CB, O$\}\times\{$CA, CB, O$\}$. A `NaN` marks a partner past the end
of the chain or segment.

A fragment is a band of $L$ consecutive rows. Its pairwise distances are reassembled in a triangular
sweep: from the row at offset $a$, read the forward distances at separations $1,2,\dots,L-1-a$, which
are exactly the partners that stay inside the fragment. For a fragment of length 4 starting at row
$r$,

| row | forward separations inside the fragment | partners |
|-----|------------------------------------------|----------|
| $r$   | 1, 2, 3 | $r{+}1,\ r{+}2,\ r{+}3$ |
| $r{+}1$ | 1, 2 | $r{+}2,\ r{+}3$ |
| $r{+}2$ | 1 | $r{+}3$ |
| $r{+}3$ | (none) | |

which yields the $\binom{4}{2}=6$ pairs once each. A length 9 fragment yields $\binom{9}{2}=36$ pairs
the same way. Of the nine directional atom pairs in each stored distance, the four CA/CB combinations
enter the energy. The `segment_id` column increments across chain breaks and gaps, so a fragment
never spans a discontinuity.

### The binary index

Reading the per-residue CSV into a dataframe is tens of GB and minutes for the full PDB, so a
one-time build converts it to a `FragmentIndex`
([`retrieval/index.py`](../openawsem/memory/retrieval/index.py)) of NumPy arrays:

```
python -m openawsem.memory.retrieval.index <db_dir>
```

| array | dtype | shape | axes | content |
|-------|-------|-------|------|---------|
| `ctx` | int8 | $(N, C)$ | residue, context offset | amino acid index at each forward offset (segment edge as 127) |
| `dist` | float32 | $(N, S, 9)$ | residue, separation, atom pair | forward pair distances in Angstrom |
| `chain_code` | int32 | $(N,)$ | residue | source chain id |
| `internal_index` | int32 | $(N,)$ | residue | residue index within its chain |
| `segment_id` | int32 | $(N,)$ | residue | contiguous-segment id |
| `chain_offset` | int64 | $(n_\text{chains}{+}1,)$ | chain | first row of each chain (CSR ranges) |

On load, `dist` is memory-mapped and only `ctx`, a few hundred MB, is held resident, so even the full
PDB opens in seconds. The engine works with the subset of rows whose forward context is valid for
length $L$ (no segment edge), giving `valid_rows` and a $(N, L)$ array of fragment residue identities
used for both scoring and aggregation.

## Mutation evaluation

A trial mutation at position $p$ samples the energy change $\Delta E$ without altering any state. It
changes the query residue at one offset of each window covering $p$, of which there are at most $L$.
Within such a window, the score of fragment $f$, written $S_f$, shifts by a term that depends only on
the amino acid the fragment carries at the affected offset $o$,

$$
S_f \;\to\; S_f + \bigl(B[\text{new},c_{f,o}] - B[\text{old},c_{f,o}]\bigr)
\quad\Longrightarrow\quad
w_f \;\to\; w_f\,g_{c_{f,o}},\qquad
g_a = \exp\!\left(\frac{B[\text{new},a]-B[\text{old},a]}{T}\right),
$$

so the reweighting factor $g$ is a vector of one entry per amino acid type. To evaluate the reweighted
energy without touching the $N$ fragments, aggregate them by the amino acid at each offset. For every
window $w$, offset $o$, and amino acid type $a$, define the weight sum $S^0$ and the weighted-energy
sum $S^1$,

$$
S^0_{w,o,a}=\!\!\sum_{f:\,c_{f,o}=a}\!\! w_f,
\qquad
S^1_{w,o,a}=\!\!\sum_{f:\,c_{f,o}=a}\!\! w_f\,G_{wf}.
$$

These tables have shape $(N_w, L, n_\text{aa})$ and are kept on the host. The weighted energy of a
window, $\Phi_w$, reads directly off them, $\Phi_w=\sum_a S^1_{w,o,a}$ (the same for any offset), and
the reweighted energy after the mutation is two dot products of the factor $g$ with these tables,

$$
\Phi_w' \;=\; N_\text{mem}\,\frac{\sum_a g_a\,S^1_{w,o,a}}{\sum_a g_a\,S^0_{w,o,a}},
\qquad
\Delta E \;=\; -\,k_{fm}\!\!\sum_{w\ \text{covering}\ p}\!\!\bigl(\Phi_w'-\Phi_w\bigr).
$$

With $g$ the identity, this returns $\Phi_w$, so the formula is exact at zero change. The work is
proportional to $L\,n_\text{aa}$, about 189 multiply-adds, independent of $N$. This is why a trial
$\Delta E$ is about 35 microseconds on a 2.3 million fragment database and about 40 microseconds on an
11 million fragment database. The `dE` call reads state but does not modify it, so any number of
trials can be evaluated between acceptances.

The engine therefore holds two database-sized arrays, indexed by window and fragment, and three small
aggregation arrays that a trial reads:

| array | symbol | dtype | shape | axes | content |
|-------|--------|-------|-------|------|---------|
| `E_table` | $G_{wf}$ | float32 | $(N_w, N)$ | window, fragment | stored per-fragment energy of fragment $f$ on window $w$ |
| `scores` | $S_f$ | int16/int32 | $(N_w, N)$ | window, fragment | BLOSUM score of each fragment against the window |
| `S0` | $S^0_{w,o,a}$ | float64 | $(N_w, L, n_\text{aa})$ | window, offset, amino acid | weight sum per amino acid |
| `S1` | $S^1_{w,o,a}$ | float64 | $(N_w, L, n_\text{aa})$ | window, offset, amino acid | weighted-energy sum per amino acid |
| `E_wt` | $\Phi_w$ | float64 | $(N_w,)$ | window | weighted energy of each window |

The energy table $G$ has the same shape as `scores`, one float per window and database fragment, and
holds the precomputed energies described above; on 1r69 it is about 0.5 GB at 2.3 million fragments
and 2.4 GB at 11 million. A trial reads only the aggregation arrays $S^0$, $S^1$, and $\Phi_w$, which
are a few values per window, so its cost is set by $L$ and $n_\text{aa}$ and not by $N$. An acceptance,
in contrast, rebuilds the aggregation arrays from $G$ and `scores`, which is where the database size
enters.

## Mutation acceptance

When a trial is accepted, `commit(position, new_aa)` makes it permanent. It records the new residue,
then rebuilds the aggregation tables of the affected windows so that subsequent `dE` calls are taken
against the new sequence. Only the windows covering the mutated position $p$ change, at most $L$ of
them; every other window keeps its tables, and the stored per-fragment energies $G_{wf}$ never change,
since they are sequence-independent.

Rebuilding one window means recomputing its fragment scores, weights, and aggregation tables from the
updated sequence, in four steps:

1. **Update the scores.** Add the BLOSUM difference $B[\text{new},c_{f,o}] - B[\text{old},c_{f,o}]$ to
   the stored score $S_f$ of each fragment in the window. Only the single mutated offset $o$
   contributes, so this is one additive pass over the fragments.
2. **Reweight.** Form the Boltzmann weights $w_f$ from the updated scores (an exponential and a
   normalization back to $N_\text{mem}$).
3. **Re-aggregate.** Accumulate the weight sum $S^0$ and the weighted-energy sum $S^1$ over the
   fragments, binned by the amino acid $c_{f,o}$ at each offset (a bincount on the CPU, a scatter-add
   on the GPU). This refills the tables that `dE` reads.
4. **Read the window energy.** Recover the weighted energy of the window as $\Phi_w=\sum_a S^1_{w,0,a}$.

Each step sweeps all $N$ fragments of each affected window, so an acceptance costs $O(N\,L)$. The
re-aggregation in step 3 dominates, about three quarters of the CPU time and over nine tenths of the
GPU time: it is a set of floating-point additions into a small number of amino acid bins, bound by
memory bandwidth rather than by arithmetic. Because the weighted energy $\Phi_w$ is rebuilt from the
fragments rather than patched incrementally, the running total energy does not drift over a long walk.

| operation | cost | 2.3M fragments | 11M fragments |
|-----------|------|----------------|---------------|
| `dE` (evaluate a trial) | $O(L\,n_\text{aa})$ | 35 us | 40 us |
| `commit` (accept a move) | $O(N\,L)$ | 1.6 s (CPU) | 14.7 s (CPU) |

The asymmetry, a constant-time evaluation against a database-sized acceptance, is intrinsic. A
mutation shifts the softmax denominator $\sum_g e^{S_g/T}$, which couples the weights $w_f$ of all
fragments, not only those carrying the mutated amino acid, so there is no cheap local patch to $S^0$
and $S^1$; rebuilding them from the updated scores is both correct and drift-free.

### Truncated commit for Monte-Carlo throughput

In a Metropolis walk `dE` runs on every proposal and `commit` only on accepted moves, yet the
acceptance cost still sets the throughput ceiling. Setting `commit_topk=K` rebuilds the aggregation
tables over only the $K$ highest-weight fragments per window, reselected at each acceptance, so the
re-aggregation shrinks from $N$ to $K$ while `dE` is unchanged. The selection uses the integer score
$S_f$, which is monotonic in the weight $w_f$, so the exponential runs over $K$ fragments rather than
all $N$, and the survivors are renormalized to $N_\text{mem}$.

This is an approximation whose accuracy is governed jointly by $K$ and the temperature $T$, because
$T$ controls how much weight sits outside the top $K$ (see the next section). At a low temperature a
small $K$ reproduces the full-database energy closely; at the default temperature the weight has a
longer tail and a larger $K$ is needed. The residual cost of an acceptance is then the score update
and the top-$K$ selection, both of which still read every fragment once, so a very large database
would call for an incremental selection.

## Parallelization

The two database-sized operations, the one-time geometry precompute and the per-acceptance commit,
run on the GPU when `device='cuda'`; the trial $\Delta E$, which is independent of database size,
stays on the CPU.

### Device split

A trial $\Delta E$ is about 189 operations on arrays of a few values per window. A GPU kernel launch
has more latency (of order 0.1 to 1 ms) than the whole computation takes on the host (about 35 us), so
there is nothing to gain on the GPU: a GPU `dE` measures about 1.3 ms, slower than the CPU by more than
an order of magnitude. The geometry precompute and the commit rebuild, in contrast, are sweeps over the
database-sized arrays $G$ and `scores`, which the GPU streams efficiently. The split is therefore to
keep `dE` on the host and place the precompute and the commit on the GPU. On the GPU, $G$ and the
integer scores are held as tensors; the `dE` loop reads the small aggregation arrays, which are copied
back to the host after each commit.

Measured on 1r69, RTX 3080 Ti against a CPU:

| operation | 2.3M CPU | 2.3M GPU | 11M CPU | 11M GPU |
|-----------|---------:|---------:|--------:|--------:|
| precompute $G$ | 122 s | 2.6 s (48x) | 648 s | 11 s (58x) |
| full energy (all windows) | 10.0 s | 1.1 s (9x) | 99.5 s | 4.9 s (20x) |
| commit (about 9 windows, full database) | 1.6 s | 0.2 s (8x) | 14.7 s | 0.9 s (16x) |
| `dE` (single trial) | 35 us | 1.3 ms | 40 us | 1.3 ms |

The GPU advantage grows with database size, as the fixed launch overhead is amortized over more work.
The precompute, which is an independent Gaussian sum per window and fragment, parallelizes almost
perfectly and gets the largest speedup. The commit gains less, and the reason is worth stating, since
its steps look parallelizable.

### Why acceptance is bandwidth-bound

The re-aggregation that dominates a full-database commit is a histogram: it adds each fragment's weight
into one of $n_\text{aa}$ amino acid bins per offset. The arithmetic is trivial, one add per fragment,
so the operation is not limited by compute. It is limited by two things that parallelism does not
remove. First, it must stream the full weight vector and energy vector, of length $N$, through memory
once per window; at millions of fragments this is a memory-bandwidth wall, and once the bandwidth is
saturated, adding threads does not help. Second, the additions land in only $n_\text{aa}$ bins, so many
threads contend for the same accumulator and serialize. Profiling a single full-database window rebuild
shows the histogram is the cost, while the score update, the softmax, and the energy readback are
minor:

| step | CPU share | GPU share |
|------|----------:|----------:|
| score update (BLOSUM delta) | 7% | 2% |
| reweighting (exp, max, sum) | 12% | 3% |
| energy readback | 4% | 2% |
| re-aggregation (bincount or scatter-add) | 74% | 92% |

The re-aggregation takes an even larger share on the GPU, because the other steps speed up on the GPU
while the histogram stays bandwidth-bound.

### What has been optimized

Several changes reduced the constant factors without changing the result:

* The window energy is read as $\Phi_w=\sum_a S^1_{w,0,a}$ instead of a second pass over the fragments,
  removing one full sweep.
* The GPU re-aggregation is a single scatter over all $L$ offsets at once, using a zero-copy view of
  the weight vector, about 17 percent faster than a per-offset loop.
* The aggregation blocks of all affected windows are copied to the host in one transfer rather than one
  per window, removing about 18 synchronizations per commit, each about 0.4 ms.
* The aggregation accumulates in float64. A float32 scatter is about 2.3 times faster but reopens a
  gap of about 0.02 kJ/mol in the identity $\Delta E = E_\text{after}-E_\text{before}$, so float64 is
  the default and float32 is left as an option.

### Contention-free histogram (the exact full-database commit)

The dominant cost above is atomic contention: the `scatter_add` sends $N\cdot L$ additions into only
$n_\text{aa}$ global bins, which serialize, so it runs at about 2 percent of peak memory bandwidth. When
`triton_commit=True` (the default on a CUDA or ROCm device, for the exact `commit_topk=None` path), the
scatter is replaced by a Triton block-partial histogram: each program reduces its tile of fragments by
amino acid with shared-memory tree reductions (no atomics), writes its own $(L, n_\text{aa})$ partial,
and the partials are summed. There is no shared accumulator, so there is no contention, and the result
is deterministic. The kernel also reads $\texttt{ctx}$ as int8 rather than int64 and recovers the softmax
denominator from the histogram (no separate sum-exp pass), so the surrounding $O(N)$ traffic shrinks too.

The aggregation precision is set by `hist_dtype`: `f32` reduces each tile in float32 into a float64 total
(the trial $\Delta E$ stays within about $10^{-5}$ kJ/mol of the float64 scatter, inside the mutation
tolerance) and `f64` accumulates entirely in float64 (bit-exact $\Delta E = E_\text{after}-E_\text{before}$).
On an RTX 3080 Ti this lowers the exact full-database commit on 1r69 from about 215 ms to about 7 ms in
`f32` (a factor of 30, roughly 140 commits per second) and to about 54 ms in `f64`; on the larger
11-million-fragment database the commit drops from about 1.0 s to about 29 ms (a factor of 36). The
histogram is now compute-bound on the masked reductions rather than contention-bound, so further gains
would come from the reduction itself.

With the truncated commit (`commit_topk`, see [Mutation acceptance](#mutation-acceptance)) the
histogram is no longer the bottleneck, because it now runs over $K$ fragments instead of $N$. At
`soft_temp=1.0` and $K=10^4$ the remaining commit cost is the work that is still proportional to $N$:
the BLOSUM score update over all fragments (about 56 percent), the host transfer (about 28 percent),
and the top-$K$ selection over all fragments (about 19 percent). Selecting the top $K$ on the integer
score rather than the weight lets the exponential run over $K$ fragments only, and the batched transfer
removes the per-window synchronizations; together these raised the walk from about 2400 to about 3860
accepted moves per second and cut a commit from 5.7 to 3.1 ms, with the trial $\Delta E$ unchanged.

### The remaining limit

Even with the histogram truncated, every commit reads all $N$ fragments at least once, to apply the
score update and to select the top $K$. That irreducible scan is the floor, and it scales with the
database, so it is most visible on the larger database (the 11 million fragment commit is about 4.7
times the 2.3 million one). The open lever is an incremental top-$K$ that avoids rescanning every
fragment at each acceptance; an attempt to instead correct the dropped tail with a rank-1 background
term recovered about 70 percent of the truncation error at the default temperature but stalled near 6
kJ/mol, so lowering the temperature remains the cleaner route.

## Choosing the scoring temperature

The temperature $T$ sets the concentration of the Boltzmann weights $w_f\propto e^{S_f/T}$. A low
temperature places the weight on the few fragments most similar in sequence, giving a sharp,
target-specific geometry from genuine homologs. A high temperature spreads the weight across many
fragments, approaching a database-average distogram that washes out the local geometry. Two
considerations set the value.

Folding 1r69 with the soft memory favors a low temperature:

| memory | 1r69 fold $Q$ |
|--------|---------------|
| conventional fragment memory | 0.515 |
| hard top 20 | 0.517 |
| soft, $T=3$ | 0.497 |
| soft, $T=2$ (default) | 0.555 |
| soft, $T=1$ | 0.754 |

The truncated commit also favors a low temperature, because $T$ controls how much weight sits in the
tail beyond the top $K$ fragments. At $T=2$ the tail is heavy and $K$ of order $2\times10^5$ is needed
to match the full database; at $T=1$ the weight concentrates and $K$ of order $10^4$ is near-exact
(the trial $\Delta E$ differs from the full-database value by about 0.07 kJ/mol, and the internal
consistency $\Delta E = E_\text{after}-E_\text{before}$ holds to about 0.01 kJ/mol). Both
considerations point the same way, so for design the recommended setting is `soft_temp=1.0` with
`commit_topk=10_000`, giving roughly 2400 accepted moves per second. The same `soft_temp` should be
used for `generate` and for the mutation engine, so the folded potential and the design objective are
the same energy function.

## Precision

The trial $\Delta E$ is algebraically equal to $E_\text{after}-E_\text{before}$; in finite precision
they differ only by the order of summation over millions of terms. On the CPU the weights, aggregation
tables, and window energies accumulate in float64, and $\Delta E$ matches the difference of total
energies to about $1.6\times10^{-11}$ kJ/mol. On the GPU the scatter accumulates in float64 as well;
since the per-window scatter is bandwidth-bound, float64 costs about 2.4 times the float32 version and
still runs about 20 times faster than the CPU. The GPU and CPU agree to about $10^{-6}$ kJ/mol on
total energy and about $7\times10^{-3}$ kJ/mol on individual trials, the residual coming from the
energy table being held in float32. The energy table is kept in float32 deliberately, to halve its
size and the GPU memory traffic, while the float64 accumulation is confined to the reductions where it
matters.

## Glycine masking

The conventional fragment memory drops CB-containing pairs at glycine positions and reapplies the mask
as the sequence changes. `mc_fragments` excludes structural glycines, which lack a CB in the fixed
target, but does not reapply the mask when a mutation turns a position into or out of glycine.
Non-glycine mutations agree with the reference `FullDBMutationEnergy` to float32 precision; glycine
mutations carry an approximation of about 1 kJ/mol. For most design walks this is negligible, but it
is the one place the energy is not exact.

## Usage

```python
from openawsem.memory.generators.mc_fragments import MCFragments
from openawsem.memory.structures import fetch_structure

structure = fetch_structure("1r69")
db = "~/Databases/fragment_db/output_pc30"

# Exact full-database engine (CPU)
eng = MCFragments(db).mutation_engine(sequence, structure)

# Exact full-database engine (GPU)
eng = MCFragments(db, device="cuda").mutation_engine(sequence, structure)

# Fast Monte-Carlo engine: low temperature with a truncated commit
gen = MCFragments(db, device="cuda", soft_temp=1.0, commit_topk=10_000)
eng = gen.mutation_engine(sequence, structure)

eng.total_energy()         # current fragment-memory energy (kJ/mol)
eng.dE(position, "A")      # energy change of mutating 1-based position to alanine (no state change)
eng.commit(position, "A")  # accept the mutation and rebuild the affected windows
```

Key constructor parameters:

| parameter | default | role |
|-----------|---------|------|
| `database` | | `fragdb` output directory with a built `fragment_index/` |
| `fragment_length` | 9 | fragment length $L$ |
| `well_width` | 0.1 | Gaussian well width, $\sigma_{ij}=\texttt{well\_width}\,|i-j|^{0.15}$ in nm |
| `soft_temp` | 2.0 | scoring temperature $T$; use 1.0 for fast Monte-Carlo |
| `n_mem` | 20 | weight normalization $N_\text{mem}$ |
| `min_seq_sep`, `max_seq_sep` | 3, 9 | sequence separation range for pairs (effective cap $L-1$) |
| `k_fm` | 0.04184 | fragment-memory force constant (kJ/mol) |
| `device` | `None` | `None` for CPU; `'cuda'` to run precompute and commit on the GPU |
| `commit_topk` | `None` | `None` for an exact full-database commit; $K$ for a truncated commit |
| `triton_commit` | `True` | use the contention-free Triton histogram for the exact GPU commit (falls back to the scatter on CPU / no triton / when `commit_topk` is set) |
| `hist_dtype` | `'f32'` | Triton histogram precision: `'f32'` (fast, $\Delta E$ within ~$10^{-5}$ kJ/mol) or `'f64'` (bit-exact) |
| `soft_pool` | 80 | candidate pool size per window for `generate` |
| `soft_seed_word` | 2 | exact seed length for the `generate` retrieval |
| `chunk_size`, `gpu_chunk` | 50000, 400000 | fragments per chunk in the CPU and GPU precompute |
