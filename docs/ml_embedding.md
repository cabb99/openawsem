# Learned-embedding fragment retrieval (ml)

`ml` is a fragment-memory backend that selects database fragments for the AWSEM fragment-memory
potential by a *learned structural metric* instead of by sequence similarity. Each sliding window
of the target is encoded to a unit vector; the most similar database fragments are retrieved by
cosine through an approximate-nearest-neighbour index; and their stored CA/CB distances become the
Gaussian wells of the potential. Encoding by a metric trained to track local *structure* lets the
memory include fragments whose sequence is far from the target but whose backbone geometry is
right — coverage that a BLOSUM sequence search cannot reach.

Code: [`openawsem/memory/generators/ml.py`](../openawsem/memory/generators/ml.py)

The backend implements the standard `FragmentBackend` interface:

* `generate(sequence)` returns AWSEM `MemoryWells`, so the object is a drop-in force-field memory
  generator (`awsem generate-memory --method ml`).
* `mutation_engine(sequence, structure)` returns an `MlMutationEnergy` exposing `total_energy()`,
  `dE(position, new_aa)`, and `commit(position, new_aa)`, for Monte-Carlo sequence design.

## Overview

The AWSEM fragment-memory potential biases a structure toward the local geometry of homologous
fragments: for each window of the target, a set of database fragments is selected, and the pairwise
distances they carry become Gaussian wells acting on the target's atoms. The classic selection rule
is *sequence* similarity (PSI-BLAST, or the BLOSUM62 search in `local_db`). Sequence similarity is a
proxy for structural similarity, and it is an imperfect one: two windows with nearly identical
sequence can adopt different backbone geometry, and two windows with very different sequence can
share a fold. Where the proxy fails, the wrong fragments are selected and the wells pull the
structure the wrong way.

`ml` replaces the sequence-similarity proxy with a metric learned to predict structural similarity
directly. A small neural network encodes a 9-mer to a unit vector `z`, and cosine distance between
encodings is trained to match the difference in the fragments' actual pairwise distances. Retrieval
then asks "which database fragments are *structurally* like this window", answered as a
nearest-neighbour query in the embedding space and accelerated by an ANN index. The retrieved
fragments' distances build wells exactly as before, so the output is an ordinary `MemoryWells` that
flows through the unchanged force field.

The encoder is the only part that changes between variants, behind a fixed pipeline
(fingerprint → metric training → ANN index → wells). This gives an ablation ladder of increasing
power and cost:

* **v0** — no learning: a 9-mer is its concatenated BLOSUM62 rows, retrieved by cosine. The explicit
  "sequence similarity is enough" baseline.
* **v1** — a small MLP on the same BLOSUM features, trained with a metric loss.
* **v3** — a learned head on *frozen ESM-2 per-residue embeddings*, which carry full-chain context.

Only v3 beats the v0 baseline, and it does so decisively (see [Results](#results)): the useful
signal is the protein-language-model *context*, not extra capacity on the bare 9-mer.

### Notation

| symbol | meaning |
|--------|---------|
| $L$ | window / fragment length, default 9 |
| $N$ | number of fragments in the database (≈2.3M for the pc30 set) |
| $x$ | a query 9-mer (the encoder input) |
| $z=f_\theta(x)/\lVert f_\theta(x)\rVert$ | the L2-normalized embedding, $z\in\mathbb{R}^d$ |
| $d$ | embedding dimension ($180$ for v0, $64$ for the learned heads) |
| $s_f=z_q\!\cdot\!z_f$ | cosine similarity between query and database fragment $f$ |
| $\tau$ | temperature for the retrieval weights and the metric loss |
| $N_\text{mem}$ | weight normalization per window (`n_mem`), default 20 |
| $r_{ij},\,r_{ij}^{f}$ | atom-pair distance in the target and in fragment $f$ |
| $\sigma_{ij}=\sigma_0\,\lvert i-j\rvert^{0.15}$ | Gaussian well width |
| $k_{fm}$ | fragment-memory force constant, default 0.04184 kJ/mol |
| $S_r$ | structural fingerprint of fragment $r$ (training signal only) |
| $d_\text{struct}$ | well-width-weighted structural distance between two fragments |

## The retrieval pipeline

At inference the encoder is the entire model; everything downstream is geometry and bookkeeping
shared with the other backends.

![How the ml backend chooses fragments](figures/ml_pipeline.svg)

*The whole idea in five steps: slide a window along the protein, summarize it as a "fingerprint"
vector, find the database fragments with the nearest fingerprints, borrow their measured atom
distances, and add them as soft springs that shape the fold. The only thing that changes between
versions is how the fingerprint is computed — v0 by sequence, v3 by an AI language model that
matches on shape.*

The database vectors $z_f$ are encoded once and stored in the index; a query protein is encoded per
window at generation time. The wells are built by the same soft-accumulation code as `local_db`
(`accumulate_soft_wells`), so the resulting energy is the standard fragment-memory potential

$$
V_\text{FM} \;=\; -\,k_{fm}\sum_{w}\sum_{(i,j)\in w}\;\sum_{f\in\text{top-}k(w)} w_f\,
\exp\!\left(-\frac{\bigl(r_{ij}-r_{ij}^{f}\bigr)^2}{2\,\sigma_{ij}^2}\right),
$$

with the per-window fragment weights set by the retrieval similarity,

$$
w_f \;=\; N_\text{mem}\,\frac{e^{\,s_f/\tau}}{\sum_{g\in\text{top-}k}e^{\,s_g/\tau}} .
$$

Where a window has no confident neighbour ($\max_f s_f$ below `backoff_sim`), it falls back to the
context-free marginal distogram (`tensors_openawsem.npz`, the same tensor `fragment_db` uses),
moment-matched to one Gaussian per pair.

## Structural fingerprint and the training signal

The encoder is trained against a definition of structural similarity computed directly from the
database geometry; this fingerprint is used only during training and never at inference. The
`FragmentIndex` already stores, per residue, the forward distances to its neighbours over the nine
directional atom-pair channels $\{$CA,CB,O$\}\times\{$CA,CB,O$\}$. A fragment's fingerprint is the
vector of all its in-window pair distances,

$$
S_r \;=\; \bigl(\,d^{(c)}_{ab}\,\bigr)_{0\le a<b<L,\;c\in\text{channels}},
\qquad
d^{(c)}_{ab} \;=\; \texttt{dist}\bigl[r+a,\; \mathrm{sep}{=}b{-}a,\; c\bigr],
$$

read by a triangular sweep over the band of $L$ rows (the same reassembly `mc_fragments` uses). The
distance between two fragments is measured in *well-width units*, so it matches the sensitivity of
the energy — a deviation matters in proportion to how narrow that pair's well is,

$$
d_\text{struct}(r,r') \;=\; \bigl\lVert\, w\odot\bigl(S_r-S_{r'}\bigr)\,\bigr\rVert_2,
\qquad
w_{ab} \;=\; \frac{1}{\sigma_{ab}}, \quad \sigma_{ab}=\sigma_0\,(b-a)^{0.15}.
$$

All nine channels (including O) enter the fingerprint, because CA–O and O–O patterns sharpen the
helix/strand distinction at no cost to the metric; only CA/CB wells are emitted into the energy, as
in standard AWSEM. Code: [`openawsem/memory/ml/fingerprint.py`](../openawsem/memory/ml/fingerprint.py).

## The encoders

All encoders map a 9-mer to a unit vector and differ only in $f_\theta$. v0 is parametric-free; v1
and v3 are trained (see [Metric learning](#metric-learning)). The learned head is a small
batch-normalized MLP, `Linear→BN→ReLU→Linear→BN→ReLU→Linear`, hidden width 256, output $d=64$.

![ml encoder architectures v0/v1/v3](figures/ml_encoders.svg)

*The three encoders, each a chain of boxes carrying a vector of numbers from the input (grey) to the
length-1 output `z` (green). Each `Linear→BatchNorm→ReLU` run of the trained head is collapsed into
one **mix** block (blue), with a final **compress** projection; v3 prepends the frozen **ESM-2**
model (orange) and an **average** over the window's 9 residues. Box size scales with √(vector length)
— a length-D vector drawn as roughly a √D × √D square — so a bigger box literally means more numbers.
The figure is generated automatically from the PyTorch modules — forward-hook introspection → a
semantic block spec → [neural-netz](https://typst.app/universe/package/neural-netz/) Typst → SVG — by
[`docs/figures/make_arch_figs.py`](figures/make_arch_figs.py); the intermediate spec is in
[`docs/figures/encoders.json`](figures/encoders.json):*

```
v0 : Input(180) → L2-norm(180)
v1 : Input(180) → Dense(256) → Dense(256) → Linear(64) → L2-norm(64)
v3 : Input(chain) → ESM-2 frozen(640) → mean-pool(640) → Dense(256) → Dense(256) → Linear(64) → L2-norm(64)
```

Formally, with $B$ the BLOSUM62 matrix, $\mathrm{onehot}$ over the 20 amino acids, and $h_i$ the
frozen ESM-2 embedding of residue $i$ *in its full-chain context*,

$$
\text{v0:}\;\; f_\theta(x)=\bigl[\,B[x_1,:]\,\Vert\cdots\Vert\,B[x_L,:]\,\bigr],
\qquad
\text{v1:}\;\; f_\theta(x)=\mathrm{MLP}_\theta\!\bigl(\bigl[\,B[x_1,:]\,\Vert\cdots\Vert\,B[x_L,:]\,\bigr]\bigr),
$$

$$
\text{v3:}\;\; f_\theta(x)=\mathrm{MLP}_\theta\!\left(\frac{1}{L}\sum_{a=0}^{L-1} h_{\,s+a}\right),
\qquad
z=\frac{f_\theta(x)}{\lVert f_\theta(x)\rVert_2}.
$$

The key property of v3 is that each $h_i$ already encodes the whole chain (ESM-2 attends over the
full sequence), so even a window's local embedding carries long-range context that a bare 9-mer
cannot. The ESM-2 features are precomputed once over the database into a memmap aligned to the
`FragmentIndex` rows ([`ml/esm_features.py`](../openawsem/memory/ml/esm_features.py)); the database
encoding then only mean-pools and runs the head, while a query protein is run through ESM-2 once at
generation time. Code: [`ml/encoder.py`](../openawsem/memory/ml/encoder.py),
[`ml/encoder_torch.py`](../openawsem/memory/ml/encoder_torch.py),
[`ml/esm_encoder.py`](../openawsem/memory/ml/esm_encoder.py).

## Metric learning

v1 and v3 are trained so that cosine distance in embedding space tracks the structural distance
$d_\text{struct}$. Training is contrastive (InfoNCE): an anchor is pulled toward a structurally
similar *positive* and pushed away from *negatives*,

$$
\mathcal{L} \;=\; -\,\log\frac{\exp\!\bigl(z_a\!\cdot\!z_p/\tau\bigr)}
{\exp\!\bigl(z_a\!\cdot\!z_p/\tau\bigr)+\sum_{n}\exp\!\bigl(z_a\!\cdot\!z_n/\tau\bigr)} .
$$

The whole value of learning lives in how the pairs are mined, because that is where a learned metric
can beat sequence similarity:

```
   anchor a  ─► f_θ ─► z_a
   positive  =  structural nearest neighbour (small d_struct) on a DIFFERENT chain
                 → structurally close but sequence-far : what BLOSUM would MISS
   hard neg  =  BLOSUM-similar fragment with LARGE d_struct
                 → sequence-similar but structurally wrong : what BLOSUM gets WRONG
                 (in-batch negatives from the other positives are added for free)
```

Positives are found with a FAISS search over the fingerprints $S$; hard negatives with the existing
BLOSUM k-mer search, filtered by $d_\text{struct}$. The chains are split by `chain_code % 7`, with
the held-out seventh never seen in training, so evaluation is on unseen chains. Code:
[`ml/train.py`](../openawsem/memory/ml/train.py).

A note on what failed: v1 (the same training on bare-9-mer BLOSUM features) does *not* beat v0. A
9-mer is too weak a determinant of backbone geometry — two windows with nearly identical sequence
can fold differently, and the encoder, being a deterministic function of sequence, cannot separate
them. v3 succeeds because ESM-2 supplies the surrounding context that breaks that degeneracy.

## Mutation energy for Monte-Carlo

`MlMutationEnergy` ([`mutation.py`](../openawsem/memory/mutation.py)) evaluates the change in
fragment-memory energy under a point mutation on a fixed structure. A mutation at position $p$ alters
only the (at most $L$) windows covering $p$; for each, the engine re-encodes the mutated 9-mer,
re-retrieves its neighbours, rebuilds the wells, and differences the window energy,

$$
\Delta V \;=\; \sum_{w\ \text{covering}\ p}\Bigl(E_w[\text{mutant}] - E_w[\text{WT}]\Bigr),
\qquad
E_w \;=\; -\,k_{fm}\!\!\sum_{(i,j)\in w}\sum_{f} w_f\,
\exp\!\left(-\frac{(r_{ij}-r_{ij}^{f})^2}{2\sigma_{ij}^2}\right).
$$

Unlike the BLOSUM-additive engines (`mc_fragments`, `fulldb`), the learned similarity is *not*
additive over sequence positions, so there is no closed-form reweighting: the covering windows must
be re-embedded and re-searched. This is fast only if retrieval is fast, which is why the engine uses
the **HNSW** index (below). It reaches about 234 Monte-Carlo trials per second end-to-end on 1r69
(against about 2–3/s with an exact flat index), and `dE` agrees with a full mutant-memory recompute
to about $10^{-5}$. The cheap v0 encoder is used for design; a v3 move would require an ESM-2 forward
pass per step, so v3 is a generation-time coverage tool rather than a high-throughput design metric.

### The ANN index

`EmbeddingIndex` ([`retrieval/embedding.py`](../openawsem/memory/retrieval/embedding.py)) stores the
database vectors $z_f$ and answers top-$k$ cosine queries with two backends:

| backend | recall | per query (2.3M db) | use |
|---------|--------|---------------------|-----|
| flat (`IndexFlatIP`, exact) | 1.000 | ~40 ms | the ΔE oracle; small databases |
| HNSW (`IndexHNSWFlat`, graph) | 0.999 (top-20) | ~0.09 ms | generation and Monte-Carlo |

The HNSW graph takes a few minutes to build, so it is persisted with `faiss.write_index` and
reloaded; the flat index rebuilds instantly from the stored vectors. A numpy matmul fallback covers
the case where FAISS is absent (v0 then works without any extra dependency).

## Results

All on the pc30 database (≈2.3M fragments at 30 % sequence-identity cull). The held-out *gate* is the
decisive metric-quality test: on unseen chains, among BLOSUM-similar fragment pairs (where sequence
search is confusable), does embedding distance separate the structurally close from the structurally
far? Reported as AUC.

| encoder | input | hard-negative AUC | Spearman($d_\text{emb},d_\text{struct}$) |
|---------|-------|:-----------------:|:----------------------------------------:|
| v0 BLOSUM-cosine | 9-mer | 0.572 | 0.150 |
| v1 BLOSUM-MLP | 9-mer | 0.565 | — |
| **v3 ESM-2 150M** | 9-mer in chain context | **0.748** | **0.477** |
| **v3 ESM-2 650M** | 9-mer in chain context | **0.801** | **0.578** |

The win is monotonic in model size, and **adversarially verified** by four independent checks:
robust across 7 seeds (margin ≈0.17); it needs *both* ESM context *and* the trained head (a v1
control trained the same way on BLOSUM scores 0.542, below v0; raw mean-pooled ESM with no head only
0.585); and it reflects real geometry, not just an AUC — v3 retrieves geometrically closer fragments
(weighted neighbour native-distance RMSE 0.271 nm vs v0 0.370 nm). No leakage: the head trains on
disjoint chains and the structural labels are database coordinates ESM never sees. Caveat: the split
is chain-level, not homology-level.

### Folding

The ultimate test is whether better retrieval folds proteins better. On a panel of **8 external
proteins, none in the database** (so an honest out-of-distribution test), each folded from extended
with v0 and v3 memories (2 replicates, 8×10⁶ steps), the tail fold quality $Q$ is:

| target | class | v0 | v3 | Δ |
|--------|-------|:---:|:---:|:---:|
| 1enh | α | 0.30 | **0.67** | +0.37 |
| 2gb1 | β | 0.40 | **0.72** | +0.32 |
| 2cro | α | 0.32 | **0.49** | +0.17 |
| 1r69 | α | 0.42 | 0.50 | +0.08 |
| 1ctf | α/β | 0.33 | 0.40 | +0.07 |
| 1ubq | α/β | 0.63 | 0.63 | 0.00 |
| 1shg | β | 0.60 | 0.59 | −0.01 |
| 1csp | β | 0.29 | 0.28 | −0.01 |
| **mean** | | **0.412** | **0.534** | **+0.122** |

v3 wins clearly on 5/8 and loses on none; the large wins are targets where v0 *fails to fold*
($Q\approx0.3$–$0.4$) and v3 reaches near-native. The ties are saturated cases (both already fold, or
both fail). 1r69 alone — the protein used in earlier single-target tests — is exactly such a
saturated case, which is why the single-target comparison was inconclusive and the multi-target
panel was needed. A cheap CPU retrieval screen (the neighbour-distance RMSE above) predicts the fold
direction and can pre-select targets before paying for folding.

### Summary

* **v0** ships as the dependency-light default; its wells are numerically ≈ `local_db` (per-pair
  energy correlation 0.94–0.99) and it beats the context-free marginal.
* **v1** (a learned metric on the bare 9-mer) buys nothing — a confirmed negative, not a tuning miss.
* **v3** (a head on frozen ESM-2 context) is the real result: it wins the gate by a wide,
  size-scaling margin, retrieves geometrically better fragments, and improves multi-target folding
  by +0.12 mean $Q$ without regressing any target.
* The learned metric also powers a working Monte-Carlo `dE` engine (~234 trials/s via HNSW), separate
  from the BLOSUM-additive `mc_fragments`/`fulldb` engines.

## Usage

```bash
conda activate openawsem_torch          # openawsem + torch (CUDA) + faiss + fair-esm

# v0 — no training, works anywhere:
python -m openawsem.memory.cli --method ml --encoder v0 \
    --database ~/Databases/fragment_db/output_pc30 \
    --sequence target.fasta --n-mem 20 --out frags_ml_v0.json

# v3 — precompute ESM-2 features once, train the head, then generate:
python -m openawsem.memory.ml.esm_features <db> --model t30_150M --out <esm_dir>
python -m openawsem.memory.ml.train <db> --encoder v3 --esm-dir <esm_dir> \
    --esm-model t30_150M --out enc_v3.pt
```

```python
from openawsem.memory.generators.ml import Ml

# Monte-Carlo sequence design with the learned ΔE (HNSW index for speed):
gen = Ml("~/Databases/fragment_db/output_pc30", encoder="v0", index_type="hnsw")
eng = gen.mutation_engine(sequence, structure)
eng.total_energy()         # current fragment-memory energy
eng.dE(position, "A")      # energy change of a trial mutation (no state change)
eng.commit(position, "A")  # accept the mutation
```

Key constructor parameters of `Ml`:

| parameter | default | role |
|-----------|---------|------|
| `database` | | `fragdb` output directory with a built `fragment_index/` |
| `encoder` | `"v0"` | `"v0"`, or an `Esm2Encoder`/path for a learned encoder |
| `embedding_index` | `None` | directory of a prebuilt `EmbeddingIndex` (else built in memory) |
| `index_type` | `"flat"` | `"flat"` (exact) or `"hnsw"` (fast, for generation / Monte-Carlo) |
| `top_k` | 20 | fragments retrieved per window |
| `temp` | 0.07 | retrieval-weight temperature $\tau$ |
| `n_mem` | 20 | weight normalization $N_\text{mem}$ |
| `min_seq_sep`, `max_seq_sep` | 3, 9 | pair separation range (effective cap $L-1$) |
| `well_width` | 0.1 | $\sigma_0$ in $\sigma_{ij}=\sigma_0\lvert i-j\rvert^{0.15}$ (nm) |
| `backoff`, `backoff_sim` | `True`, 0.3 | fall back to the marginal distogram for low-similarity windows |
| `k_fm` | 0.04184 | fragment-memory force constant (kJ/mol) |
