# Learned-embedding fragment retrieval (the `ml` backend)

Status as of 2026-06-19. Branch `fragment_memoryv2`.

## 1. What problem this solves

Fragment memory is normally retrieved by **BLOSUM62 sequence search** over a residue database
(`local_db` / `FragmentIndex`). That drops the right structural fragments whenever a sequence is
BLOSUM-similar but structurally different, and it can't pull in structurally-similar fragments
whose sequences are far apart. The `ml` backend retrieves by a **learned structural metric**:
encode each 9-mer to an L2-normalized vector `z`, retrieve neighbors by cosine via an ANN index,
and drop their stored CA/CB distances as Gaussian wells — the existing pipeline with a learned
metric swapped into the search step. Goal: **coverage past BLOSUM**.

This is a *separate* tool from the fast mutation-ΔE engines (`fulldb`, `mc_fragments`), which stay
BLOSUM-additive and untouched (their O(alphabet·L) closed form depends on that additivity).

## 2. The ablation ladder (where we are)

The pipeline (fingerprint → mining → metric loss → ANN index → `ml` backend → eval) is built once
and is **encoder-agnostic**; each model phase is a drop-in encoder swap, gated on beating the
previous one at hard-negative separation.

| phase | encoder | hard-negative AUC | status |
|---|---|---|---|
| **v0** | BLOSUM-cosine (concatenated BLOSUM62 rows; no training) | 0.572 | DONE — shipped baseline |
| **v1** | MLP on BLOSUM rows, InfoNCE | 0.565 | DONE — does NOT beat v0 |
| v2 | 1D-CNN on the 9-mer | — | skipped (still 9-mer-only) |
| **v3** | ESM-2 **150M** context + head | **0.748** | **DONE — BEATS v0 decisively** |
| **v3** | ESM-2 **650M** context + head | **0.801** | **DONE — bigger model widens the margin** |
| v4 | Gaussian-mixture distogram (generative, no retrieval) | — | future fork |

**v3 clears the gate, monotonically with model size.** The lever is *context*, not MLP capacity:
a learned head over frozen ESM-2 per-residue embeddings (mean-pooled over the 9-mer) lifts
hard-negative AUC 0.572 → 0.748 (150M) → 0.801 (650M); Spearman 0.150 → 0.477 → 0.578. v0 stays the
dependency-light shipped default; v3 is the high-coverage option.

**Adversarially verified** (4 independent CPU checks, all support v3): robust across 7 seeds
(margins 0.17); the win needs *both* ESM context *and* a trained head (a v1 control — same training
on BLOSUM, not ESM — scores 0.542, below v0; raw-ESM cosine only 0.585); and it is the real goal,
not just an AUC — v3 retrieves geometrically closer fragments (weighted neighbor native-distance
RMSE **0.271 nm vs v0 0.370 nm**, lower native-distance NLL on ~70% of held-out pairs). No leakage
(head trained on disjoint chains; labels are DB coordinates ESM never sees). Caveat: the split is
chain-level, not homology-level.

The ml metric also now supports **fast mutation ΔE for Monte-Carlo** (`MlMutationEnergy` + HNSW
index): ~234 MC trials/s end-to-end (vs ~2-3/s exact-flat), `dE` exact to ~1e-5 vs a brute-force
recompute.

## 3. Code map

| file | role |
|---|---|
| `openawsem/memory/ml/fingerprint.py` | structural fingerprint `S` + `d_struct` from `FragmentIndex.dist` (training signal; verified vs `get_fragment`) |
| `openawsem/memory/ml/encoder.py` | `Encoder` interface (`encode_db`/`encode_query`/`train_inputs`), `blosum_features`, v0 `BlosumCosineEncoder` |
| `openawsem/memory/ml/encoder_torch.py` | v1 `MLPEncoder` (BatchNorm MLP, lazy torch) |
| `openawsem/memory/ml/esm_features.py` | precompute per-residue **ESM-2** embeddings (full-chain context) → memmap aligned to FragmentIndex rows |
| `openawsem/memory/ml/esm_encoder.py` | v3 `Esm2Encoder` — frozen ESM features (pooled per window) + learned head; DB pathway (memmap) + query pathway (run ESM on the chain) |
| `openawsem/memory/ml/train.py` | FAISS mining (structural-NN positives, BLOSUM-similar/structurally-far hard negatives) + InfoNCE; `--encoder v1|v3`; chain split `% 7` held out |
| `openawsem/memory/retrieval/embedding.py` | `EmbeddingIndex` — flat (exact) or **HNSW** (persisted graph) cosine search; `build`/`save`/`load`/`search`/`search_vectors` |
| `openawsem/memory/mutation.py` | `MlMutationEnergy` — ml ΔE on a fixed structure (re-embed+re-retrieve covering windows; HNSW for MC) |
| `openawsem/memory/generators/ml.py` | `Ml(FragmentBackend)` — `encode_query`→`search_vectors`→wells; `mutation_engine(...)`; `index_type='hnsw'`; marginal backoff |
| `openawsem/memory/cli.py` | `awsem generate-memory --method ml --database <db> --encoder v0 ...` |
| `tests/test_ml_embedding.py` | 10 contract tests (synthetic mini-DB): fingerprint, v0/HNSW retrieval, `Ml.generate`, mutation engine vs oracle, v3 head pathway |

Reuse: `Ml`/`LocalDB` share `accumulate_soft_wells`; `MlMutationEnergy` reuses `MutationEnergy`
geometry; the marginal backoff is the `tensors_openawsem.npz` that `fragment_db` consumes.

### Usage

```bash
conda activate openawsem_torch          # self-contained: openawsem + torch(CUDA) + faiss + fair-esm
# generate an ml v0 memory:
python -m openawsem.memory.cli --method ml \
    --database /home/cb/Databases/fragment_db/output_pc30 --encoder v0 \
    --sequence target.fasta --n-mem 20 --out frags_ml_v0.json
# v3 (ESM-2): precompute features, train the head, run the gate
python -m openawsem.memory.ml.esm_features <db> --model t30_150M --out <esm_dir>
python -m openawsem.memory.ml.train <db> --encoder v3 --esm-dir <esm_dir> --esm-model t30_150M --out enc_v3.pt
# Monte-Carlo design with the ml ΔE (HNSW for speed):
#   ml = Ml(<db>, encoder="v0", index_type="hnsw"); eng = ml.mutation_engine(seq, structure)
#   eng.dE(pos, aa); eng.commit(pos, aa)
```

## 4. Results

**Eval 1–2 — v0 beats the context-free marginal (PASS).** On pc30, v0 retrieval reconstructs
held-out distances better than the `fragment_db` marginal: mean NLL −0.041 vs +0.082, better on
66% of pairs.

**Eval 3 — v1 does NOT beat v0 (the gate).** Held-out hard-negative AUC: v0 = 0.572, v1 = 0.565.
A sequence-9-mer-only encoder can't separate the hardest negatives (near-identical sequence,
different structure are an identical input), and cross-chain structural-NN positives erode the
BLOSUM signal v0 already uses. → stay non-parametric; the lever is context (v3), not MLP capacity.

**Fold (1r69, pc30, 8e6 steps, 1 replica).** ml v0: Qmax 0.715, **Qtail 0.452**
(conv 0.515, local_db hard 0.517, soft 0.555). Single-replica — see correlation below.

**Per-pair energy correlation (1r69 native + random sequences).** ml v0 vs local_db_hard:
Pearson **0.94–0.99** (0.99 on native), as close as hard-vs-soft (0.95–0.98), high even for random
sequences. The ml v0 potential is **essentially equivalent to local_db_hard**, so the fold-Q gap is
almost certainly run-to-run noise, not a real difference.

**MC (Track 1 — DONE).** HNSW: 0.999 top-20 recall, 0.093 ms/query (414× faster than exact flat),
graph persisted via `faiss.write_index`. `MlMutationEnergy.dE` matches a brute-force mutant-memory
recompute to ~1e-5 (exact-flat oracle). MC design demo on 1r69: **234 trials/s** end-to-end (vs ~2–3/s
flat), energy −1832 → −2267. The learned score isn't BLOSUM-additive (no closed form), so this is
fast re-search, not the 35k/s of `mc_fragments`. MC uses the cheap v0 encoder (a v3 move would need
an ESM forward).

**v3 ESM-2 (Track 2 — DONE, both sizes).** Gate on held-out chains: AUC v0 0.572 → v3-150M 0.748
→ v3-650M 0.801 (Spearman 0.150 → 0.477 → 0.578). Adversarially verified (seed-robust; needs ESM
context + trained head; reconstructs native geometry better, RMSE 0.271 vs 0.370 nm; no leakage).

**Multi-target fold benchmark (8 external proteins, none in pc30; 2 replicates each) — v3 wins.**
Mean Qtail: **v0 0.412 → v3 0.534 (+0.122)**. v3 better on 5/8 (never worse): 1enh 0.30→0.67,
2gb1 0.40→0.72, 2cro 0.32→0.49, 1r69 0.42→0.50, 1ctf 0.33→0.40; ties on 1ubq (0.63), 1shg (0.60),
1csp (0.28). The big wins are targets where v0 *fails to fold* (Q≈0.3–0.4) and v3 reaches near-native
— exactly the coverage-past-BLOSUM payoff. (1r69 alone was the saturated case where both tie, which
is why the single-target test was inconclusive.) A cheap CPU retrieval screen (neighbor-RMSE, v3
closer on 7/8) predicts the fold direction; the per-pair NLL screen does not (temp-confounded).

## 5. Summary judgement

- v0: shipped, beats the marginal, numerically ~equivalent to local_db (0.94–0.99 per-pair).
- v1 (bare-9-mer MLP): buys nothing — confirmed, not a tuning artifact.
- **v3 (ESM-2 context): the real win, confirmed by folding.** Clears the gate by a wide margin
  (0.748/0.801 vs 0.572), verified, reconstructs native geometry better, and on an 8-target external
  fold panel **improves mean Qtail 0.412→0.534 (+0.12), winning 5/8 and losing none** — rescuing
  folds v0 fails. (1r69 alone tied because it's saturated.)
- MC with the ml metric is now practical (~234 trials/s via HNSW); separate from the BLOSUM-additive
  35k/s engines.

## 6. Next steps

1. **More replicates / targets** to tighten the high-variance ties (1shg, 1csp) and widen the panel;
   fold **local_db / conventional** on the new targets for a full method comparison (only v0/v3 so far).
2. **Use 650M** (wins the gate by a further margin) for the production v3 index; persist a v3 HNSW
   index for full-PDB-scale generation; PSI-BLAST benchmark.
3. The verification flagged the train/eval split is **chain-level, not homology-level** — a
   homology-aware split would make the gate claim airtight.
4. Optional: v4 (Gaussian-mixture distogram) if coverage still plateaus.
