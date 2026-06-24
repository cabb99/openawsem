"""Train the v1 metric encoder: mine triplets on structural distance, fit with a triplet loss.

The encoder learns to make cosine distance track *structural* distance ``d_struct`` (well-width
units; :mod:`openawsem.memory.ml.fingerprint`).  Mining is where a learned metric can beat
BLOSUM:

* **positive** = a structural nearest neighbor (small ``d_struct``) on a *different chain* --
  pulls together structurally-similar fragments whose sequences differ.
* **hard negative** = a BLOSUM-similar fragment that is structurally far -- the case sequence
  retrieval gets wrong.

Run (in ``openawsem_torch``)::

    python -m openawsem.memory.ml.train <fragdb_dir> --out <weights.pt> [--n-pool ... --epochs ...]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from openawsem.memory.retrieval import FragmentIndex
from openawsem.memory.ml import fingerprint as fp
from openawsem.memory.ml.encoder import blosum_features
from openawsem.memory.ml.encoder_torch import MLPEncoder

logger = logging.getLogger(__name__)

#: chains with ``chain_code % EVAL_MOD == 0`` are the held-out eval split (never trained on),
#: so the v0-vs-v1 gate (:mod:`scratch...eval_hard_negatives`) is measured on unseen chains.
EVAL_MOD = 7


def is_eval_chain(chain_code):
    return np.asarray(chain_code) % EVAL_MOD == 0


def build_pool(idx, L, n_pool, *, well_width=0.1, min_sep=1, max_sep=None, seed=0, split="train"):
    """Sample valid length-``L`` rows from the ``train``/``eval``/``all`` chain split.

    Returns ``(rows, codes, F, chains)``.  ``F`` is the weighted structural fingerprint (L2 over
    it = ``d_struct``); ``chains`` is the chain code per row (to require sequence-diverse
    positives)."""
    rows_all = np.asarray(idx._encoded_for_length(L)[0])
    eval_mask = is_eval_chain(np.asarray(idx.chain_code)[rows_all])
    if split == "train":
        rows_all = rows_all[~eval_mask]
    elif split == "eval":
        rows_all = rows_all[eval_mask]
    rng = np.random.default_rng(seed)
    if n_pool and n_pool < len(rows_all):
        rows = np.sort(rng.choice(rows_all, size=n_pool, replace=False))
    else:
        rows = np.asarray(rows_all)
    codes = np.asarray(idx.ctx[rows, :L], dtype=np.int16)
    F = fp.weighted_features(idx, rows, L, well_width=well_width, min_sep=min_sep, max_sep=max_sep)
    chains = np.asarray(idx.chain_code)[rows]
    return rows, codes, F, chains


def mine_triplets(codes, F, chains, *, n_anchor, k_struct=10, k_blosum=50, margin_ratio=1.5,
                  pos_keep=0.5, seed=0):
    """Mine ``(anchor, positive, negative)`` index triplets over a pool.

    positive: nearest structural neighbor on a different chain (sequence-diverse, structure-close).
    negative: among BLOSUM-similar fragments, the structurally-farthest one that is at least
    ``margin_ratio`` x the positive's ``d_struct`` away (a genuinely hard, structurally-wrong
    match).  Triplets whose positive ``d_struct`` is above the ``pos_keep`` quantile are dropped,
    so training sees only genuinely structurally-close positives (noisy positives starve the
    metric signal)."""
    import faiss
    n = len(F)
    Fn = np.ascontiguousarray(F, dtype=np.float32)
    V = np.ascontiguousarray(blosum_features(codes), dtype=np.float32)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)

    struct = faiss.IndexFlatL2(Fn.shape[1]); struct.add(Fn)
    blos = faiss.IndexFlatIP(V.shape[1]); blos.add(V)

    rng = np.random.default_rng(seed)
    anchors = rng.choice(n, size=min(n_anchor, n), replace=False)
    _, sNN = struct.search(Fn[anchors], k_struct + 1)            # structural neighbors
    _, bNN = blos.search(V[anchors], k_blosum + 1)               # BLOSUM neighbors

    a_idx, p_idx, n_idx, d_pos_list = [], [], [], []
    for r, a in enumerate(anchors):
        pos = next((j for j in sNN[r] if j != a and chains[j] != chains[a]), None)
        if pos is None:
            continue
        d_pos = float(np.linalg.norm(Fn[a] - Fn[pos]))
        cand = [j for j in bNN[r] if j != a]
        if not cand:
            continue
        d_cand = np.linalg.norm(Fn[a] - Fn[np.asarray(cand)], axis=1)
        order = np.argsort(-d_cand)                              # farthest first
        neg = next((cand[o] for o in order if d_cand[o] > margin_ratio * max(d_pos, 1e-6)), None)
        if neg is None:
            continue
        a_idx.append(a); p_idx.append(pos); n_idx.append(neg); d_pos_list.append(d_pos)
    a_idx, p_idx, n_idx = np.array(a_idx), np.array(p_idx), np.array(n_idx)
    d_pos_arr = np.array(d_pos_list)
    if len(d_pos_arr) and pos_keep < 1.0:
        thr = np.quantile(d_pos_arr, pos_keep)
        keep = d_pos_arr <= thr
        a_idx, p_idx, n_idx = a_idx[keep], p_idx[keep], n_idx[keep]
        logger.info("positive d_struct: median=%.3f kept<=%.3f (%.0f%%)",
                    float(np.median(d_pos_arr)), float(thr), 100 * keep.mean())
    logger.info("mined %d triplets from %d anchors", len(a_idx), len(anchors))
    return a_idx, p_idx, n_idx


def info_nce(za, zp, zn, tau):
    """NT-Xent: for each anchor the positive is its ``zp``; negatives are every other in-batch
    ``zp`` plus all mined hard negatives ``zn``.  In-batch negatives make collapse a high-loss
    state, so the embedding cannot trivially map everything to one point (the triplet failure)."""
    import torch
    B = za.shape[0]
    cands = torch.cat([zp, zn], dim=0)          # (2B, d): positives then hard negatives
    logits = (za @ cands.t()) / tau             # (B, 2B); column i is the positive for anchor i
    target = torch.arange(B, device=za.device)
    return torch.nn.functional.cross_entropy(logits, target)


def make_encoder(encoder, *, database, L, hidden, dim, esm_dir=None, esm_model="t30_150M"):
    """``"v1"`` -> a BLOSUM-MLP encoder; ``"v3"`` -> an ESM-2 head over precomputed features."""
    if encoder == "v1":
        return MLPEncoder(L=L, hidden=hidden, dim=dim)
    if encoder == "v3":
        from openawsem.memory.ml.esm_encoder import Esm2Encoder
        from openawsem.memory.ml.esm_features import load_features
        feat, meta = load_features(database, esm_model, out=esm_dir)
        return Esm2Encoder(feat, model_key=esm_model, layer=meta["layer"], d_esm=meta["dim"],
                           L=L, hidden=hidden, dim=dim)
    raise ValueError(f"encoder must be 'v1' or 'v3', got {encoder!r}")


def train(database, *, encoder="v1", L=9, hidden=256, dim=64, n_pool=300_000, n_anchor=50_000,
          epochs=15, batch=1024, lr=1e-3, tau=0.1, well_width=0.1, min_sep=3, max_sep=9, seed=0,
          esm_dir=None, esm_model="t30_150M", out=None):
    import torch
    idx = FragmentIndex.load(database)
    rows, codes, F, chains = build_pool(idx, L, n_pool, well_width=well_width,
                                        min_sep=min_sep, max_sep=max_sep, seed=seed)
    logger.info("pool=%d fragments; mining triplets ...", len(rows))
    a, p, neg = mine_triplets(codes, F, chains, n_anchor=n_anchor, seed=seed)
    if len(a) == 0:
        raise RuntimeError("no triplets mined; widen the pool or relax margin_ratio")

    enc = make_encoder(encoder, database=database, L=L, hidden=hidden, dim=dim,
                       esm_dir=esm_dir, esm_model=esm_model)
    inputs = enc.train_inputs(idx, rows)               # codes (v1) or pooled ESM features (v3)
    ia, ip, in_ = inputs[a], inputs[p], inputs[neg]
    opt = torch.optim.Adam(enc.module.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        enc.module.train()
        perm = rng.permutation(len(a))
        tot = 0.0
        for b0 in range(0, len(perm), batch):
            sel = perm[b0:b0 + batch]
            if len(sel) < 2:
                continue
            loss = info_nce(enc.forward(ia[sel]), enc.forward(ip[sel]), enc.forward(in_[sel]), tau)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(sel)
        logger.info("epoch %2d/%d  InfoNCE loss %.4f", ep + 1, epochs, tot / len(a))
    if out:
        enc.save(out)
        logger.info("saved encoder -> %s", out)
    return enc


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="python -m openawsem.memory.ml.train")
    ap.add_argument("database")
    ap.add_argument("--out", required=True, help="encoder weights path (.pt)")
    ap.add_argument("--encoder", default="v1", choices=["v1", "v3"])
    ap.add_argument("--esm-dir", default=None, help="v3: dir with precomputed esm_<model>.npy")
    ap.add_argument("--esm-model", default="t30_150M", choices=["t30_150M", "t33_650M"])
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-pool", type=int, default=300_000)
    ap.add_argument("--n-anchor", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    train(a.database, encoder=a.encoder, esm_dir=a.esm_dir, esm_model=a.esm_model,
          hidden=a.hidden, dim=a.dim, n_pool=a.n_pool, n_anchor=a.n_anchor,
          epochs=a.epochs, tau=a.tau, seed=a.seed, out=a.out)


if __name__ == "__main__":
    main()
