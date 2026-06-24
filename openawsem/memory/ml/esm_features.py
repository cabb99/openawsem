"""Precompute per-residue ESM-2 embeddings (in full-chain context) for a fragment database.

For every residue in a :class:`~openawsem.memory.retrieval.index.FragmentIndex`, store the ESM-2
embedding of that residue computed within its contiguous chain segment, aligned to the index rows
(``esm[row]`` is the embedding of the residue at global row ``row``).  A v3 window encoder later
mean-pools the L residues of each fragment from this memmap.

Reconstruction uses the index alone: rows are grouped by ``chain_offset`` and ordered by
``internal_index`` within a chain; ``_DECODE[ctx[row, 0]]`` is the residue's own letter;
``segment_id`` marks contiguous segments (ESM runs per segment, not across chain breaks).

Run (in ``openawsem_torch``)::

    python -m openawsem.memory.ml.esm_features <fragdb_dir> --model t30_150M --out <dir>
    python -m openawsem.memory.ml.esm_features <fragdb_dir> --model t33_650M --out <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

from openawsem.memory.retrieval import FragmentIndex
from openawsem.memory.retrieval.index import _DECODE

logger = logging.getLogger(__name__)

MODELS = {
    "t30_150M": ("esm2_t30_150M_UR50D", 30, 640),
    "t33_650M": ("esm2_t33_650M_UR50D", 33, 1280),
}
ESM_MAX = 1022                       # ESM-2 context window (excl. BOS/EOS)


def segments(index):
    """Yield ``(sequence, rows)`` per contiguous chain segment of ``index``.

    ``rows`` are the global FragmentIndex rows of the segment's residues (in order); ``sequence``
    is their one-letter string.  Rows within a chain are already in ``internal_index`` order."""
    ctx0 = np.asarray(index.ctx[:, 0])
    seg = np.asarray(index.segment_id)
    off = np.asarray(index.chain_offset)
    for c in range(len(off) - 1):
        lo, hi = int(off[c]), int(off[c + 1])
        if hi <= lo:
            continue
        s = seg[lo:hi]
        bnd = np.flatnonzero(np.diff(s) != 0) + 1
        starts = np.concatenate([[0], bnd, [hi - lo]])
        for a, b in zip(starts[:-1], starts[1:]):
            rows = np.arange(lo + a, lo + b)
            seq = "".join(_DECODE[ctx0[rows]])
            yield seq, rows


def _chunks(seq, rows):
    """Split a segment longer than ESM_MAX into non-overlapping <=ESM_MAX pieces."""
    if len(seq) <= ESM_MAX:
        yield seq, rows
        return
    for i in range(0, len(seq), ESM_MAX):
        yield seq[i:i + ESM_MAX], rows[i:i + ESM_MAX]


def _run_batch(model, bc, batch, layer, device):
    import torch
    data = [(str(k), seq) for k, (seq, _) in enumerate(batch)]
    _, _, toks = bc(data)
    toks = toks.to(device)
    with torch.no_grad():
        rep = model(toks, repr_layers=[layer])["representations"][layer]
    return rep.float().cpu().numpy()


def precompute(database, model_key="t30_150M", out=None, *, l2_budget=2_500_000, max_batch=128,
               device="cuda"):
    import torch
    import esm

    loader_name, layer, dim = MODELS[model_key]
    idx = FragmentIndex.load(database)
    N = len(np.asarray(idx.segment_id))
    out = out or os.path.join(database, "esm")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"esm_{model_key}.npy")
    emb = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(N, dim))

    logger.info("loading ESM-2 %s ...", loader_name)
    model, alphabet = getattr(esm.pretrained, loader_name)()
    bc = alphabet.get_batch_converter()
    model = model.to(device).eval().half()

    pieces = [c for seq, rows in segments(idx) for c in _chunks(seq, rows)]
    pieces.sort(key=lambda sr: len(sr[0]))         # length-sorted -> tight batches
    logger.info("%d residues in %d segment-pieces; embedding with %s (layer %d, dim %d)",
                N, len(pieces), loader_name, layer, dim)

    done = i = 0
    while i < len(pieces):
        maxlen = len(pieces[i][0])
        # attention is O(batch * L^2); size the batch by an L^2 budget, OOM-resilient (halve & retry)
        bsz = max(1, min(max_batch, l2_budget // max(maxlen * maxlen, 1)))
        while True:
            batch = pieces[i:i + bsz]
            try:
                rep = _run_batch(model, bc, batch, layer, device)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bsz == 1:
                    raise
                bsz = max(1, bsz // 2)
        for j, (seq, rows) in enumerate(batch):
            emb[rows] = rep[j, 1:1 + len(seq)].astype(np.float16)   # strip BOS; per-residue
            done += len(seq)
        i += len(batch)
        if done % 100_000 < maxlen * bsz or i >= len(pieces):
            logger.info("  %d/%d residues embedded", done, N)
    emb.flush()
    with open(os.path.join(out, f"esm_{model_key}.json"), "w") as f:
        json.dump({"model": loader_name, "model_key": model_key, "layer": layer, "dim": dim,
                   "n": int(N)}, f)
    logger.info("wrote %s (%d x %d fp16)", path, N, dim)
    return path


def load_features(database, model_key, out=None):
    """Memmap ``(N_residues, dim)`` fp16 ESM features + meta, aligned to FragmentIndex rows."""
    out = out or os.path.join(database, "esm")
    with open(os.path.join(out, f"esm_{model_key}.json")) as f:
        meta = json.load(f)
    feat = np.load(os.path.join(out, f"esm_{model_key}.npy"), mmap_mode="r")
    return feat, meta


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="python -m openawsem.memory.ml.esm_features")
    ap.add_argument("database")
    ap.add_argument("--model", default="t30_150M", choices=list(MODELS))
    ap.add_argument("--out", default=None, help="output dir (default <database>/esm)")
    ap.add_argument("--l2-budget", type=int, default=2_500_000,
                    help="batch sized so batch*maxlen^2 <= this (OOM-resilient; lower under GPU contention)")
    a = ap.parse_args(argv)
    precompute(a.database, a.model, a.out, l2_budget=a.l2_budget)


if __name__ == "__main__":
    main()
