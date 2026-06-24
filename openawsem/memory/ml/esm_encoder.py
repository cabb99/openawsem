"""v3 encoder: frozen ESM-2 context features + a small learned head.

The 9-mer is represented by mean-pooling the per-residue ESM-2 embeddings of its L residues (each
already encodes the whole chain's context); a learned head maps that to the L2-normalized retrieval
vector.  Two pathways (see :class:`~openawsem.memory.ml.encoder.Encoder`):

* DB / training -- pool from the precomputed per-residue memmap (:mod:`esm_features`), keyed by
  FragmentIndex rows; only the head runs.
* query -- run ESM-2 on the full query chain, pool per window, then the head.

torch and esm are imported lazily.
"""
from __future__ import annotations

import json
import os

import numpy as np

from openawsem.memory.ml.encoder import Encoder
from openawsem.memory.ml.encoder_torch import build_module, _torch
from openawsem.memory.ml.esm_features import MODELS


class Esm2Encoder(Encoder):
    """Frozen ESM-2 features + learned head -> L2-normalized embedding (v3)."""

    version = "v3"

    def __init__(self, esm_feat=None, *, model_key="t30_150M", layer=None, d_esm=None, L=9,
                 hidden=256, dim=64, module=None, device=None):
        torch = _torch()
        loader, dflt_layer, dflt_dim = MODELS[model_key]
        self.esm_feat = esm_feat                       # (N_res, d_esm) fp16 memmap, or None (query)
        self.loader = loader
        self.model_key = model_key
        self.layer = int(layer if layer is not None else dflt_layer)
        self.d_esm = int(d_esm if d_esm is not None else dflt_dim)
        self.L = int(L)
        self.hidden = int(hidden)
        self.dim = int(dim)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.module = (module or build_module(self.d_esm, self.hidden, self.dim)).to(self.device)
        self._esm_model = None
        self._bc = None

    # ----- pooled features (DB pathway) ---------------------------------- #
    def pooled(self, rows) -> np.ndarray:
        """Mean-pool the L residues of each window from the ESM memmap -> ``(M, d_esm)`` float32."""
        if self.esm_feat is None:
            raise RuntimeError("Esm2Encoder has no precomputed features; build them with "
                               "`python -m openawsem.memory.ml.esm_features`")
        rows = np.asarray(rows)
        gather = self.esm_feat[rows[:, None] + np.arange(self.L)]       # (M, L, d_esm)
        return np.asarray(gather, dtype=np.float32).mean(axis=1)

    def train_inputs(self, fragment_index, rows) -> np.ndarray:
        return self.pooled(rows)

    def forward(self, feats):
        """Differentiable head over pooled features -> L2-normalized embeddings (torch)."""
        torch = _torch()
        x = torch.as_tensor(np.asarray(feats, dtype=np.float32), device=self.device)
        return torch.nn.functional.normalize(self.module(x), dim=1)

    def _encode_feats(self, feats) -> np.ndarray:
        torch = _torch()
        self.module.eval()
        out = []
        with torch.no_grad():
            for c0 in range(0, len(feats), 100_000):
                out.append(self.forward(feats[c0:c0 + 100_000]).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)

    def encode_db(self, fragment_index, rows) -> np.ndarray:
        return self._encode_feats(self.pooled(rows))

    # ----- query pathway (run ESM on the chain) -------------------------- #
    def _esm(self):
        if self._esm_model is None:
            import esm
            model, alphabet = getattr(esm.pretrained, self.loader)()
            self._esm_model = model.to(self.device).eval().half()
            self._bc = alphabet.get_batch_converter()
        return self._esm_model, self._bc

    def residue_features(self, sequence) -> np.ndarray:
        """Per-residue ESM-2 embeddings for a full chain -> ``(len, d_esm)`` float32."""
        torch = _torch()
        model, bc = self._esm()
        _, _, toks = bc([("q", sequence)])
        with torch.no_grad():
            rep = model(toks.to(self.device), repr_layers=[self.layer])["representations"][self.layer]
        return rep[0, 1:1 + len(sequence)].float().cpu().numpy()

    def encode_codes(self, codes) -> np.ndarray:                       # not the v3 pathway
        raise NotImplementedError("v3 encodes from ESM context, not codes; use encode_query")

    def encode_query(self, sequence) -> np.ndarray:
        n = len(sequence) - self.L + 1
        if n <= 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        feat = self.residue_features(sequence)                          # (len, d_esm)
        wins = np.stack([feat[s:s + self.L].mean(axis=0) for s in range(n)])
        return self._encode_feats(wins)

    # ----- persistence (head only) --------------------------------------- #
    def save(self, path):
        torch = _torch()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.module.state_dict(), path)
        with open(path + ".json", "w") as f:
            json.dump({"version": self.version, "model_key": self.model_key, "layer": self.layer,
                       "d_esm": self.d_esm, "L": self.L, "hidden": self.hidden, "dim": self.dim}, f)


def load_esm_encoder(path, esm_feat=None):
    """Load a saved v3 head; pass ``esm_feat`` (the memmap) for the DB pathway, omit for query-only."""
    torch = _torch()
    with open(path + ".json") as f:
        meta = json.load(f)
    enc = Esm2Encoder(esm_feat, model_key=meta["model_key"], layer=meta["layer"],
                      d_esm=meta["d_esm"], L=meta["L"], hidden=meta["hidden"], dim=meta["dim"])
    enc.module.load_state_dict(torch.load(path, map_location=enc.device))
    enc.module.eval()
    return enc
