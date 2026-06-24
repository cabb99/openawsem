"""Learned MLP encoder (v1): BLOSUM-row features -> small L2-normalized embedding.

Same input as v0 (concatenated BLOSUM62 rows, :func:`encoder.blosum_features`), so v1 is a
strict learned refinement of the v0 baseline: an MLP trained with a metric loss to make cosine
distance track *structural* distance (see :mod:`openawsem.memory.ml.train`).  torch is imported
lazily here so the base env still imports the package.
"""
from __future__ import annotations

import json
import os

import numpy as np

from openawsem.memory.ml.encoder import Encoder, blosum_features


def _torch():
    try:
        import torch
        return torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("the learned encoder needs torch; use the openawsem_torch env") from exc


def build_module(in_dim, hidden, out_dim):
    torch = _torch()
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class MLPEncoder(Encoder):
    """Trainable MLP over BLOSUM-row features; output L2-normalized for cosine retrieval."""

    version = "v1"

    def __init__(self, L=9, hidden=256, dim=64, module=None, device=None):
        torch = _torch()
        self.L = int(L)
        self.in_dim = self.L * 20
        self.hidden = int(hidden)
        self.dim = int(dim)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.module = (module or build_module(self.in_dim, self.hidden, self.dim)).to(self.device)

    def features(self, codes):
        """``(M, L)`` codes -> torch float tensor of BLOSUM-row features on ``self.device``."""
        torch = _torch()
        return torch.as_tensor(blosum_features(codes), dtype=torch.float32, device=self.device)

    def forward(self, codes):
        """Differentiable L2-normalized embeddings (torch tensor) for training."""
        torch = _torch()
        z = self.module(self.features(codes))
        return torch.nn.functional.normalize(z, dim=1)

    def encode_codes(self, codes) -> np.ndarray:
        torch = _torch()
        self.module.eval()
        out = []
        with torch.no_grad():
            for c0 in range(0, len(codes), 200_000):
                out.append(self.forward(codes[c0:c0 + 200_000]).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)

    # ----- persistence --------------------------------------------------- #
    def save(self, path):
        torch = _torch()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.module.state_dict(), path)
        with open(path + ".json", "w") as f:
            json.dump({"version": self.version, "L": self.L, "hidden": self.hidden,
                       "dim": self.dim}, f)


def load_torch_encoder(path, L=9):
    torch = _torch()
    with open(path + ".json") as f:
        meta = json.load(f)
    enc = MLPEncoder(L=meta.get("L", L), hidden=meta["hidden"], dim=meta["dim"])
    enc.module.load_state_dict(torch.load(path, map_location=enc.device))
    enc.module.eval()
    return enc
