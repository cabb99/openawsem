"""v4 (generative): predict the fragment-memory Gaussian wells directly from ESM-2 context.

Where v0/v3 *retrieve* real fragments and copy their stored distances, v4 *predicts*, for each
in-window residue pair and CA/CB atom channel, a ``K``-component Gaussian mixture over the pair
distance -- with **no database lookup at inference**.  Because the fragment-memory energy already
sums every well sharing a ``(resid_i,name_i,resid_j,name_j)`` group, a ``K``-component mixture is
literally ``K`` wells with ``weight_k = n_mem*pi_k``, ``r_mean=mu_k``, ``sigma=sigma_k``; v4 is the
context-conditioned generalization of :meth:`~openawsem.memory.generators.ml.Ml._accumulate_backoff`
(which emits one Gaussian per pair from a *context-free* marginal table).

The head is a shared per-pair MLP over ``[h_a, h_b, sep_embed]`` (the two residues' frozen ESM-2
vectors plus a learned separation embedding); four independent output heads, one per directional
CA/CB channel.  ``sigma`` is fixed to the AWSEM law ``well_width*sep**0.15`` by default (a soft
histogram over distance); ``free_sigma=True`` predicts it with a floor.  torch is imported lazily.
"""
from __future__ import annotations

import json
import os

import numpy as np

from openawsem.memory.ml.encoder_torch import _torch
from openawsem.memory.ml.fingerprint import in_window_pairs
from openawsem.memory.retrieval.index import ATOM_PAIRS

#: directional CA/CB atom-pair channels that become wells (GLY drops its CB downstream).
CACB_PAIRS = (("CA", "CA"), ("CA", "CB"), ("CB", "CA"), ("CB", "CB"))
#: their column index into the 9-channel ``FragmentIndex.dist`` / ``fingerprint.assemble`` layout.
CACB_CHANNELS = [ATOM_PAIRS.index(p) for p in CACB_PAIRS]      # [0, 1, 3, 4]
_NM_PER_A = 0.1
MU_LO, MU_HI = 0.2, 2.5                                        # nm; mixture mean range


def _build_head(d_esm, *, n_sep, e_sep, hidden, trunk_out, K, n_channels, free_sigma,
                window_pool=False):
    """Construct the per-pair mixture head (lazy torch); ``n_sep`` indexes by raw separation.

    With ``window_pool`` the head also sees the window-mean ESM vector (a non-local summary of the
    whole 9-mer) alongside the two residues' vectors."""
    torch = _torch()
    nn = torch.nn
    params = 3 if free_sigma else 2                            # (logit_pi, mu[, log_sigma]) per comp
    in_dim = (3 if window_pool else 2) * d_esm + e_sep

    class _GMHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.window_pool = window_pool
            self.sep_embed = nn.Embedding(n_sep, e_sep)
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, trunk_out), nn.ReLU(),
            )
            self.heads = nn.ModuleList([nn.Linear(trunk_out, K * params) for _ in range(n_channels)])

        def forward(self, ha, hb, sep_idx, wctx=None):
            parts = [ha, hb] + ([wctx] if self.window_pool else []) + [self.sep_embed(sep_idx)]
            h = self.trunk(torch.cat(parts, dim=-1))
            return [head(h).view(-1, K, params) for head in self.heads]   # n_channels x (M, K, params)

    return _GMHead()


class DistogramModel:
    """Generative Gaussian-mixture distogram over CA/CB pair distances, conditioned on ESM-2 context."""

    version = "v4"

    def __init__(self, esm_feat=None, *, model_key="t30_150M", d_esm=640, L=9, K=4,
                 min_seq_sep=3, max_seq_sep=9, well_width=0.1, e_sep=32, hidden=512, trunk_out=256,
                 free_sigma=False, window_pool=False, module=None, device=None):
        torch = _torch()
        self.esm_feat = esm_feat                              # (N_res, d_esm) memmap, or None (query)
        self.model_key = model_key
        self.d_esm = int(d_esm)
        self.L = int(L)
        self.K = int(K)
        self.min_seq_sep = int(min_seq_sep)
        self.max_seq_sep = int(max_seq_sep)
        self.well_width = float(well_width)
        self.e_sep = int(e_sep)
        self.hidden = int(hidden)
        self.trunk_out = int(trunk_out)
        self.free_sigma = bool(free_sigma)
        self.window_pool = bool(window_pool)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pa, pb, ps = in_window_pairs(self.L, min_sep=self.min_seq_sep, max_sep=self.max_seq_sep)
        self.pair_a, self.pair_b, self.pair_sep = pa, pb, ps  # fixed layout, length P
        self.sigma_fixed = self.well_width * self.pair_sep.astype(np.float64) ** 0.15   # (P,) nm
        self.n_channels = len(CACB_PAIRS)
        self.module = module or _build_head(
            self.d_esm, n_sep=self.max_seq_sep + 1, e_sep=self.e_sep, hidden=self.hidden,
            trunk_out=self.trunk_out, K=self.K, n_channels=self.n_channels, free_sigma=self.free_sigma,
            window_pool=self.window_pool)
        self.module = self.module.to(self.device)

    # ----- ESM context (DB pathway: per-residue, NOT pooled) ----------------- #
    def context_db(self, rows) -> np.ndarray:
        """Per-residue ESM context of each fragment window -> ``(M, L, d_esm)`` float32."""
        if self.esm_feat is None:
            raise RuntimeError("DistogramModel has no precomputed ESM features; build them with "
                               "`python -m openawsem.memory.ml.esm_features`")
        rows = np.asarray(rows)
        gather = self.esm_feat[rows[:, None] + np.arange(self.L)]          # (M, L, d_esm)
        return np.asarray(gather, dtype=np.float32)

    # ----- forward: mixtures for a batch of window contexts ------------------ #
    def forward(self, ctx):
        """``ctx`` ``(M, L, d_esm)`` torch -> ``(pi, mu, sigma)`` each ``(M, P, C, K)`` torch.

        ``pi`` sums to 1 over ``K``; ``mu``/``sigma`` in nm.  ``C`` = 4 CA/CB channels."""
        torch = _torch()
        M = ctx.shape[0]
        pa = torch.as_tensor(self.pair_a, dtype=torch.long, device=ctx.device)
        pb = torch.as_tensor(self.pair_b, dtype=torch.long, device=ctx.device)
        sep = torch.as_tensor(self.pair_sep, dtype=torch.long, device=ctx.device)
        ha = ctx[:, pa, :].reshape(M * len(pa), self.d_esm)               # (M*P, d_esm)
        hb = ctx[:, pb, :].reshape(M * len(pb), self.d_esm)
        P = len(pa)
        sep_idx = sep.repeat(M)                                           # (M*P,)
        wctx = None
        if self.window_pool:
            wctx = ctx.mean(dim=1)[:, None, :].expand(M, P, self.d_esm).reshape(M * P, self.d_esm)
        outs = self.module(ha, hb, sep_idx, wctx)                        # C x (M*P, K, params)
        pis, mus, sigs = [], [], []
        sig_fixed = torch.as_tensor(self.sigma_fixed, dtype=torch.float32, device=ctx.device)
        for c, out in enumerate(outs):
            out = out.view(M, P, self.K, -1)
            pi = torch.softmax(out[..., 0], dim=-1)                       # (M, P, K)
            mu = MU_LO + (MU_HI - MU_LO) * torch.sigmoid(out[..., 1])
            if self.free_sigma:
                floor = 0.5 * sig_fixed[None, :, None]
                sigma = floor + torch.nn.functional.softplus(out[..., 2])
            else:
                sigma = sig_fixed[None, :, None].expand(M, P, self.K)
            pis.append(pi); mus.append(mu); sigs.append(sigma)
        stack = lambda xs: torch.stack(xs, dim=2)                         # -> (M, P, C, K)
        return stack(pis), stack(mus), stack(sigs)

    # ----- inference: one window -> per-(pair,channel) mixtures -------------- #
    def predict_window(self, context):
        """``context`` ``(L, d_esm)`` -> numpy ``(pi, mu, sigma)`` each ``(P, C, K)``."""
        torch = _torch()
        self.module.eval()
        with torch.no_grad():
            ctx = torch.as_tensor(np.asarray(context, dtype=np.float32)[None], device=self.device)
            pi, mu, sigma = self.forward(ctx)
        return (pi[0].cpu().numpy(), mu[0].cpu().numpy(), sigma[0].cpu().numpy())

    def residue_features(self, sequence):
        """Run ESM-2 once on a query chain -> ``(len, d_esm)`` (reuses the v3 query path)."""
        from openawsem.memory.ml.esm_encoder import Esm2Encoder
        enc = Esm2Encoder(None, model_key=self.model_key, d_esm=self.d_esm, L=self.L,
                          device=self.device)
        return enc.residue_features(sequence)

    # ----- persistence (head only) ------------------------------------------ #
    def save(self, path):
        torch = _torch()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.module.state_dict(), path)
        with open(path + ".json", "w") as f:
            json.dump({"version": self.version, "model_key": self.model_key, "d_esm": self.d_esm,
                       "L": self.L, "K": self.K, "min_seq_sep": self.min_seq_sep,
                       "max_seq_sep": self.max_seq_sep, "well_width": self.well_width,
                       "e_sep": self.e_sep, "hidden": self.hidden, "trunk_out": self.trunk_out,
                       "free_sigma": self.free_sigma, "window_pool": self.window_pool}, f)


def load_distogram(path, esm_feat=None):
    """Load a saved v4 head; pass ``esm_feat`` (memmap) for the DB pathway, omit for query-only."""
    torch = _torch()
    with open(path + ".json") as f:
        m = json.load(f)
    model = DistogramModel(esm_feat, model_key=m["model_key"], d_esm=m["d_esm"], L=m["L"], K=m["K"],
                           min_seq_sep=m["min_seq_sep"], max_seq_sep=m["max_seq_sep"],
                           well_width=m["well_width"], e_sep=m["e_sep"], hidden=m["hidden"],
                           trunk_out=m["trunk_out"], free_sigma=m["free_sigma"],
                           window_pool=m.get("window_pool", False))
    model.module.load_state_dict(torch.load(path, map_location=model.device))
    model.module.eval()
    return model


# --------------------------------------------------------------------------- #
# Targets and loss
# --------------------------------------------------------------------------- #
def cacb_targets(index, rows, L, *, min_seq_sep=3, max_seq_sep=9):
    """Per-fragment observed CA/CB pair distances + finite-mask -> ``(M, P, C)`` nm, both float32.

    Reads the four CA/CB channels of ``fingerprint.assemble``'s 9-channel output; NaN (segment
    edge or absent GLY CB) -> ``mask=False``.  ``P`` and the (a,b,sep) layout match
    :meth:`DistogramModel.forward`."""
    from openawsem.memory.ml import fingerprint as fp
    S, mask, _ = fp.assemble(index, rows, L, min_sep=min_seq_sep, max_sep=max_seq_sep)   # (M,P,9) nm
    d = S[:, :, CACB_CHANNELS].astype(np.float32)
    m = mask[:, :, CACB_CHANNELS]
    return d, m


def mixture_nll(pi, mu, sigma, target, mask, *, pair_w=None, eps=1e-8):
    """Masked, 1/sigma-weighted negative log-likelihood of ``target`` under the mixture (torch).

    ``pi/mu/sigma`` ``(M,P,C,K)``; ``target/mask`` ``(M,P,C)``.  ``pair_w`` ``(P,)`` optional
    well-width weighting (broadcast over C,K).  Returns ``(mean_nll, comp_entropy)``."""
    torch = _torch()
    t = target[..., None]                                                 # (M,P,C,1)
    log_norm = -0.5 * ((t - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    log_mix = torch.logsumexp(torch.log(pi + eps) + log_norm, dim=-1)     # (M,P,C)
    w = torch.ones_like(log_mix)
    if pair_w is not None:
        w = pair_w[None, :, None].to(log_mix.device) * w
    w = w * mask.float()
    nll = -(w * log_mix).sum() / w.sum().clamp_min(1.0)
    entropy = -(pi * torch.log(pi + eps)).sum(-1)                         # (M,P,C)
    ent = (entropy * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return nll, ent
