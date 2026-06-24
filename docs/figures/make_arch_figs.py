"""Generate architecture figures for the ml encoders, automatically from the PyTorch models.

Pipeline (as recommended): PyTorch model -> forward hooks -> a *semantic* architecture description
(consecutive Linear/Norm/activation/Dropout collapsed into one named block) -> neural-netz Typst
source -> vector SVG via `typst`.

The semantic spec is dumped to encoders.json (the inspectable intermediate); the Typst is written to
ml_encoders.typ and compiled to ml_encoders.svg.

Run:  CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo> python docs/figures/make_arch_figs.py
      typst compile docs/figures/ml_encoders.typ docs/figures/ml_encoders.svg
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from openawsem.memory.ml.encoder_torch import MLPEncoder
from openawsem.memory.ml.esm_encoder import Esm2Encoder

HERE = os.path.dirname(__file__)
ACT = {"ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU"}
NORM = {"BatchNorm1d", "LayerNorm", "GroupNorm"}
ABSORB = ACT | NORM | {"Dropout"}


def trace_leaf_ops(seq, in_dim):
    """Forward hooks on leaf modules -> ordered [(op_name, out_features)] for a dummy batch."""
    events, handles = [], []
    for m in seq.modules():
        if not list(m.children()) and m is not seq:
            handles.append(m.register_forward_hook(
                lambda mod, _i, out: events.append((type(mod).__name__, int(out.shape[-1])))))
    seq.eval()
    with torch.no_grad():
        seq(torch.zeros(2, in_dim))
    for h in handles:
        h.remove()
    return events


def collapse(events):
    """Collapse Linear(+Norm/Act/Dropout) runs into named blocks; last (no activation) = projection."""
    blocks = []
    cur = None
    for op, w in events:
        if op == "Linear":
            if cur:
                blocks.append(cur)
            cur = {"width": w, "ops": ["Linear"], "act": False}
        elif cur is not None and op in ABSORB:
            cur["ops"].append(op)
            cur["act"] = cur["act"] or op in ACT
    if cur:
        blocks.append(cur)
    for b in blocks:
        b["kind"] = "Dense" if b["act"] else "Linear"   # Dense = Linear+Norm+Act collapsed
    return blocks


def head_blocks(encoder, in_dim):
    return collapse(trace_leaf_ops(encoder.module, in_dim))


def build_specs():
    """Full semantic spec per encoder: a list of named blocks (the inspectable intermediate)."""
    v1 = MLPEncoder(L=9, hidden=256, dim=64)
    v3 = Esm2Encoder(esm_feat=None, model_key="t30_150M", d_esm=640, L=9, hidden=256, dim=64)
    specs = {
        "v0  (BLOSUM-cosine, no training)": [
            {"kind": "Input", "label": "9-mer\\nBLOSUM62 rows", "width": 180},
            {"kind": "Norm", "label": "L2-normalize", "width": 180, "out": "z in R^180"},
        ],
        "v1  (BLOSUM-MLP, trained)": [
            {"kind": "Input", "label": "9-mer\\nBLOSUM62 rows", "width": 180},
            *head_blocks(v1, 180),
            {"kind": "Norm", "label": "L2-normalize", "width": 64, "out": "z in R^64"},
        ],
        "v3  (ESM-2 context + trained head)": [
            {"kind": "Input", "label": "chain\\nsequence", "width": 0},
            {"kind": "PLM", "label": "ESM-2\\n(frozen)", "width": 640},
            {"kind": "Pool", "label": "mean-pool\\n9-mer", "width": 640},
            *head_blocks(v3, 640),
            {"kind": "Norm", "label": "L2-normalize", "width": 64, "out": "z in R^64"},
        ],
    }
    return specs


# ----- neural-netz Typst emitter ----------------------------------------- #
TYPE = {"Input": "custom", "PLM": "convres", "Pool": "fc", "Dense": "fc", "Linear": "fc",
        "Norm": "custom"}
FILL = {"Input": "#9aa7b8", "PLM": "#e0884e", "Pool": "#c0c8d4", "Dense": "#6fa8dc",
        "Linear": "#3d6fb4", "Norm": "#7bc47f"}


def layer_typst(b, i):
    base = b.get("label") or b["kind"]
    if b["kind"] == "Norm":
        label = f'{base}\\n{b.get("out","")}'
    elif b.get("width"):
        label = f'{base}\\n({b["width"]})'
    else:
        label = base
    keys = [f'type: "{TYPE[b["kind"]]}"', f'name: "L{i}"', f'label: "{label}"',
            f'fill: rgb("{FILL[b["kind"]]}")', "offset: 2.0"]
    if b["kind"] in ("Dense", "Linear"):
        keys += ["height: 6", "depth: 0"]            # 2D slab; dimension is in the label
    elif b["kind"] == "PLM":
        keys += ["height: 11", "depth: 4"]           # 3D slab to mark the big frozen model
    else:
        keys += ["height: 6", "depth: 0"]
    return "    (" + ", ".join(keys) + "),"


def emit_typst(specs):
    out = ['#import "@preview/neural-netz:0.3.0": draw-network', "",
           "#set page(width: auto, height: auto, margin: 14pt, fill: white)",
           '#set text(font: "DejaVu Sans", size: 9pt)', ""]
    for title, blocks in specs.items():
        out.append(f"#text(weight: \"bold\", size: 11pt)[{title}]")
        out.append("#v(2pt)")
        out.append("#draw-network((")
        out += [layer_typst(b, i) for i, b in enumerate(blocks)]
        out.append('), palette: "warm", scale: 90%)')
        out.append("#v(16pt)")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    specs = build_specs()
    with open(os.path.join(HERE, "encoders.json"), "w") as f:
        json.dump(specs, f, indent=2)
    with open(os.path.join(HERE, "ml_encoders.typ"), "w") as f:
        f.write(emit_typst(specs))
    print("semantic spec -> encoders.json")
    for k, v in specs.items():
        print(f"  {k}: " + " -> ".join(
            f'{b["kind"]}({b["width"]})' if b.get("width") else b["kind"] for b in v))
    print("typst source -> ml_encoders.typ")
