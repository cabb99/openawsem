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
            {"kind": "Norm", "label": "L2-normalize", "width": 180, "out": "180 numbers"},
        ],
        "v1  (BLOSUM-MLP, trained)": [
            {"kind": "Input", "label": "9-mer\\nBLOSUM62 rows", "width": 180},
            *head_blocks(v1, 180),
            {"kind": "Norm", "label": "L2-normalize", "width": 64, "out": "64 numbers"},
        ],
        "v3  (ESM-2 context + trained head)": [
            {"kind": "Input", "label": "chain\\nsequence", "width": 0},
            {"kind": "PLM", "label": "ESM-2\\n(frozen)", "width": 640},
            {"kind": "Pool", "label": "mean-pool\\n9-mer", "width": 640},
            *head_blocks(v3, 640),
            {"kind": "Norm", "label": "L2-normalize", "width": 64, "out": "64 numbers"},
        ],
    }
    return specs


# ----- neural-netz Typst emitter ----------------------------------------- #
# Every block is a 3-D box (type "convres") whose side scales with sqrt(dimension): a length-D
# vector is drawn as roughly a sqrt(D) x sqrt(D) square, so box size reads off the array size.
FILL = {"Input": "#9aa7b8", "PLM": "#e0884e", "Pool": "#aebfd0", "Dense": "#6fa8dc",
        "Linear": "#3d6fb4", "Norm": "#7bc47f"}
PLAIN = {"Input": "input", "PLM": "ESM-2 (frozen)\\nreads whole chain", "Pool": "average the 9",
         "Dense": "mix", "Linear": "compress", "Norm": "scale to length 1"}


def layer_typst(b, i):
    dim = b.get("width") or 0
    side = round(max(3.0, (dim ** 0.5) * 0.62), 1) if dim else 4.0   # ~ sqrt(D) (a D-vector = √D square)
    base = b.get("label") or PLAIN.get(b["kind"], b["kind"])
    if b["kind"] == "Norm":
        label = f'{base}\\n{b.get("out","")}'
    elif b["kind"] in ("Dense", "Linear", "Pool"):
        label = f'{PLAIN[b["kind"]]}\\n({dim})'
    elif dim:
        label = f'{base}\\n({dim})'
    else:
        label = base
    return ("    (" + ", ".join([
        'type: "convres"', f'name: "L{i}"', f'label: "{label}"',
        f'fill: rgb("{FILL[b["kind"]]}")', "offset: 2.0",
        f"height: {side}", f"depth: {round(side * 0.7, 1)}", "width: 1.6"]) + "),")


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
