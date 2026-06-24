#import "@preview/neural-netz:0.3.0": draw-network

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#set text(font: "DejaVu Sans", size: 9pt)

#text(weight: "bold", size: 11pt)[v0  (BLOSUM-cosine, no training)]
#v(2pt)
#draw-network((
    (type: "custom", name: "L0", label: "9-mer\nBLOSUM62 rows\n(180)", fill: rgb("#9aa7b8"), offset: 2.0, height: 6, depth: 0),
    (type: "custom", name: "L1", label: "L2-normalize\nz in R^180", fill: rgb("#7bc47f"), offset: 2.0, height: 6, depth: 0),
), palette: "warm", scale: 90%)
#v(16pt)

#text(weight: "bold", size: 11pt)[v1  (BLOSUM-MLP, trained)]
#v(2pt)
#draw-network((
    (type: "custom", name: "L0", label: "9-mer\nBLOSUM62 rows\n(180)", fill: rgb("#9aa7b8"), offset: 2.0, height: 6, depth: 0),
    (type: "fc", name: "L1", label: "Dense\n(256)", fill: rgb("#6fa8dc"), offset: 2.0, height: 6, depth: 0),
    (type: "fc", name: "L2", label: "Dense\n(256)", fill: rgb("#6fa8dc"), offset: 2.0, height: 6, depth: 0),
    (type: "fc", name: "L3", label: "Linear\n(64)", fill: rgb("#3d6fb4"), offset: 2.0, height: 6, depth: 0),
    (type: "custom", name: "L4", label: "L2-normalize\nz in R^64", fill: rgb("#7bc47f"), offset: 2.0, height: 6, depth: 0),
), palette: "warm", scale: 90%)
#v(16pt)

#text(weight: "bold", size: 11pt)[v3  (ESM-2 context + trained head)]
#v(2pt)
#draw-network((
    (type: "custom", name: "L0", label: "chain\nsequence", fill: rgb("#9aa7b8"), offset: 2.0, height: 6, depth: 0),
    (type: "convres", name: "L1", label: "ESM-2\n(frozen)\n(640)", fill: rgb("#e0884e"), offset: 2.0, height: 11, depth: 4),
    (type: "fc", name: "L2", label: "mean-pool\n9-mer\n(640)", fill: rgb("#c0c8d4"), offset: 2.0, height: 6, depth: 0),
    (type: "fc", name: "L3", label: "Dense\n(256)", fill: rgb("#6fa8dc"), offset: 2.0, height: 6, depth: 0),
    (type: "fc", name: "L4", label: "Dense\n(256)", fill: rgb("#6fa8dc"), offset: 2.0, height: 6, depth: 0),
    (type: "fc", name: "L5", label: "Linear\n(64)", fill: rgb("#3d6fb4"), offset: 2.0, height: 6, depth: 0),
    (type: "custom", name: "L6", label: "L2-normalize\nz in R^64", fill: rgb("#7bc47f"), offset: 2.0, height: 6, depth: 0),
), palette: "warm", scale: 90%)
#v(16pt)
