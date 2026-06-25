#import "@preview/neural-netz:0.3.0": draw-network

#set page(width: auto, height: auto, margin: 14pt, fill: white)
#set text(font: "DejaVu Sans", size: 9pt)

#text(weight: "bold", size: 11pt)[v0  (BLOSUM-cosine, no training)]
#v(2pt)
#draw-network((
    (type: "convres", name: "L0", label: "9-mer\nBLOSUM62 rows\n(180)", fill: rgb("#9aa7b8"), offset: 2.0, height: 8.3, depth: 5.8, width: 1.6),
    (type: "convres", name: "L1", label: "L2-normalize\n180 numbers", fill: rgb("#7bc47f"), offset: 2.0, height: 8.3, depth: 5.8, width: 1.6),
), palette: "warm", scale: 90%)
#v(16pt)

#text(weight: "bold", size: 11pt)[v1  (BLOSUM-MLP, trained)]
#v(2pt)
#draw-network((
    (type: "convres", name: "L0", label: "9-mer\nBLOSUM62 rows\n(180)", fill: rgb("#9aa7b8"), offset: 2.0, height: 8.3, depth: 5.8, width: 1.6),
    (type: "convres", name: "L1", label: "mix\n(256)", fill: rgb("#6fa8dc"), offset: 2.0, height: 9.9, depth: 6.9, width: 1.6),
    (type: "convres", name: "L2", label: "mix\n(256)", fill: rgb("#6fa8dc"), offset: 2.0, height: 9.9, depth: 6.9, width: 1.6),
    (type: "convres", name: "L3", label: "compress\n(64)", fill: rgb("#3d6fb4"), offset: 2.0, height: 5.0, depth: 3.5, width: 1.6),
    (type: "convres", name: "L4", label: "L2-normalize\n64 numbers", fill: rgb("#7bc47f"), offset: 2.0, height: 5.0, depth: 3.5, width: 1.6),
), palette: "warm", scale: 90%)
#v(16pt)

#text(weight: "bold", size: 11pt)[v3  (ESM-2 context + trained head)]
#v(2pt)
#draw-network((
    (type: "convres", name: "L0", label: "chain\nsequence", fill: rgb("#9aa7b8"), offset: 2.0, height: 4.0, depth: 2.8, width: 1.6),
    (type: "convres", name: "L1", label: "ESM-2\n(frozen)\n(640)", fill: rgb("#e0884e"), offset: 2.0, height: 15.7, depth: 11.0, width: 1.6),
    (type: "convres", name: "L2", label: "average the 9\n(640)", fill: rgb("#aebfd0"), offset: 2.0, height: 15.7, depth: 11.0, width: 1.6),
    (type: "convres", name: "L3", label: "mix\n(256)", fill: rgb("#6fa8dc"), offset: 2.0, height: 9.9, depth: 6.9, width: 1.6),
    (type: "convres", name: "L4", label: "mix\n(256)", fill: rgb("#6fa8dc"), offset: 2.0, height: 9.9, depth: 6.9, width: 1.6),
    (type: "convres", name: "L5", label: "compress\n(64)", fill: rgb("#3d6fb4"), offset: 2.0, height: 5.0, depth: 3.5, width: 1.6),
    (type: "convres", name: "L6", label: "L2-normalize\n64 numbers", fill: rgb("#7bc47f"), offset: 2.0, height: 5.0, depth: 3.5, width: 1.6),
), palette: "warm", scale: 90%)
#v(16pt)
