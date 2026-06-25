// Conceptual, jargon-free overview of how the ml backend chooses fragments.
// Render:  typst compile docs/figures/ml_pipeline.typ docs/figures/ml_pipeline.svg
#set page(width: 960pt, height: auto, margin: 18pt, fill: white)
#set text(font: "DejaVu Sans", size: 10pt)

#let step(num, title, body, col) = rect(
  radius: 7pt, inset: 10pt, width: 150pt, height: 120pt,
  fill: col, stroke: 0.6pt + rgb("#5b6b7a"),
)[
  #text(weight: "bold", size: 10.5pt)[#num. #title]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#22303c"))[#body]
]
#let arrow = align(horizon)[#text(size: 22pt, fill: rgb("#5b6b7a"))[→]]

#let BLUE = rgb("#dbe9f6")
#let GREEN = rgb("#e4f1e0")
#let GREY = rgb("#eceff3")

#align(center)[#text(size: 14pt, weight: "bold")[How the ml backend chooses fragments]]
#v(2pt)
#align(center)[#text(size: 9pt, fill: rgb("#55626e"))[
  Fragment memory guides folding by copying the local shape of similar known fragments.
  The only question is *which* fragments count as "similar".]]
#v(12pt)

#grid(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
  rows: 118pt, align: horizon, column-gutter: 7pt,
  step([1], "A short piece", [Slide a 9-residue window along the target protein — one little
        peptide at a time \ \ #text(font: "DejaVu Sans Mono", size: 8pt)[…K T A Y I L K Q R…]], GREY),
  arrow,
  step([2], "Make an embedding", [An #emph[encoder] turns the piece into a point in space (a short
        list of numbers, the #emph[embedding]). Pieces that should fold alike land close together.], BLUE),
  arrow,
  step([3], "Find look-alikes", [Search ~2 million stored fragments for the closest embeddings —
        the #emph[nearest neighbours].], BLUE),
  arrow,
  step([4], "Borrow geometry", [Read those neighbours' measured atom-to-atom distances — how their
        backbone is actually shaped.], GREEN),
  arrow,
  step([5], "Add soft preferences", [Turn each distance into a soft distance preference (a Gaussian
        well) that nudges the structure toward that local shape. All windows together = the folding
        guide.], GREEN),
)

#v(16pt)
#rect(radius: 7pt, inset: 11pt, fill: rgb("#fbf3e6"), stroke: 0.6pt + rgb("#d9a441"), width: 100%)[
  #text(weight: "bold")[What changes between versions — and why it matters]
  #v(4pt)
  #grid(columns: (1fr, 1fr), column-gutter: 14pt,
    [#text(weight: "bold", fill: rgb("#3d6fb4"))[v0 — describe by sequence.] The embedding is built
      only from the window's amino-acid letters (a BLOSUM score). It finds fragments with a
      #emph[similar sequence]. Simple and fast, but two pieces with similar sequence can fold
      differently.],
    [#text(weight: "bold", fill: rgb("#c0712a"))[v3 — describe with context.] The embedding starts
      from a frozen protein language model (ESM-2) that has read the #emph[whole] chain, then a
      trained head reshapes it so that #emph[similar embedding] means #emph[similar local shape] —
      even when the sequences look nothing alike. This is why v3 folds hard proteins that v0 gets
      wrong.],
  )
]
