"""Align fragment-memory fragments to the target sequence.

For each fragment in a ``frags.mem``, its CA sequence (from the gro) is placed at its
target position, giving a "staggered FASTA" view of which fragments cover which residues.
Each aligned fragment also gets a **score** = sequence identity to the target over its
span (BLAST fragments need not be identical, so this is informative about match quality).

Moved here from ``openawsem.helperFunctions.align_fragments`` so fragment alignment lives
with the rest of the fragment-memory code.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

THREE_TO_ONE = {
    "ALA": "A", "LEU": "L", "GLY": "G", "VAL": "V", "ILE": "I",
    "SER": "S", "THR": "T", "CYS": "C", "PRO": "P", "ASP": "D",
    "GLU": "E", "PHE": "F", "LYS": "K", "ARG": "R", "HIS": "H",
    "ASN": "N", "GLN": "Q", "MET": "M", "TRP": "W", "TYR": "Y",
        # Modified amino acids and non-standard amino acids
    'UNK': 'X',   # Unknown amino acid
    'MSE': 'M',  # Selenomethionine, used frequently in X-ray crystallography
    # 'SEC': 'C',  # (U) Selenocysteine, known as the 21st amino acid
    # 'PYL': 'K',  # (O) Pyrrolysine, the 22nd amino acid
    # 'PTR': 'Y',  # O-phosphotyrosine, important in signaling
    # 'TPO': 'T',  # Phosphothreonine
    # 'SEP': 'S',  # Phosphoserine
    # 'PCA': 'Q',  # Pyroglutamic acid, a cyclized form of glutamine
    # 'HYP': 'P',  # Hydroxyproline, common in collagen
    'M3L': 'K',  # N3-methyllysine, common in histones
    # 'ORN': 'K',  # Ornithine, used in some engineered proteins
    # 'CIT': 'R',  # Citrulline, commonly studied in the urea cycle and immune responses
    # 'HIC': 'R',  # Homoarginine, arises through post-translational modifications
    # 'CYD': 'C',  # Dicysteine
    'CAS': 'C',  # S-dicysteine
    # 'NLE': 'L',  # Norleucine, used in protein engineering and as a methionine surrogate
    # 'NVA': 'V',  # Norvaline, used in metabolic studies and protein structure analysis
    # 'BMT': 'T',  # Beta-methylthreonine, found in some antibiotics
    # 'DHA': 'S',  # Dehydroalanine, formed by post-translational modification
    # '5HP': 'P',  # 5-hydroxyproline, another modification found in collagen
    # 'SOC': 'C',  # Cysteic acid, an oxidation product of cysteine
    # 'OCS': 'C',  # Cysteine sulfonic acid, another oxidation product
    # 'MHO': 'M',  # Methionine sulfoxide, results from oxidative stress
    # 'PLQ': 'W'   # Pyrroloquinoline quinone adduct, found in some enzyme studies
    }


def parse_fasta(filename) -> dict:
    """Parse a FASTA file into an ``{name: sequence}`` dict."""
    fasta, name = {}, None
    with open(filename) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:]
                if name in fasta:
                    raise ValueError(f"Duplicate sequence identifier: {name}")
                fasta[name] = ""
            elif name is None:
                raise ValueError("FASTA data before any header")
            else:
                fasta[name] += line.replace(" ", "")
    return fasta


def _fragment_ca_sequence(gro_file, mem_init, length) -> str:
    gro = pd.read_csv(gro_file, skiprows=2, sep=r"\s+",
                      names=["ResID", "resName", "name", "serial", "x", "y", "z"])
    sel = gro[(gro["name"] == "CA") & (gro["ResID"] >= mem_init) & (gro["ResID"] < mem_init + length)]
    return "".join(sel["resName"].map(THREE_TO_ONE).fillna("X"))


def align_fragments(fasta, frags, working_directory=".") -> pd.DataFrame:
    """Return a per-fragment table: ``name, seq_init, length, fragment, target, score``.

    ``score`` is the fraction of CA residues matching the target sequence over the
    fragment's aligned span (identity in ``[0, 1]``).
    """
    fasta_data = parse_fasta(fasta)
    target = "".join(fasta_data.values())
    parent = Path(working_directory)
    table = pd.read_csv(frags, skiprows=4, sep=r"\s+", comment="#",
                        names=["Gro", "seq_init", "mem_init", "length", "strength"])
    rows = []
    for mem in table.itertuples():
        fragment = _fragment_ca_sequence(parent / mem.Gro, mem.mem_init, mem.length)
        span = target[mem.seq_init - 1: mem.seq_init - 1 + mem.length]
        matches = sum(f == t for f, t in zip(fragment, span))
        score = matches / mem.length if mem.length else 0.0
        rows.append({"name": Path(mem.Gro).stem, "seq_init": mem.seq_init, "length": mem.length,
                     "fragment": fragment, "target": span, "score": score})
    return pd.DataFrame(rows, columns=["name", "seq_init", "length", "fragment", "target", "score"])


def _staggered_target(fasta_data) -> str:
    max_length = max(len(seq) for seq in fasta_data.values())
    out, offset = "", 0
    for name, sequence in fasta_data.items():
        aligned = ("-" * offset) + sequence + ("-" * (max_length - len(sequence) - offset))
        out += f">{name}\n{aligned}\n"
        offset += len(sequence)
    return out


def write_staggered_fasta(fasta, frags, working_directory=".", output=None, copy=None) -> pd.DataFrame:
    """Write the staggered-FASTA alignment view; return the per-fragment score table."""
    fasta_data = parse_fasta(fasta)
    target = "".join(fasta_data.values())
    scores = align_fragments(fasta, frags, working_directory)
    if copy is not None:
        table = pd.read_csv(frags, skiprows=4, sep=r"\s+", comment="#",
                            names=["Gro", "seq_init", "mem_init", "length", "strength"])
        for gro in table["Gro"]:
            shutil.copy(Path(working_directory) / gro, Path(copy))
    out_path = Path("/dev/stdout") if output is None else Path(output)
    with open(out_path, "w") as fh:
        fh.write(_staggered_target(fasta_data))
        for row in scores.itertuples():
            seq = "-" * (row.seq_init - 1) + row.fragment + "-" * (len(target) - row.length - (row.seq_init - 1))
            fh.write(f">{row.name}\n{seq}\n")
    return scores


def main(args=None):
    parser = argparse.ArgumentParser(description="Align fragment-memory fragments to a FASTA.")
    parser.add_argument("--fasta", required=True, help="Path to the FASTA file.")
    parser.add_argument("--frags", default="frags.mem", help="Path to the fragment memory file.")
    parser.add_argument("--working_directory", default=".", help="Directory the gro paths are relative to.")
    parser.add_argument("--output", default=None, help="Output path for the staggered FASTA.")
    parser.add_argument("--copy", default=None, help="Copy the referenced gros to this directory.")
    parser.add_argument("--scores", default=None, help="Optional CSV path for the per-fragment scores.")
    args = parser.parse_args() if args is None else parser.parse_args(args)

    scores = write_staggered_fasta(args.fasta, args.frags, args.working_directory, args.output, args.copy)
    if args.scores:
        scores.to_csv(args.scores, index=False)
    print(f"Aligned {len(scores)} fragments; mean identity to target = {scores['score'].mean():.3f}")


if __name__ == "__main__":
    main()
