"""``awsem generate-memory`` -- generate a fragment memory from a sequence/structure.

Examples
--------
    awsem generate-memory --method single --structure crystal_structure.pdb --weight 20
    awsem generate-memory --method local_blast --sequence target.fasta --database <db> --n-mem 20
    awsem generate-memory --list-methods
"""
from __future__ import annotations

import argparse
from pathlib import Path


def read_sequence(spec):
    """Return a sequence string from a raw sequence or the first record of a FASTA file."""
    if spec is None:
        return None
    path = Path(spec)
    if path.exists():
        chunks = []
        with open(path) as fh:
            for line in fh:
                if line.startswith(">"):
                    if chunks:
                        break
                    continue
                chunks.append(line.strip())
        return "".join(chunks)
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="awsem generate-memory",
        description="Generate a fragment memory (saved as JSON).")
    parser.add_argument("--method", default="single", help="generation method (single, local_blast, ...)")
    parser.add_argument("--sequence", help="target sequence string or FASTA path")
    parser.add_argument("--structure", help="template structure (PDB/CIF) for --method single")
    parser.add_argument("--database", help="BLAST database prefix for --method local_blast")
    parser.add_argument("--n-mem", type=int, default=20, dest="n_mem")
    parser.add_argument("--frag-length", type=int, default=9, dest="frag_length")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--well-width", type=float, default=0.1, dest="well_width")
    parser.add_argument("--min-seq-sep", type=int, default=3, dest="min_seq_sep")
    parser.add_argument("--max-seq-sep", type=int, default=9, dest="max_seq_sep")
    parser.add_argument("--out", default="fragment_memory.json")
    parser.add_argument("--list-methods", action="store_true", help="list usable methods and exit")
    args = parser.parse_args(argv)

    from openawsem.memory import available_methods, generate

    if args.list_methods:
        print("Available methods:", ", ".join(available_methods()) or "none")
        return None

    sequence = read_sequence(args.sequence)
    opts = dict(min_seq_sep=args.min_seq_sep, max_seq_sep=args.max_seq_sep, well_width=args.well_width)
    if args.method == "single":
        opts.update(structure=args.structure, weight=args.weight)
    elif args.method == "local_blast":
        opts.update(database=args.database, n_mem=args.n_mem,
                    fragment_length=args.frag_length, weight=args.weight)

    memory = generate(sequence, method=args.method, **opts)
    memory.save(args.out)
    print(f"Wrote {args.method} memory with {len(memory)} restraints to {args.out}")
    return memory


if __name__ == "__main__":
    main()
