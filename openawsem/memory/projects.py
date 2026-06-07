"""Project-level helpers bridging the backends to the legacy ``frags.mem`` /
``single_frags.mem`` artifacts that ``mm_create_project`` and the energy term expect.

These keep the historical ``create_single_memory`` / ``create_fragment_memories`` names
and rough signatures, but build the memory with the new backends (molscene parsing,
local-mmCIF / CIF / PDB fetch) and write the legacy files via
:meth:`MemoryWells.to_frags_mem`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from openawsem.memory.align import parse_fasta
from openawsem.memory.generators import LocalBlast
from openawsem.memory.generators.single import SingleStructure

logger = logging.getLogger(__name__)


def create_single_memory(pdb, chain="-1", *, weight=20, output="single_frags.mem", gro_dir="."):
    """Write a single (template) memory ``single_frags.mem`` + per-chain gros from ``pdb``.

    ``chain="-1"`` uses all chains; a string like ``"AB"`` selects chains A and B.
    """
    name = Path(pdb).stem
    chains = None if str(chain) == "-1" else (list(chain) if isinstance(chain, str) else chain)
    memory = SingleStructure(pdb, chain=chains, weight=weight).generate(gro_dir=gro_dir, name=name)
    memory.to_frags_mem(output)
    return memory


def create_fragment_memories(database, fasta_file, memories_per_position=20, brain_damage=1,
                             fragment_length=9, frag_lib_dir="fraglib", weight=1,
                             evalue_threshold=10000, cif_dir=None, output="frags.mem",
                             download_format="cif", **legacy):
    """Generate a fragment memory by local BLAST and write ``frags.mem`` + a gro library.

    Structures are fetched from ``cif_dir`` (a local wwPDB-divided mmCIF database) when
    given, otherwise downloaded (``download_format`` ``"cif"`` or ``"pdb"``).  Obsolete
    legacy arguments (``pdb_dir``, ``index_dir``, ``failed_pdb_list_file``, ``pdb_seqres``,
    ``cutoff_identical``) are accepted and ignored.

    NOTE: homolog / ``brain_damage`` exclusion is not yet ported, so the generated memory
    may include homologous (and self) structures.
    """
    if brain_damage:
        logger.warning("create_fragment_memories: homolog/brain_damage=%s filtering is not yet "
                       "ported to the new backend and is ignored.", brain_damage)
    target = "".join(parse_fasta(fasta_file).values())
    backend = LocalBlast(database=str(database), n_mem=memories_per_position,
                         fragment_length=fragment_length, evalue=float(evalue_threshold),
                         weight=weight, cif_dir=cif_dir, download_format=download_format)
    memory = backend.generate(target, gro_dir=frag_lib_dir)
    memory.to_frags_mem(output)
    return memory
