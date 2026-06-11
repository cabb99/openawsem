"""Single-sequence (BLAST-like) fragment retrieval over a residue-centric distance table.

Vendored from the ``fragdb`` project (numpy + pandas only).  :class:`FragmentDatabase` loads a
``residues.csv[.gz]`` produced by ``fragdb build`` and supports:

* :meth:`FragmentDatabase.search` -- BLOSUM62 k-mer search for sequence-similar fragments
* :meth:`FragmentDatabase.get_fragment` -- the stored pairwise distances of a fragment

Used by the ``local_db`` fragment-memory generator to build restraints from real fragment
distances without PSI-BLAST or structure fetch.
"""
from __future__ import annotations

from openawsem.memory.retrieval.database import FragmentDatabase, get_fragment, search
from openawsem.memory.retrieval.index import FragmentIndex

__all__ = ["FragmentDatabase", "FragmentIndex", "search", "get_fragment"]
