"""``single`` backend: one template structure -> a self/template memory."""
from __future__ import annotations

from pathlib import Path

from openawsem.memory.memory import MemoryWells
from openawsem.memory.generators.base import FragmentBackend, Registry


@Registry.register("single")
class SingleStructure(FragmentBackend):
    """One template structure -> a self/template memory.

    Each chain of ``structure`` (a path or molscene ``Scene``) becomes one fragment;
    residues are renumbered continuously from 1 across chains to match the AWSEM target
    numbering.  ``weight`` defaults to 1 (use 20 to reproduce the legacy single memory).
    """

    def __init__(self, structure=None, *, chain=None, weight=1.0,
                 min_seq_sep=3, max_seq_sep=9, well_width=0.1):
        self.structure = structure
        self.chain = chain
        self.weight = weight
        self.min_seq_sep = min_seq_sep
        self.max_seq_sep = max_seq_sep
        self.well_width = well_width

    def generate(self, sequence=None) -> "MemoryWells":
        from openawsem.memory import _molscene
        if self.structure is None:
            raise ValueError("SingleStructure needs a structure (path or molscene Scene)")
        scene = self.structure
        if isinstance(scene, (str, Path)):
            scene = _molscene.load_structure(scene)
        atoms = _molscene.cacb_nm(scene)
        if self.chain is not None:
            chains = [self.chain] if isinstance(self.chain, str) else list(self.chain)
            atoms = atoms[atoms["chain"].isin(chains)]
        fragments = []
        offset = 0
        for _chain, group in atoms.groupby("chain", sort=False):
            resids = sorted(group["resid"].unique())
            remap = {rid: offset + k + 1 for k, rid in enumerate(resids)}
            frag = group.copy()
            frag["target_resid"] = frag["resid"].map(remap)
            frag["weight"] = self.weight
            fragments.append(frag)
            offset += len(resids)
        memory = MemoryWells.from_fragments(fragments, min_seq_sep=self.min_seq_sep,
                                            max_seq_sep=self.max_seq_sep, well_width=self.well_width)
        structure = str(self.structure) if isinstance(self.structure, (str, Path)) else "<Scene>"
        memory.provenance.update({"method": "single", "structure": structure})
        return memory
