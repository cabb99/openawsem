"""Topology rules of the helical term (rqm/helical_terms.md)."""
import types

import numpy as np
import pytest

from openawsem.functionTerms.helicalTerms import HelixParameters, HelixTopology
from tests._helical_data import load_case


def _topology(chains, sequence=None):
    """CA/O/CB layout (3 particles per residue) for chains of the given lengths."""
    n = sum(chains)
    sequence = sequence or 'A' * n
    idx = np.arange(n)
    chain_index = np.repeat(np.arange(len(chains)), chains)
    cb = np.where(np.array(list(sequence)) == 'G', -1, 3 * idx + 2)
    return HelixTopology(sequence, chain_index, 3 * idx, 3 * idx + 1, cb, 3 * n)


def test_pairs_stay_inside_chains():  # rq-08e20c7f
    lengths = [1, 2, 4, 5, 6, 22]
    t = _topology(lengths)
    pairs = t.pairs(HelixParameters())
    assert np.all(t.chain_index[pairs.i] == t.chain_index[pairs.j])
    assert np.all(pairs.j - pairs.i == 4)
    starts = np.cumsum([0] + lengths)
    per_chain = {k: [(i, j) for i, j in zip(pairs.i, pairs.j) if t.chain_index[i] == k] for k in range(6)}
    assert per_chain[0] == per_chain[1] == per_chain[2] == []
    assert per_chain[3] == [(starts[3], starts[3] + 4)]
    assert len(per_chain[4]) == 2 and len(per_chain[5]) == 18
    # The stored LAMMPS case built with these chains agrees with AWSEM-MD (see the gold-standard tests).
    assert load_case('short_chains').topology.pairs(HelixParameters()).i.tolist() == pairs.i.tolist()


def test_last_residue_can_be_a_donor():  # rq-05e4cc4b
    pairs = _topology([10]).pairs(HelixParameters())
    assert list(zip(pairs.i.tolist(), pairs.j.tolist())) == [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9)]
    pairs3 = _topology([10]).pairs(HelixParameters(i_diff=3))
    assert pairs3.j.max() == 9 and len(pairs3) == 7


def test_density_neighbours_across_chains_are_kept():  # rq-6f821ff5
    t = _topology([3, 3])
    mask = t.density_mask()
    assert mask[2, 3] and mask[3, 2]          # chain boundary: consecutive indices, different chains
    assert not mask[0, 1] and not mask[4, 5]  # same chain, |r - s| = 1
    assert not mask[1, 1]                     # self
    assert mask[0, 2] and mask[3, 5]          # same chain, |r - s| = 2
    r, s = t.density_pairs()
    assert np.all(r < s) and len(r) == int(np.triu(mask, 1).sum())


def test_glycine_uses_ca_as_density_bead():  # rq-76746c8b
    t = _topology([5], sequence='AGAGP')
    assert t.density_bead.tolist() == [2, 3, 8, 9, 14]
    assert t.cb.tolist() == [2, -1, 8, -1, 14]


@pytest.mark.parametrize('kwargs', [
    dict(cb=[2, -1, 8]),                   # non-glycine without CB
    dict(sequence='AXA'),                  # unknown residue
    dict(ca=[0, -1, 6]),                   # missing CA
    dict(o=[1, 4]),                        # wrong length
    dict(cb=[2, 5, 99]),                   # outside the system
    dict(chain_index=[0, 1, 0]),           # chain split in two
], ids=['no-cb', 'unknown-residue', 'no-ca', 'wrong-length', 'out-of-range', 'split-chain'])
def test_invalid_topologies_are_rejected(kwargs):  # rq-4c990796
    base = dict(sequence='AAA', chain_index=[0, 0, 0], ca=[0, 3, 6], o=[1, 4, 7], cb=[2, 5, 8], n_particles=9)
    base.update(kwargs)
    with pytest.raises(ValueError):
        HelixTopology(**base)


def test_pairs_with_missing_oxygen_are_skipped():  # rq-2f586ec6
    t = _topology([10])
    o = t.o.copy()
    o[2] = -1  # residue 2 is the acceptor of (2, 6) and O_{j-1} of (-1, 3), which does not exist
    o[7] = -1  # residue 7 is O_{j-1} of (4, 8) and the acceptor of (7, 11), which does not exist
    t2 = HelixTopology(t.sequence, t.chain_index, t.ca, o, t.cb, t.n_particles)
    pairs = t2.pairs(HelixParameters())
    assert list(zip(pairs.i.tolist(), pairs.j.tolist())) == [(0, 4), (1, 5), (3, 7), (5, 9)]


def test_topology_from_openmm():  # rq-3fb15d40
    app = pytest.importorskip('openmm.app')
    top = app.Topology()
    expected_ca, expected_cb = [], []
    for chain_residues in (['ALA', 'IGL', 'IPR'], ['LYS', 'GLY']):
        chain = top.addChain()
        for name in chain_residues:
            res = top.addResidue(name, chain)
            for atom in ('N', 'CA', 'C', 'O') + (() if name in ('GLY', 'IGL') else ('CB',)):
                a = top.addAtom(atom, app.element.carbon, res)
                if atom == 'CA':
                    expected_ca.append(a.index)
                if atom == 'CB':
                    expected_cb.append(a.index)
    t = HelixTopology.from_openmm(top)
    assert t.sequence == 'AGPKG'
    assert t.chain_index.tolist() == [0, 0, 0, 1, 1]
    assert t.ca.tolist() == expected_ca
    assert [c for c in t.cb if c >= 0] == expected_cb and t.cb[1] == t.cb[4] == -1
    assert t.n_particles == top.getNumAtoms()
    # OpenAWSEM's generic NGP residues need an explicit sequence.
    top2 = app.Topology()
    chain = top2.addChain()
    for _ in range(2):
        res = top2.addResidue('NGP', chain)
        for atom in ('CA', 'O', 'CB'):
            top2.addAtom(atom, app.element.carbon, res)
    with pytest.raises(ValueError):
        HelixTopology.from_openmm(top2)
    assert HelixTopology.from_openmm(top2, sequence='LK').sequence == 'LK'
    with pytest.raises(ValueError):
        HelixTopology.from_openmm(top2, sequence='LKA')


def test_topology_from_openawsem():  # rq-8c38f81c
    oa = types.SimpleNamespace(nres=5, natoms=20, seq='AGPKG', chain_starts=[0, 3], chain_ends=[2, 4],
                               ca=[0, 4, 8, 12, 16], o=[1, 5, 9, 13, 17], cb=[2, -1, 10, 14, -1], k_awsem=1.0)
    t = HelixTopology.from_openawsem(oa)
    assert t.chain_index.tolist() == [0, 0, 0, 1, 1]
    assert t.cb.tolist() == [2, -1, 10, 14, -1]
    assert t.density_bead.tolist() == [2, 4, 10, 14, 16]
