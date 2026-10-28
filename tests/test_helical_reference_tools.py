"""AWSEM-MD reference tooling of the helical terms: PDB reader, LAMMPS bridge, benchmark (rqm/helical_terms.md)."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from openawsem.functionTerms.helicalTerms import reference_energy_forces
from tests._helical_data import load_case

TOOLS = Path(__file__).resolve().parents[1] / 'devtools' / 'helical_reference'
if not TOOLS.exists():
    pytest.skip('devtools/helical_reference is not part of this installation', allow_module_level=True)
sys.path.insert(0, str(TOOLS))
ref = importlib.import_module('awsemmd_reference')
bench = importlib.import_module('benchmark')


@pytest.mark.skipif(ref.find_lammps() is None, reason='needs an AWSEM-MD LAMMPS executable (set AWSEMMD_LMP)')
def test_lammps_is_run_when_available(tmp_path):  # rq-7d184230
    case = load_case('1r69_1r69_example')
    x = case.positions[0] + np.random.default_rng(5).normal(0, 0.4, case.positions[0].shape)
    t = case.topology
    protein = ref.CoarseGrainedProtein.from_residues([t.sequence], x[t.ca], x[t.o], x[2::3])
    energies, forces = ref.run_lammps(protein, case.params, [x], workdir=str(tmp_path))
    e, f = reference_energy_forces(x, t, case.params)
    assert energies[0] == pytest.approx(e, abs=1e-9)
    assert np.abs(forces[0] - f).max() < 1e-9
    assert (tmp_path / 'frame_0000' / 'in.helix').exists()


def test_missing_lammps_is_reported(monkeypatch, tmp_path):  # rq-21a3480b
    monkeypatch.setenv('AWSEMMD_LMP', str(tmp_path / 'does_not_exist'))
    monkeypatch.setenv('PATH', str(tmp_path))
    assert ref.find_lammps() is None
    case = load_case('glypro_default')
    x = case.positions[0]
    t = case.topology
    protein = ref.CoarseGrainedProtein.from_residues([t.sequence], x[t.ca], x[t.o], x[2::3])
    with pytest.raises(FileNotFoundError):
        ref.run_lammps(protein, case.params)


def test_benchmark_runs_on_a_small_system():  # rq-f5817d31
    protein = bench.make_system(200)
    assert len(protein.sequence) == 200 and len(protein.chain_names) == 2
    methods = [m for m in list(bench.python_methods()) + list(bench.openmm_methods()) if m != 'openmm-compound']
    rows = bench.run_benchmark([20], repeats=1, methods=methods, verbose=False)
    assert rows
    for row in rows:
        assert row['status'] == 'ok', row
        assert row['setup_s'] >= 0 and row['eval_ms'] > 0
        tolerance = np.inf if row['method'] == 'openmm-frozen_density' else 1e-8
        assert row['max_force_err'] < tolerance, row
    assert {'numpy-dense', 'numpy-kdtree', 'reference-loops'} <= {r['method'] for r in rows}


PDB = """\
ATOM      1  N   ALA A   1      -0.677  -1.230  -0.491  1.00  0.00           N
ATOM      2  CA  ALA A   1      -0.001   0.064  -0.491  1.00  0.00           C
ATOM      3  C   ALA A   1       1.499  -0.110  -0.491  1.00  0.00           C
ATOM      4  O   ALA A   1       2.030  -1.227  -0.502  1.00  0.00           O
ATOM      5  CB AALA A   1      -0.509   0.856   0.727  0.50  0.00           C
ATOM      6  CB BALA A   1      -9.509   9.856   9.727  0.50  0.00           C
ATOM      7  N   GLY A   2       2.218   0.993  -0.479  1.00  0.00           N
ATOM      8  CA  GLY A   2       3.660   0.932  -0.467  1.00  0.00           C
ATOM      9  C   GLY A   2       4.158   2.364  -0.455  1.00  0.00           C
ATOM     10  O   GLY A   2       3.371   3.315  -0.455  1.00  0.00           O
HETATM   11  O   HOH A 101      10.000  10.000  10.000  1.00  0.00           O
TER
ATOM     12  N   LYS B   1       5.000   5.000   5.000  1.00  0.00           N
ATOM     13  CA  LYS B   1       6.000   5.000   5.000  1.00  0.00           C
ATOM     14  C   LYS B   1       6.500   6.300   5.000  1.00  0.00           C
ATOM     15  O   LYS B   1       6.000   7.300   5.000  1.00  0.00           O
ATOM     16  CB  LYS B   1       6.500   4.300   6.200  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  N   ALA A   1      99.000  99.000  99.000  1.00  0.00           N
ENDMDL
"""


def test_coarse_grained_structure_from_pdb(tmp_path):  # rq-2261a61a
    path = tmp_path / 'in.pdb'
    path.write_text(PDB)
    p = ref.CoarseGrainedProtein.from_pdb(path)
    assert p.sequence == 'AGK' and p.chain_names == ['A', 'B']
    assert p.atom_names == ['CA', 'O', 'CB', 'CA', 'O', 'HB', 'CA', 'O', 'CB']
    np.testing.assert_allclose(p.positions[2], [-0.509, 0.856, 0.727])  # altloc A only
    hb = ref.ideal_cb(np.array([2.218, 0.993, -0.479]), np.array([3.660, 0.932, -0.467]), np.array([4.158, 2.364, -0.455]))
    np.testing.assert_allclose(p.positions[5], hb)
    assert np.linalg.norm(p.positions[5] - p.positions[3]) == pytest.approx(1.53, abs=0.1)  # a C-C bond length
    assert p.chain_sequences() == ['AG', 'K']
    assert p.topology.density_bead.tolist() == [2, 3, 8]
    # Write the beads and read them back.
    out = tmp_path / 'beads.pdb'
    p.to_pdb(out)
    q = ref.CoarseGrainedProtein.from_pdb(out)
    assert q.sequence == p.sequence and q.chain_names == p.chain_names
    np.testing.assert_allclose(q.positions[[0, 1, 2, 3, 4, 6, 7, 8]], p.positions[[0, 1, 2, 3, 4, 6, 7, 8]],
                               atol=1e-3)
    np.testing.assert_allclose(q.positions[5], q.positions[3], atol=1e-3)  # bead-only glycine: HB falls back to CA
    # chains= selects chains.
    assert ref.CoarseGrainedProtein.from_pdb(path, chains=['B']).sequence == 'K'


@pytest.mark.parametrize('edit', [
    lambda s: s.replace('ATOM      4  O   ALA A   1', 'ATOM      4  OG  ALA A   1'),
    lambda s: s.replace(' CB AALA', ' CG AALA').replace(' CB BALA', ' CG BALA'),
    lambda s: s.replace('LYS B', 'XYZ B'),
], ids=['no-O', 'no-CB', 'unknown-residue'])
def test_malformed_pdb_residues_are_rejected(tmp_path, edit):  # rq-e3b558af
    path = tmp_path / 'bad.pdb'
    path.write_text(edit(PDB))
    with pytest.raises(ValueError):
        ref.CoarseGrainedProtein.from_pdb(path)
