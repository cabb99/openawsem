#!/usr/bin/env python
"""
Regenerate the LAMMPS/AWSEM-MD gold-standard data used by the helical-term tests.

For every case in ``CASES`` this builds a coarse-grained (CA, O, CB/HB) structure,
a set of frames (native, noisy, expanded, compressed, ...) and a parameter set,
runs AWSEM-MD's ``fix backbone`` with only the ``[Helix]`` term enabled, and saves
positions, energies (kcal/mol) and forces (kcal/mol/Angstrom) to
``tests/data/helical_reference/<case>.npz``.

Usage::

    AWSEMMD_LMP=/path/to/lmp_serial python devtools/helical_reference/make_reference.py \
        --awsemmd /path/to/awsemmd

``--awsemmd`` is a clone of https://github.com/adavtyan/awsemmd; the PDB files below are
read from it and from this repository (the commits used are recorded in MANIFEST.json).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from awsemmd_reference import CoarseGrainedProtein, find_lammps, run_lammps
from openawsem.functionTerms.helicalTerms import HelixParameters

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'tests' / 'data' / 'helical_reference'

# (source repo, path inside it, description)
PDBS = {
    'h4': ('awsemmd', 'tests/debugging/h4/h4.pdb', 'Histone H4 N-terminal tail from the AWSEM-MD debugging set, disordered control (26 res)'),
    '1r69': ('openawsem', 'tests/data/1r69.pdb', '434 repressor N-terminal domain, 5 helices (63 res)'),
    '1cta': ('awsemmd', 'examples/examples_new/1CTA_Dimer_Binding/1ctaDimer.pdb',
             'Troponin C site III helix-loop-helix homodimer (2 x 34 res)'),
    'fis': ('awsemmd', 'examples/examples_new/fisDNA_example/buildProtein/fis.pdb', 'FIS all-helical homodimer'),
    '1mbn': ('openawsem', 'tests/data/1mbn-crystal_structure.pdb', 'Sperm whale myoglobin (153 res)'),
    '2xov': ('openawsem', 'examples/2xov/2xov.pdb', 'GlpG rhomboid, membrane helical bundle'),
    '1ubq': ('openawsem', 'tests/data/1ubq-crystal_structure.pdb', 'Ubiquitin, alpha/beta control'),
    '1srl': ('awsemmd', 'examples/examples_for_9Oct12_version/1SRL_Go-Model/1SRL.pdb', 'Src SH3, all-beta control'),
}

PARAMS = {
    'default': HelixParameters(),
    '1r69_example': HelixParameters.awsemmd_1r69_example(),
    'pro_acceptor': HelixParameters(pro_acceptor_flag=True, h4prob_pro_acceptor=0.4416, cutoff=15.0),
    'i_diff3': HelixParameters(i_diff=3, cutoff=15.0),
    'modified': HelixParameters(k_helix=0.8, epsilon=2.0, gamma_p=1.5, gamma_w=-0.5, kappa=5.0, kappa_sigma=4.0,
                                threshold=2.0, cutoff=15.0, r_min=4.0, r_max=7.0, sigma_HO=0.9, sigma_NO=0.6,
                                r0_HO=2.1, r0_NO=3.0),
    'short_cutoff': HelixParameters(cutoff=3.2),
}


def standard_frames(x, rng):
    """Native, noisy (0.25/0.5/1.0 A), expanded (x1.25) and compressed (x0.85) frames."""
    centre = x.mean(axis=0)
    frames = [x]
    frames += [x + rng.normal(0, s, x.shape) for s in (0.25, 0.5, 1.0)]
    frames += [centre + 1.25 * (x - centre), centre + 0.85 * (x - centre)]
    return frames, ['native', 'noise0.25', 'noise0.5', 'noise1.0', 'scale1.25', 'scale0.85']


def mutate(protein, sequences):
    """Same coordinates, new sequence (HB/CB labels follow the new glycines)."""
    t = protein.topology
    ca, o, b = protein.positions[t.ca], protein.positions[t.o], protein.positions[2::3]
    return CoarseGrainedProtein.from_residues(sequences, ca, o, b)


def split_chains(protein, lengths):
    """Same coordinates, residues regrouped into chains of the given lengths."""
    seq = protein.sequence
    assert sum(lengths) == len(seq)
    t = protein.topology
    ca, o, b = protein.positions[t.ca], protein.positions[t.o], protein.positions[2::3]
    starts = np.cumsum([0] + list(lengths))
    return CoarseGrainedProtein.from_residues([seq[a:z] for a, z in zip(starts[:-1], starts[1:])], ca, o, b)


def sub_protein(protein, start, stop):
    t = protein.topology
    sl = slice(start, stop)
    return CoarseGrainedProtein.from_residues([protein.sequence[sl]], protein.positions[t.ca][sl],
                                              protein.positions[t.o][sl], protein.positions[2::3][sl])


def build_cases(sources):
    rng = np.random.default_rng(20261027)
    load = lambda key: CoarseGrainedProtein.from_pdb(sources[key])
    cases = {}

    def add(name, protein, params_key, frames=None, labels=None, description=''):
        if frames is None:
            frames, labels = standard_frames(protein.positions, rng)
        cases[name] = dict(protein=protein, params=params_key, frames=frames, labels=labels, description=description)

    # Natural proteins, gold-standard parameter sets.
    for key in ('h4', '1r69', '1cta', 'fis', '1mbn', '2xov'):
        add(f'{key}_default', load(key), 'default', description=PDBS[key][2])
    for key in ('1r69', '1cta', '1mbn'):
        add(f'{key}_1r69_example', load(key), '1r69_example', description=PDBS[key][2])
    for key in ('1ubq', '1srl'):
        add(f'{key}_1r69_example', load(key), '1r69_example', description=PDBS[key][2])
    add('1mbn_pro_acceptor', load('1mbn'), 'pro_acceptor', description='Myoglobin with pro_acceptor_flag on')
    add('1r69_i_diff3', load('1r69'), 'i_diff3', description='1R69 with i_diff = 3')
    add('fis_modified', load('fis'), 'modified', description='FIS dimer with every parameter changed')

    # Edge cases built from the first myoglobin helices.
    mb = load('1mbn')
    helix = sub_protein(mb, 2, 22)
    seq = 'GPAGLPKGPAEGPLAGKPGG'  # G/P at acceptor, donor and density-neighbour positions
    add('glypro_default', mutate(helix, [seq]), 'default', description='Helix with Gly/Pro everywhere')
    add('glypro_pro_acceptor', mutate(helix, [seq]), 'pro_acceptor',
        description='Same, proline acceptor propensity switched on')
    first40 = sub_protein(mb, 0, 40)
    add('short_chains', split_chains(first40, [1, 2, 4, 5, 6, 22]), '1r69_example',
        description='Chains of 1, 2, 4, 5, 6 and 22 residues packed as in myoglobin')
    add('single_residue_chains', split_chains(sub_protein(mb, 0, 12), [1] * 12), '1r69_example',
        description='Twelve one-residue chains (no pairs; density across chains only)')
    centre = helix.positions.mean(axis=0)
    scales = [1.9, 1.95, 2.0, 2.05, 2.1]
    add('cutoff_edge', helix, 'default',
        frames=[centre + s * (helix.positions - centre) for s in scales], labels=[f'scale{s}' for s in scales],
        description='Stretched helix: O-N distances straddle the 6 A cutoff')
    add('cutoff_short', helix, 'short_cutoff',
        description='Native/noisy helix with a 3.2 A O-N cutoff: many pairs straddle it with sizeable weight')
    scales = [0.3, 0.4, 0.5, 0.6]
    add('compressed', first40, '1r69_example',
        frames=[first40.positions.mean(0) + s * (first40.positions - first40.positions.mean(0)) for s in scales],
        labels=[f'scale{s}' for s in scales],
        description='Strongly compressed: bead distances below the well range (rmin_theta)')
    extended = first40.positions.copy()
    extended[:, 0] += 3.8 * np.repeat(np.arange(40), 3)
    add('extended', first40, '1r69_example', frames=[extended], labels=['extended'],
        description='Residues pulled apart along x: no helical pairs, no density')
    return cases


def openawsem_trajectory_case(openawsem_root: Path, out: Path, lmp: str):
    """LAMMPS reference for frames of OpenAWSEM's own 1MBN test trajectory.

    Reads OpenAWSEM's coarse-grained topology (CA, C, O, CB, N, H atoms) and its
    simulation movie as data (mdtraj), runs AWSEM-MD on the CA/O/CB beads of each
    frame and stores energies and forces mapped back onto the 900-atom OpenAWSEM
    system, so that the OpenMM forces can be checked on a real OpenAWSEM topology.
    """
    import mdtraj
    from openmm.app import PDBFile

    from openawsem.functionTerms.helicalTerms import HelixTopology
    data = openawsem_root / 'tests' / 'data'
    pdb = data / '1mbn-openmmawsem.pdb'
    seq = ''.join(line.strip() for line in (data / '1mbn-crystal_structure.fasta').read_text().splitlines()
                  if not line.startswith('>'))
    topology = HelixTopology.from_openmm(PDBFile(str(pdb)).topology, sequence=seq)
    frames = mdtraj.load(str(data / '1mbn-movie.dcd'), top=str(pdb)).xyz.astype(float) * 10.0
    params = PARAMS['1r69_example']
    t = topology
    beads = np.where(t.cb >= 0, t.cb, t.ca)
    cg_frames = [np.stack([x[t.ca], x[t.o], x[beads]], axis=1).reshape(-1, 3) for x in frames]
    protein = CoarseGrainedProtein.from_residues([seq], cg_frames[0][0::3], cg_frames[0][1::3], cg_frames[0][2::3])
    energies, cg_forces = run_lammps(protein, params, cg_frames, lmp=lmp)
    forces = np.zeros_like(frames)
    for k in range(len(frames)):
        forces[k, t.ca] += cg_forces[k][0::3]
        forces[k, t.o] += cg_forces[k][1::3]
        forces[k, beads] += np.where((t.cb >= 0)[:, None], cg_forces[k][2::3], 0.0)
    target = out / 'openawsem'
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / '1mbn_movie.npz', sequence=seq, positions=frames, energies=energies, forces=forces,
                        params=json.dumps(params.as_dict()))
    print(f'openawsem/1mbn_movie        {len(seq):4d} res  E = {np.round(energies, 4)}')
    return dict(description="OpenAWSEM 1MBN test topology and trajectory (tests/data/1mbn-openmmawsem.pdb, "
                            "1mbn-movie.dcd)", params='1r69_example', frames=len(frames),
                energies=[round(float(e), 6) for e in energies])


def git_head(path):
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=path, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ''


# rq-9b29a1d1 rq-7c04f64f
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--awsemmd', required=True, type=Path)
    parser.add_argument('--lmp', default=None)
    parser.add_argument('--out', default=OUT, type=Path)
    args = parser.parse_args()
    lmp = args.lmp or find_lammps()
    roots = {'awsemmd': args.awsemmd, 'openawsem': REPO}
    sources = {key: roots[repo] / path for key, (repo, path, _) in PDBS.items()}
    commits = {name: git_head(root) for name, root in roots.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, case in build_cases(sources).items():
        protein, params = case['protein'], PARAMS[case['params']]
        frames = np.array(case['frames'], dtype=float)
        energies, forces = run_lammps(protein, params, frames, lmp=lmp)
        np.savez_compressed(
            args.out / f'{name}.npz',
            sequences=np.array(protein.chain_sequences()), chain_names=np.array(protein.chain_names),
            atom_names=np.array(protein.atom_names), positions=frames, energies=energies, forces=forces,
            labels=np.array(case['labels']), params=json.dumps(params.as_dict()),
        )
        manifest[name] = dict(description=case['description'], params=case['params'], n_residues=len(protein.sequence),
                              n_chains=len(protein.chain_names), frames=list(case['labels']),
                              energies=[round(float(e), 6) for e in energies])
        print(f'{name:28s} {len(protein.sequence):4d} res  E = {np.round(energies, 4)}')
    manifest['openawsem/1mbn_movie'] = openawsem_trajectory_case(REPO, args.out, lmp)
    meta = dict(lammps=subprocess.run([lmp, '-h'], capture_output=True, text=True).stdout.splitlines()[1:3],
                commits=commits, pdb_sources={k: f'{r}:{p}' for k, (r, p, _) in PDBS.items()}, cases=manifest)
    (args.out / 'MANIFEST.json').write_text(json.dumps(meta, indent=1) + '\n')


if __name__ == '__main__':
    main()
