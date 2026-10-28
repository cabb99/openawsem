"""Shared loaders for the helical-term tests (LAMMPS/AWSEM-MD reference data)."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from openawsem.functionTerms.helicalTerms import HelixParameters, HelixTopology

DATA = Path(__file__).parent / 'data'
REFERENCE_DIR = DATA / 'helical_reference'
CASES = sorted(p.stem for p in REFERENCE_DIR.glob('*.npz'))
# Helical proteins and parameter variants, and edge cases/controls (used for the slower OpenMM checks).
HELICAL = ['1r69_default', '1cta_default', 'fis_default', '1mbn_default', '2xov_default', '1r69_1r69_example',
           '1cta_1r69_example', '1mbn_1r69_example', '1mbn_pro_acceptor', '1r69_i_diff3', 'fis_modified']
EDGE = ['glypro_default', 'glypro_pro_acceptor', 'short_chains', 'single_residue_chains', 'cutoff_edge',
        'cutoff_short', 'compressed', 'extended', 'h4_default', '1ubq_1r69_example', '1srl_1r69_example']


@dataclass
class CGProtein:
    """Coarse-grained protein in AWSEM-MD bead order: CA, O, CB (HB for glycine) per residue."""
    positions: np.ndarray    # (3 n_res, 3), Angstrom
    topology: HelixTopology

    @property
    def n_particles(self):
        return len(self.positions)

    @property
    def sequence(self):
        return self.topology.sequence


def cg_protein(sequences, ca, o, cb_or_hb) -> CGProtein:
    """One sequence per chain; per-residue CA, O and CB/HB coordinates of all chains."""
    seq = ''.join(sequences)
    n = len(seq)
    positions = np.empty((3 * n, 3))
    positions[0::3], positions[1::3], positions[2::3] = (np.asarray(a, dtype=float).reshape(-1, 3)
                                                         for a in (ca, o, cb_or_hb))
    idx = np.arange(n)
    chain_index = np.repeat(np.arange(len(sequences)), [len(s) for s in sequences])
    cb = np.where(np.array(list(seq)) == 'G', -1, 3 * idx + 2) if n else np.zeros(0, int)
    topology = HelixTopology(sequence=seq, chain_index=chain_index, ca=3 * idx, o=3 * idx + 1, cb=cb,
                             n_particles=3 * n)
    return CGProtein(positions, topology)


@dataclass
class ReferenceCase:
    name: str
    protein: CGProtein
    params: HelixParameters
    positions: np.ndarray   # (n_frames, n_particles, 3), Angstrom
    energies: np.ndarray    # (n_frames,), kcal/mol
    forces: np.ndarray      # (n_frames, n_particles, 3), kcal/mol/Angstrom
    labels: list

    @property
    def topology(self):
        return self.protein.topology


@lru_cache(maxsize=None)
def load_case(name: str) -> ReferenceCase:
    d = np.load(REFERENCE_DIR / f'{name}.npz')
    x = d['positions'][0]
    protein = cg_protein([str(s) for s in d['sequences']], x[0::3], x[1::3], x[2::3])
    params = HelixParameters(**json.loads(str(d['params'])))
    return ReferenceCase(name, protein, params, d['positions'], d['energies'], d['forces'],
                         [str(s) for s in d['labels']])
