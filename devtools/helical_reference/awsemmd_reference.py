"""
AWSEM-MD reference tooling for the helical terms (development only).

* :class:`CoarseGrainedProtein` reads CA, O and CB from a PDB file in AWSEM-MD bead order
  (CA, O, CB per residue; glycine carries an ``HB`` bead at the ideal CB position that no
  helix interaction uses, as in AWSEM-MD data files).
* :func:`run_lammps` writes an AWSEM-MD project with only ``[Helix]`` (plus ``[ABC]`` and
  ``[Epsilon]``) enabled, runs ``run 0`` for each frame and returns the helix energy
  (``f_2[10]``, kcal/mol) and per-atom forces (kcal/mol/A).

The LAMMPS executable must include the AWSEM-MD package (https://github.com/adavtyan/awsemmd);
it is found through ``AWSEMMD_LMP`` or ``lmp_serial``/``lmp`` on ``PATH``.
"""
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from openawsem.functionTerms.helicalTerms import THREE_TO_ONE, HelixParameters, HelixTopology


ONE_TO_THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU',
                'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
                'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}


def ideal_cb(n, ca, c):
    """Ideal CB position from backbone N, CA, C (same formula as the AWSEM/OpenAWSEM tools)."""
    b = ca - n
    c = c - ca
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca


# rq-b8026202
@dataclass
class CoarseGrainedProtein:
    """Coordinates (Angstrom) and topology of a CA/O/CB(HB) protein."""
    positions: np.ndarray        # (n_particles, 3), Angstrom
    atom_names: List[str]        # 'CA', 'O', 'CB' or 'HB'
    topology: HelixTopology
    chain_names: List[str]       # original chain id of each chain index

    @property
    def sequence(self) -> str:
        return self.topology.sequence

    @property
    def n_particles(self) -> int:
        return len(self.positions)

    def chain_sequences(self) -> List[str]:
        """Sequence of each chain, in chain order."""
        chain = self.topology.chain_index
        return [''.join(np.array(list(self.sequence))[chain == k]) for k in np.unique(chain)]

    def residue_of_atom(self) -> np.ndarray:
        """Residue index of every particle."""
        res = np.empty(self.n_particles, dtype=np.int64)
        t = self.topology
        for r in range(t.n_residues):
            res[t.ca[r]] = r
            res[t.o[r]] = r
            res[3 * r + 2] = r
        return res

    def with_positions(self, positions) -> 'CoarseGrainedProtein':
        positions = np.asarray(positions, dtype=float)
        if positions.shape != self.positions.shape:
            raise ValueError(f"positions must have shape {self.positions.shape}")
        return CoarseGrainedProtein(positions, list(self.atom_names), self.topology, list(self.chain_names))

    # Constructors -----------------------------------------------------
    @classmethod
    def from_residues(cls, sequences: Sequence[str], ca, o, cb_or_hb,
                      chain_names: Optional[Sequence[str]] = None) -> 'CoarseGrainedProtein':
        """Build from per-residue CA, O and CB/HB coordinates (Angstrom), chain by chain.

        ``sequences`` is a list with one string per chain; coordinates are given for
        the concatenated residues.
        """
        seq = ''.join(sequences)
        ca, o, cb = (np.asarray(x, dtype=float).reshape(-1, 3) for x in (ca, o, cb_or_hb))
        n = len(seq)
        if not (len(ca) == len(o) == len(cb) == n):
            raise ValueError("ca, o and cb_or_hb need one row per residue")
        positions = np.empty((3 * n, 3))
        positions[0::3], positions[1::3], positions[2::3] = ca, o, cb
        names = []
        for aa in seq:
            names += ['CA', 'O', 'HB' if aa == 'G' else 'CB']
        chain_index = np.concatenate([[k] * len(s) for k, s in enumerate(sequences)]) if n else np.zeros(0, int)
        idx = np.arange(n)
        cb_index = np.where(np.array(list(seq)) == 'G', -1, 3 * idx + 2) if n else np.zeros(0, int)
        topology = HelixTopology(sequence=seq, chain_index=chain_index, ca=3 * idx, o=3 * idx + 1,
                                 cb=cb_index, n_particles=3 * n)
        if chain_names is None:
            chain_names = [chr(ord('A') + k) for k in range(len(sequences))]
        return cls(positions, names, topology, list(chain_names))

    @classmethod
    def from_pdb(cls, path: Union[str, Path], chains: Optional[Sequence[str]] = None) -> 'CoarseGrainedProtein':
        """Read CA, O and CB from an all-atom (or CA/O/CB) PDB file.

        Only the first model and the first alternate location are used. Residues
        without a CA are skipped; non-standard residue names raise ``ValueError``.
        A residue without O raises ``ValueError`` (AWSEM-MD needs every O). Glycine
        gets an HB bead at the ideal CB position (or on CA when N/C are missing).
        """
        residues = {}
        order = []
        for line in Path(path).read_text().splitlines():
            record = line[:6]
            if record.startswith('ENDMDL'):
                break
            if record not in ('ATOM  ', 'HETATM'):
                continue
            altloc = line[16]
            if altloc not in (' ', 'A', '1'):
                continue
            resname = line[17:20].strip()
            chain = line[21]
            if chains is not None and chain not in chains:
                continue
            key = (chain, line[22:27])
            if record == 'HETATM' and resname not in THREE_TO_ONE:
                continue
            if key not in residues:
                residues[key] = {'resname': resname, 'atoms': {}}
                order.append(key)
            name = line[12:16].strip()
            residues[key]['atoms'].setdefault(name, np.array([float(line[30:38]), float(line[38:46]),
                                                              float(line[46:54])]))
        seqs, ca, o, cb, chain_names = [], [], [], [], []
        for key in order:
            res = residues[key]
            atoms = res['atoms']
            if 'CA' not in atoms:
                continue
            if res['resname'] not in THREE_TO_ONE:
                raise ValueError(f"Unknown residue {res['resname']} {key} in {path}")
            aa = THREE_TO_ONE[res['resname']]
            oxygen = atoms.get('O', atoms.get('OT1', atoms.get('OXT')))
            if oxygen is None:
                raise ValueError(f"Residue {key} in {path} has no O atom")
            if aa == 'G' or 'CB' not in atoms:
                if aa != 'G':
                    raise ValueError(f"Residue {key} in {path} has no CB atom")
                beta = ideal_cb(atoms['N'], atoms['CA'], atoms['C']) if 'N' in atoms and 'C' in atoms \
                    else atoms['CA'].copy()
            else:
                beta = atoms['CB']
            if not chain_names or chain_names[-1] != key[0]:
                chain_names.append(key[0])
                seqs.append('')
            seqs[-1] += aa
            ca.append(atoms['CA'])
            o.append(oxygen)
            cb.append(beta)
        return cls.from_residues(seqs, ca, o, cb, chain_names=chain_names)

    # Persistence ------------------------------------------------------
    def to_pdb(self, path: Union[str, Path]):
        """Write the beads as a minimal PDB (CA, O, CB/HB atoms)."""
        lines = []
        res = self.residue_of_atom()
        for a, (xyz, name) in enumerate(zip(self.positions, self.atom_names)):
            r = res[a]
            chain = self.chain_names[self.topology.chain_index[r]]
            element = name[0] if name != 'HB' else 'H'
            lines.append(f"ATOM  {a + 1:5d} {name:<4s} {ONE_TO_THREE[self.sequence[r]]} {chain}{r + 1:4d}    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}")
        lines.append('END')
        Path(path).write_text('\n'.join(lines) + '\n')


ATOM_TYPES = {'CA': 1, 'O': 3, 'CB': 4, 'HB': 5}
MASSES = {1: 27.0, 2: 14.0, 3: 28.0, 4: 60.0, 5: 2.0}


# rq-843a60e6
def find_lammps() -> Optional[str]:
    """Path of an AWSEM-MD LAMMPS executable, or None."""
    exe = os.environ.get('AWSEMMD_LMP')
    if exe:
        return exe if Path(exe).exists() else shutil.which(exe)
    for name in ('lmp_serial', 'lmp'):
        found = shutil.which(name)
        if found:
            return found
    return None


def _data_file(protein: CoarseGrainedProtein, positions) -> str:
    t = protein.topology
    res = protein.residue_of_atom()
    span = np.abs(positions).max() + 100.0
    lines = ['LAMMPS AWSEM helix-only reference', '',
             f'{protein.n_particles} atoms', '0 bonds', '0 angles', '0 dihedrals', '0 impropers', '',
             '5 atom types', '0 bond types', '0 angle types', '0 dihedral types', '0 improper types', '',
             f'{-span:.1f} {span:.1f} xlo xhi', f'{-span:.1f} {span:.1f} ylo yhi', f'{-span:.1f} {span:.1f} zlo zhi',
             '', 'Masses', '']
    lines += [f'{k} {m}' for k, m in MASSES.items()]
    lines += ['', 'Atoms', '']
    for a, (name, xyz) in enumerate(zip(protein.atom_names, positions)):
        r = res[a]
        x, y, z = (repr(float(c)) for c in xyz)
        lines.append(f'{a + 1} {t.chain_index[r] + 1} {r + 1} {ATOM_TYPES[name]} 0.0 {x} {y} {z}')
    return '\n'.join(lines) + '\n'


def _input_script(protein: CoarseGrainedProtein) -> str:
    names = np.array(protein.atom_names)
    ids = np.arange(1, protein.n_particles + 1)
    groups = {
        'alpha_carbons': ids[names == 'CA'],
        'beta_atoms': ids[(names == 'CB') | (names == 'HB')],
        'oxygens': ids[names == 'O'],
    }
    lines = [
        'units real', 'dimension 3', 'boundary s s s',
        'neighbor 5 bin', 'neigh_modify delay 0 every 1 check yes',
        'atom_modify sort 0 0.0', 'special_bonds fene', 'atom_style awsemmd',
        'pair_style vexcluded 2 3.5 3.5', 'read_data data.cg', 'pair_coeff * * 0.0',
    ]
    for name, members in groups.items():
        lines.append(f'group {name} id ' + ' '.join(map(str, members)))
    lines += [
        'fix 2 alpha_carbons backbone beta_atoms oxygens fix_backbone_coeff.data cg.seq',
        'thermo_style custom step pe',
        'dump 1 all custom 1 forces.dump id fx fy fz',
        'dump_modify 1 sort id format float %.17g',
        'run 0',
        'print "HELIX_ENERGY $(f_2[10]:%.17g)"',
    ]
    return '\n'.join(lines) + '\n'


def _read_forces(path: Path, n_atoms: int) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith('ITEM: ATOMS')) + 1
    data = np.array([[float(x) for x in line.split()] for line in lines[start:start + n_atoms]])
    forces = np.zeros((n_atoms, 3))
    forces[data[:, 0].astype(int) - 1] = data[:, 1:4]
    return forces


# rq-e11f916a
def run_lammps(protein: CoarseGrainedProtein, params: HelixParameters,
               frames: Optional[Sequence[np.ndarray]] = None, lmp: Optional[str] = None,
               workdir: Optional[str] = None, keep: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """AWSEM-MD helix energy (kcal/mol) and forces (kcal/mol/Angstrom) for each frame.

    ``frames`` defaults to ``[protein.positions]``. Returns ``(energies[n_frames],
    forces[n_frames, n_particles, 3])``.
    """
    lmp = lmp or find_lammps()
    if lmp is None:
        raise FileNotFoundError("No AWSEM-MD LAMMPS executable; set AWSEMMD_LMP")
    frames = [protein.positions] if frames is None else frames
    base = Path(workdir or tempfile.mkdtemp(prefix='awsem_helix_'))
    base.mkdir(parents=True, exist_ok=True)
    energies, forces = [], []
    try:
        for k, positions in enumerate(frames):
            run_dir = base / f'frame_{k:04d}'
            run_dir.mkdir(exist_ok=True)
            (run_dir / 'data.cg').write_text(_data_file(protein, np.asarray(positions, dtype=float)))
            (run_dir / 'cg.seq').write_text('\n'.join(protein.chain_sequences()) + '\n')
            (run_dir / 'fix_backbone_coeff.data').write_text(params.to_section())
            (run_dir / 'in.helix').write_text(_input_script(protein))
            result = subprocess.run([lmp, '-in', 'in.helix', '-log', 'log.lammps', '-screen', 'none'],
                                    cwd=run_dir, capture_output=True, text=True)
            log = (run_dir / 'log.lammps').read_text() if (run_dir / 'log.lammps').exists() else ''
            if result.returncode != 0 or 'HELIX_ENERGY' not in log:
                raise RuntimeError(f"LAMMPS failed in {run_dir}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
                                   f"\n{log[-2000:]}")
            line = next(line for line in log.splitlines() if line.startswith('HELIX_ENERGY'))
            energies.append(float(line.split()[1]))
            forces.append(_read_forces(run_dir / 'forces.dump', protein.n_particles))
    finally:
        if not keep and workdir is None:
            shutil.rmtree(base, ignore_errors=True)
    return np.array(energies), np.array(forces)
