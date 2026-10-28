"""
Density-dependent alpha-helical hydrogen-bond term of AWSEM, as in AWSEM-MD.

The gold standard is the active ``compute_pair`` path of ``FixBackbone`` in AWSEM-MD
(``src/fix_backbone.cpp`` of https://github.com/adavtyan/awsemmd; ``pair_flag`` is
``#define``-d to 1 there). For every pair ``(i, j = i + i_diff)`` in a chain::

    E = sum_ij -k_helix * (h4prob'(a_i) + h4prob(a_j))
              * exp(-(R_NO - r0_NO)^2 / (2 s_NO^2) - (R_HO - r0_HO)^2 / (2 s_HO^2))
              * (gamma_p + (gamma_w - gamma_p) * H(rho_i) * H(rho_j))

with ``N_j = an CA_{j-1} + bn CA_j + cn O_{j-1}`` (H_j likewise), ``R_NO < cutoff``,
``H(rho) = (1 - tanh(kappa_sigma (rho - threshold))) / 2`` and the local density
``rho_r = sum_s theta(|X_r - X_s|)`` over the CB (CA for glycine) beads of other chains
or of the same chain with ``|r - s| > 1``.

The density makes the term hard to write with OpenMM's custom forces: every pair
energy depends on all beads near residues i and j. ``CustomGBForce`` allows only one
ParticlePair computed value and two-body energy terms, ``CustomHbondForce`` at most three
particles per donor/acceptor, and ``CustomCVForce`` at most 32 collective variables.
This module therefore provides several exact implementations:

``helical_term_cvforce`` (default of :func:`density_dependent_helical_term`)
    Plain OpenMM forces: the gamma_p part as a CustomCompoundBondForce and the density
    part as blocks of pairs, each a CustomCVForce whose collective variables are the
    densities (CustomBondForce) and hydrogen-bond factors it needs; the blocks are
    nested into one CustomCVForce. Serializable; setup and cost grow as N^2.
``helical_term_pythonforce``
    ``openmm.PythonForce`` (OpenMM >= 8.3) over the NumPy, numba or PyTorch backend.
``helical_term_torchforce``
    ``openmmtorch.TorchForce`` over the TorchScript module (GPU capable).
``helical_term_compound``
    One CustomCompoundBondForce whose bonds carry every density bead; exact but its
    Context creation grows faster than N^2, so it is limited to small systems.
``helical_term_frozen_density``
    Not exact: the environment weight of each pair is frozen at a given structure (or
    at a constant, gamma_p by default, like :func:`hydrogenBondTerms.helical_term`).

The pure-Python backends (:func:`reference_energy_forces`, :class:`NumpyHelix`,
:class:`NumbaHelix`, :func:`make_helix_module`) work on coordinates directly in AWSEM-MD
units (Angstrom, kcal/mol). Validation against LAMMPS/AWSEM-MD and benchmarks are in
``tests/test_helical_terms*.py`` and ``devtools/helical_reference``; requirements in
``rqm/helical_terms.md``.
"""
import dataclasses
import math
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import openmm

AA = 'ARNDCQEGHILKMFPSTWYV'
KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1
MAX_CVS = 32
DELTA_HELIX_XI = 1e-6  # AWSEM-MD drops density forces with |dE/drho_r + dE/drho_s| below this

# h4prob of AWSEM-MD's parameters/fix_backbone_coeff.data (proline = -3.0).
AWSEMMD_H4PROB = (0.77, 0.68, 0.07, 0.15, 0.23, 0.33, 0.27, 0.0, 0.06, 0.23,
                  0.62, 0.65, 0.50, 0.41, -3.0, 0.35, 0.11, 0.45, 0.17, 0.14)

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'CYX': 'C', 'MSE': 'M',
    'IGL': 'G', 'IPR': 'P',  # OpenAWSEM names; NGP (any other residue) needs the sequence
}


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# rq-68a3ed85 rq-825cdfb5
@dataclasses.dataclass(frozen=True)
class HelixParameters:
    """Parameters of the helical term in AWSEM-MD units (Angstrom, kcal/mol).

    The defaults are those of AWSEM-MD ``parameters/fix_backbone_coeff.data``. The
    ``[ABC]`` coefficients build N_j and H_j from CA_{j-1}, CA_j and O_{j-1}.
    """
    k_helix: float = 1.5
    epsilon: float = 1.0
    gamma_p: float = 2.0
    gamma_w: float = -1.0
    kappa: float = 7.0
    kappa_sigma: float = 7.0
    threshold: float = 3.0
    i_diff: int = 4
    cutoff: float = 6.0
    r_min: float = 4.5
    r_max: float = 6.5
    h4prob: Tuple[float, ...] = AWSEMMD_H4PROB
    pro_acceptor_flag: bool = False
    h4prob_pro_acceptor: float = -3.0
    sigma_HO: float = 0.76
    sigma_NO: float = 0.68
    r0_HO: float = 2.06
    r0_NO: float = 2.98
    an: float = 0.4831806
    bn: float = 0.703282
    cn: float = -0.1864262
    ah: float = 0.841
    bh: float = 0.89296
    ch: float = -0.73389
    theta_cutoff: bool = True  # zero the density well outside r_min/r_max -/+ 8 ln(10)/kappa, like AWSEM-MD

    def __post_init__(self):
        h4prob = tuple(float(v) for v in self.h4prob)
        if len(h4prob) != 20:
            raise ValueError(f"h4prob needs 20 values ({AA} order), got {len(h4prob)}")
        object.__setattr__(self, 'h4prob', h4prob)
        if int(self.i_diff) != self.i_diff or self.i_diff < 1:
            raise ValueError("i_diff must be an integer >= 1")
        object.__setattr__(self, 'i_diff', int(self.i_diff))
        object.__setattr__(self, 'pro_acceptor_flag', bool(self.pro_acceptor_flag))
        for name in ('sigma_HO', 'sigma_NO', 'kappa', 'kappa_sigma', 'cutoff'):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0")

    @property
    def k_eff(self) -> float:
        """``k_helix * epsilon``, the prefactor AWSEM-MD uses."""
        return self.k_helix * self.epsilon

    @property
    def theta_range(self) -> Tuple[float, float]:
        """(rmin_theta, rmax_theta): the density well is zero outside (AWSEM-MD's cWell)."""
        if not self.theta_cutoff:
            return 0.0, math.inf
        pad = 8.0 * 2.302585 / self.kappa
        return self.r_min - pad, self.r_max + pad

    def replace(self, **changes) -> 'HelixParameters':
        """A copy with some fields changed (validated)."""
        return dataclasses.replace(self, **changes)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d['h4prob'] = list(d['h4prob'])
        return d

    def propensity(self, aa: str, acceptor: bool = False) -> float:
        """Helix propensity of a residue at the acceptor (i) or donor (j) position."""
        if aa not in AA:
            raise ValueError(f"Unknown residue type {aa!r}; expected one of {AA}")
        if acceptor and aa == 'P' and self.pro_acceptor_flag:
            return self.h4prob_pro_acceptor
        return self.h4prob[AA.index(aa)]

    def to_section(self) -> str:
        """The ``[ABC]``, ``[Epsilon]`` and ``[Helix]`` sections in AWSEM-MD format."""
        return '\n'.join([
            '[ABC]', f'{self.an!r} {self.bn!r} {self.cn!r}', '0.4436538 0.2352006 0.3211455',
            f'{self.ah!r} {self.bh!r} {self.ch!r}', '',
            '[Epsilon]', f'{self.epsilon!r}', '',
            '[Helix]', f'{self.k_helix!r}', f'{self.gamma_p!r} {self.gamma_w!r}',
            f'{self.kappa!r} {self.kappa_sigma!r}', f'{self.threshold!r}', f'{self.i_diff}',
            f'{self.cutoff!r}', f'{self.r_min!r} {self.r_max!r}', ' '.join(repr(x) for x in self.h4prob),
            f'{int(self.pro_acceptor_flag)} {self.h4prob_pro_acceptor!r}',
            f'{self.sigma_HO!r} {self.sigma_NO!r}', f'{self.r0_HO!r} {self.r0_NO!r}', '',
        ])

    @classmethod
    def awsemmd_default(cls) -> 'HelixParameters':
        """AWSEM-MD ``parameters/fix_backbone_coeff.data`` (the defaults)."""
        return cls()

    @classmethod
    def awsemmd_1r69_example(cls) -> 'HelixParameters':
        """AWSEM-MD ``examples/examples_new/1R69_single_memory`` (k=0.5, cutoff=15)."""
        return cls(k_helix=0.5, cutoff=15.0, r0_HO=2.1558, r0_NO=2.9862)

    @classmethod
    def from_fix_backbone_coeff(cls, path: Union[str, Path], **overrides) -> 'HelixParameters':
        """Read ``[Helix]`` and, if present, ``[ABC]`` and ``[Epsilon]`` from an AWSEM-MD file.

        A section whose header ends in ``-`` (``[Helix]-``) is disabled and ignored; a
        missing, disabled or truncated ``[Helix]`` raises ``ValueError``.
        """
        sections = _split_sections(Path(path).read_text())
        if '[Helix]' not in sections:
            raise ValueError(f"No enabled [Helix] section in {path}")
        it = iter(sections['[Helix]'])
        try:
            values = dict(
                k_helix=float(next(it)), gamma_p=float(next(it)), gamma_w=float(next(it)),
                kappa=float(next(it)), kappa_sigma=float(next(it)), threshold=float(next(it)),
                i_diff=int(next(it)), cutoff=float(next(it)), r_min=float(next(it)), r_max=float(next(it)),
                h4prob=tuple(float(next(it)) for _ in range(20)),
                pro_acceptor_flag=bool(int(next(it))), h4prob_pro_acceptor=float(next(it)),
                sigma_HO=float(next(it)), sigma_NO=float(next(it)), r0_HO=float(next(it)), r0_NO=float(next(it)),
            )
        except (StopIteration, RuntimeError):
            raise ValueError(f"[Helix] section of {path} is truncated") from None
        if '[ABC]' in sections:
            abc = [float(x) for x in sections['[ABC]'][:9]]
            if len(abc) != 9:
                raise ValueError(f"[ABC] section of {path} needs 9 values")
            values.update(an=abc[0], bn=abc[1], cn=abc[2], ah=abc[6], bh=abc[7], ch=abc[8])
        if '[Epsilon]' in sections:
            values['epsilon'] = float(sections['[Epsilon]'][0])
        values.update(overrides)
        return cls(**values)


def _split_sections(text: str) -> dict:
    """Tokens of each enabled ``[Name]`` section (AWSEM-MD reads them token by token)."""
    sections, current = {}, None
    for line in text.splitlines():
        stripped = line.strip()
        header = re.match(r'^(\[[^\]]+\])(\S*)', stripped)
        if header:
            name, suffix = header.groups()
            current = name if suffix != '-' else None
            if current is not None:
                sections[current] = []
            continue
        if current is not None and stripped and not stripped.startswith('#'):
            sections[current].extend(stripped.split())
    return sections


# ----------------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, eq=False)
class HelixPairs:
    """Helical pairs (residue indices) and the particles each one uses."""
    i: np.ndarray            # acceptor residue
    j: np.ndarray            # donor residue, i + i_diff
    o_i: np.ndarray          # O_i
    ca_jm1: np.ndarray       # CA_{j-1}
    ca_j: np.ndarray         # CA_j
    o_jm1: np.ndarray        # O_{j-1}
    propensity: np.ndarray   # h4prob'(a_i) + h4prob(a_j)

    def __len__(self):
        return len(self.i)


# rq-33497404 rq-51684635
@dataclasses.dataclass(frozen=True, eq=False)
class HelixTopology:
    """Residue-to-particle map of a coarse-grained (CA, O, CB) protein.

    ``sequence`` has one letter per residue (all chains, in particle order);
    ``chain_index``, ``ca``, ``o`` and ``cb`` have one entry per residue, with ``-1``
    for a missing bead (``cb`` is ``-1`` exactly for glycine). Residues of a chain
    must be contiguous. Pairs ``(i, i + i_diff)`` exist within a chain (the last
    residue can be a donor); bead pairs count in the density when they are in
    different chains or more than one residue apart.
    """
    sequence: str
    chain_index: np.ndarray
    ca: np.ndarray
    o: np.ndarray
    cb: np.ndarray
    n_particles: int
    density_bead: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        n = len(self.sequence)
        arrays = {}
        for name in ('chain_index', 'ca', 'o', 'cb'):
            a = np.array(getattr(self, name), dtype=np.int64)
            if a.shape != (n,):
                raise ValueError(f"{name} must have one entry per residue ({n}), got shape {a.shape}")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
            arrays[name] = a
        for k, aa in enumerate(self.sequence):
            if aa not in AA:
                raise ValueError(f"Residue {k} has unknown type {aa!r}; expected one of {AA}")
        if n and np.any(arrays['ca'] < 0):
            raise ValueError(f"Every residue needs a CA; missing for residues {np.where(arrays['ca'] < 0)[0]}")
        gly = np.array([aa == 'G' for aa in self.sequence], dtype=bool)
        missing_cb = (~gly) & (arrays['cb'] < 0)
        if np.any(missing_cb):
            raise ValueError(f"Non-glycine residues need a CB; missing for residues {np.where(missing_cb)[0]}")
        for name in ('ca', 'o', 'cb'):
            if np.any(arrays[name] >= self.n_particles):
                raise ValueError(f"{name} index out of range for a system of {self.n_particles} particles")
        chain = arrays['chain_index']
        if n:
            starts = np.r_[True, chain[1:] != chain[:-1]]
            if len(np.unique(chain)) != int(starts.sum()):
                raise ValueError("Residues of each chain must be contiguous")
        bead = np.where(gly, arrays['ca'], arrays['cb'])
        bead.setflags(write=False)
        object.__setattr__(self, 'density_bead', bead)

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    def pairs(self, parameters: HelixParameters) -> HelixPairs:
        """Pairs ``(i, i + i_diff)`` in the same chain whose O_i and O_{j-1} exist."""
        d = parameters.i_diff
        i = np.arange(max(self.n_residues - d, 0))
        j = i + d
        ok = (self.chain_index[i] == self.chain_index[j]) & (self.o[i] >= 0) & (self.o[j - 1] >= 0)
        i, j = i[ok], j[ok]
        prop = np.array([parameters.propensity(self.sequence[a], acceptor=True) + parameters.propensity(self.sequence[b])
                         for a, b in zip(i, j)], dtype=float)
        return HelixPairs(i=i, j=j, o_i=self.o[i], ca_jm1=self.ca[j - 1], ca_j=self.ca[j], o_jm1=self.o[j - 1],
                          propensity=prop)

    def density_mask(self) -> np.ndarray:
        """(n_res, n_res) bool: bead pairs that count in the density."""
        r = np.arange(self.n_residues)
        same_chain = self.chain_index[:, None] == self.chain_index[None, :]
        return (~same_chain) | (np.abs(r[:, None] - r[None, :]) > 1)

    def density_pairs(self):
        """Upper-triangle residue pairs (r < s) that count in the density."""
        r, s = np.nonzero(np.triu(self.density_mask(), k=1))
        return r, s

    @classmethod
    def from_residues(cls, sequence: str, chain_index: Sequence[int], ca, o, cb,
                      n_particles: Optional[int] = None) -> 'HelixTopology':
        ca, o, cb = (np.asarray(x, dtype=np.int64) for x in (ca, o, cb))
        if n_particles is None:
            n_particles = int(max(ca.max(initial=-1), o.max(initial=-1), cb.max(initial=-1)) + 1)
        return cls(sequence=sequence, chain_index=np.asarray(chain_index), ca=ca, o=o, cb=cb,
                   n_particles=n_particles)

    @classmethod
    def from_openawsem(cls, oa) -> 'HelixTopology':
        """From an ``OpenMMAWSEMSystem`` (``ca``, ``o``, ``cb``, ``seq``, ``chain_starts``, ``chain_ends``)."""
        chain_index = np.zeros(oa.nres, dtype=np.int64)
        for k, (start, end) in enumerate(zip(oa.chain_starts, oa.chain_ends)):
            chain_index[start:end + 1] = k
        cb = [c if aa != 'G' else -1 for c, aa in zip(oa.cb, oa.seq)]
        return cls(sequence=oa.seq, chain_index=chain_index, ca=np.asarray(oa.ca), o=np.asarray(oa.o),
                   cb=np.asarray(cb), n_particles=oa.natoms)

    @classmethod
    def from_openmm(cls, topology, sequence: Optional[str] = None) -> 'HelixTopology':
        """From an ``openmm.app.Topology`` with atoms named CA, O and CB.

        Residues without CA are skipped. Pass ``sequence`` when residue names are not
        informative (OpenAWSEM's generic ``NGP``).
        """
        seq, chains, ca, o, cb, chain_ids = [], [], [], [], [], {}
        for residue in topology.residues():
            atoms = {atom.name: atom.index for atom in residue.atoms()}
            if 'CA' not in atoms:
                continue
            seq.append(THREE_TO_ONE.get(residue.name.upper(), 'X'))
            chains.append(chain_ids.setdefault(residue.chain.index, len(chain_ids)))
            ca.append(atoms['CA'])
            o.append(atoms.get('O', -1))
            cb.append(atoms.get('CB', -1))
        seq = ''.join(seq) if sequence is None else sequence
        if len(seq) != len(ca):
            raise ValueError(f"sequence has {len(seq)} residues but the topology has {len(ca)} CA atoms")
        return cls(sequence=seq, chain_index=np.array(chains, dtype=np.int64), ca=np.array(ca, dtype=np.int64),
                   o=np.array(o, dtype=np.int64), cb=np.array(cb, dtype=np.int64), n_particles=topology.getNumAtoms())


# ----------------------------------------------------------------------------
# Reference: loop-by-loop port of AWSEM-MD
# ----------------------------------------------------------------------------
# rq-3bf0f6b6 rq-44b5d559 rq-f3605e1e
def reference_energy_forces(positions, topology: HelixTopology, parameters: HelixParameters,
                            delta_helix_xi: float = DELTA_HELIX_XI, return_terms: bool = False):
    """Energy (kcal/mol) and forces (kcal/mol/A) exactly as AWSEM-MD's ``compute_pair`` computes them.

    Unvectorized on purpose, so that it reads side by side with ``fix_backbone.cpp``:
    density (``helix_ro``), switch (``helix_sigma_h``), ``compute_helix_dtheta_pair`` for
    each pair and the density forces of the second neighbour loop, which AWSEM-MD skips
    when ``|dE/drho_r + dE/drho_s| <= delta_helix_xi`` (use 0 for the exact gradient).
    """
    x = np.asarray(positions, dtype=float)
    p, t = parameters, topology
    n = t.n_residues
    forces = np.zeros_like(x)
    energy = 0.0
    rmin_theta, rmax_theta = p.theta_range
    rmin_sq, rmax_sq = rmin_theta ** 2, rmax_theta ** 2
    bead, chain = t.density_bead, t.chain_index

    def theta_pair(r):
        t_min = math.tanh(p.kappa * (r - p.r_min))
        t_max = math.tanh(p.kappa * (p.r_max - r))
        theta = 0.25 * (1.0 + t_min) * (1.0 + t_max)
        return theta, p.kappa * theta * (t_max - t_min) / r

    rho = np.zeros(n)
    for r in range(n):
        for s in range(r + 1, n):
            if chain[r] == chain[s] and abs(r - s) <= 1:
                continue
            dx = x[bead[r]] - x[bead[s]]
            rsq = dx @ dx
            if rmin_sq < rsq < rmax_sq:
                theta, _ = theta_pair(math.sqrt(rsq))
                rho[r] += theta
                rho[s] += theta
    th = np.tanh(p.kappa_sigma * (rho - p.threshold))
    sigma_h = 0.5 * (1.0 - th)
    sigma_h_prd = -p.kappa_sigma * sigma_h * (1.0 + th)

    xi_1, xi_2 = np.zeros(n), np.zeros(n)
    pairs = t.pairs(p)
    pair_energy = np.zeros(len(pairs))
    for k in range(len(pairs)):
        i, j = pairs.i[k], pairs.j[k]
        xo = x[pairs.o_i[k]]
        xa, xc, xp = x[pairs.ca_jm1[k]], x[pairs.ca_j[k]], x[pairs.o_jm1[k]]
        x_no = xo - (p.an * xa + p.bn * xc + p.cn * xp)
        x_ho = xo - (p.ah * xa + p.bh * xc + p.ch * xp)
        r_no = math.sqrt(x_no @ x_no)
        if not r_no < p.cutoff:
            continue
        r_ho = math.sqrt(x_ho @ x_ho)
        dr_no, dr_ho = r_no - p.r0_NO, r_ho - p.r0_HO
        dr_no_sigma, dr_ho_sigma = dr_no / p.sigma_NO ** 2, dr_ho / p.sigma_HO ** 2
        pair_theta = -p.k_eff * pairs.propensity[k] * math.exp(-0.5 * (dr_no * dr_no_sigma + dr_ho * dr_ho_sigma))
        prd_no, prd_ho = -dr_no_sigma / r_no, -dr_ho_sigma / r_ho
        sigma_gamma = p.gamma_p + (p.gamma_w - p.gamma_p) * sigma_h[i] * sigma_h[j]
        xi_1[i] = (p.gamma_w - p.gamma_p) * pair_theta * sigma_h_prd[i] * sigma_h[j]
        xi_2[i] = (p.gamma_w - p.gamma_p) * pair_theta * sigma_h[i] * sigma_h_prd[j]
        v = sigma_gamma * pair_theta
        energy += v
        pair_energy[k] = v
        forces[pairs.ca_jm1[k]] -= -v * (p.an * prd_no * x_no + p.ah * prd_ho * x_ho)
        forces[pairs.ca_j[k]] -= -v * (p.bn * prd_no * x_no + p.bh * prd_ho * x_ho)
        forces[pairs.o_jm1[k]] -= -v * (p.cn * prd_no * x_no + p.ch * prd_ho * x_ho)
        forces[pairs.o_i[k]] -= v * (prd_no * x_no + prd_ho * x_ho)

    d = p.i_diff
    for r in range(n):
        for s in range(r + 1, n):
            if chain[r] == chain[s] and abs(r - s) <= 1:
                continue
            dx = x[bead[r]] - x[bead[s]]
            rsq = dx @ dx
            if not rmin_sq < rsq < rmax_sq:
                continue
            factor = xi_1[r] + xi_1[s]
            if r - d >= 0:
                factor += xi_2[r - d]
            if s - d >= 0:
                factor += xi_2[s - d]
            if abs(factor) > delta_helix_xi:
                _, prd_theta = theta_pair(math.sqrt(rsq))
                forces[bead[r]] += -factor * prd_theta * dx
                forces[bead[s]] -= -factor * prd_theta * dx
    if return_terms:
        return energy, forces, {'rho': rho, 'sigma_h': sigma_h, 'pair_energy': pair_energy, 'pairs': pairs}
    return energy, forces


# ----------------------------------------------------------------------------
# NumPy backend
# ----------------------------------------------------------------------------
def _kdtree():
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    return cKDTree


def _neighbor_mode(neighbors, n_residues, dense_max, rmax_theta):
    if neighbors not in ('auto', 'dense', 'kdtree'):
        raise ValueError("neighbors must be 'auto', 'dense' or 'kdtree'")
    tree = _kdtree()
    if neighbors == 'auto':
        return 'kdtree' if (n_residues > dense_max and tree is not None and np.isfinite(rmax_theta)) else 'dense'
    if neighbors == 'kdtree' and (tree is None or not np.isfinite(rmax_theta)):
        raise ValueError("neighbors='kdtree' needs scipy and parameters.theta_cutoff=True")
    return neighbors


def _kdtree_pairs(topology, xb, rmax_theta):
    found = _kdtree()(xb).query_pairs(rmax_theta, output_type='ndarray')
    r, s = found[:, 0], found[:, 1]
    keep = (topology.chain_index[r] != topology.chain_index[s]) | (np.abs(r - s) > 1)
    return np.ascontiguousarray(r[keep]), np.ascontiguousarray(s[keep])


# rq-cb6fa29b
class NumpyHelix:
    """Vectorized energy and analytic forces (Angstrom, kcal/mol).

    The density uses all allowed bead pairs (``neighbors='dense'``) or a k-d tree
    restricted to the well's range (``'kdtree'``); ``'auto'`` takes the k-d tree above
    ``dense_max`` residues. ``delta_helix_xi`` > 0 reproduces AWSEM-MD's truncation of
    small density forces (default 0: exact gradient).
    """

    def __init__(self, topology: HelixTopology, parameters: HelixParameters, neighbors: str = 'auto',
                 dense_max: int = 100, delta_helix_xi: float = 0.0):
        self.topology = topology
        self.parameters = parameters
        self.delta_helix_xi = delta_helix_xi
        self.pairs = topology.pairs(parameters)
        self.bead = topology.density_bead
        self.rmin_theta, self.rmax_theta = parameters.theta_range
        self.neighbors = _neighbor_mode(neighbors, topology.n_residues, dense_max, self.rmax_theta)
        if self.neighbors == 'dense':
            self._r, self._s = topology.density_pairs()

    def _density_pairs(self, xb):
        if self.neighbors == 'dense':
            r, s = self._r, self._s
        else:
            r, s = _kdtree_pairs(self.topology, xb, self.rmax_theta)
        dx = xb[r] - xb[s]
        dist = np.sqrt(np.einsum('ij,ij->i', dx, dx))
        inside = (dist > self.rmin_theta) & (dist < self.rmax_theta)
        return r[inside], s[inside], dx[inside], dist[inside]

    def density(self, positions) -> np.ndarray:
        """Local density of every residue."""
        p = self.parameters
        r, s, _, dist = self._density_pairs(np.asarray(positions, dtype=float)[self.bead])
        theta = 0.25 * (1 + np.tanh(p.kappa * (dist - p.r_min))) * (1 + np.tanh(p.kappa * (p.r_max - dist)))
        n = self.topology.n_residues
        return np.bincount(r, theta, minlength=n) + np.bincount(s, theta, minlength=n)

    def compute(self, positions):
        p = self.parameters
        x = np.asarray(positions, dtype=float)
        n = self.topology.n_residues
        forces = np.zeros_like(x)
        r, s, dx, dist = self._density_pairs(x[self.bead])
        t_min = np.tanh(p.kappa * (dist - p.r_min))
        t_max = np.tanh(p.kappa * (p.r_max - dist))
        theta = 0.25 * (1.0 + t_min) * (1.0 + t_max)
        rho = np.bincount(r, theta, minlength=n) + np.bincount(s, theta, minlength=n)
        th = np.tanh(p.kappa_sigma * (rho - p.threshold))
        h = 0.5 * (1.0 - th)
        dh = -0.5 * p.kappa_sigma * (1.0 - th * th)

        pr = self.pairs
        xo, xa, xc, xp = x[pr.o_i], x[pr.ca_jm1], x[pr.ca_j], x[pr.o_jm1]
        x_no = xo - (p.an * xa + p.bn * xc + p.cn * xp)
        x_ho = xo - (p.ah * xa + p.bh * xc + p.ch * xp)
        r_no = np.sqrt(np.einsum('ij,ij->i', x_no, x_no))
        r_ho = np.sqrt(np.einsum('ij,ij->i', x_ho, x_ho))
        on = r_no < p.cutoff
        dr_no, dr_ho = r_no - p.r0_NO, r_ho - p.r0_HO
        hb = np.exp(-0.5 * (dr_no ** 2 / p.sigma_NO ** 2 + dr_ho ** 2 / p.sigma_HO ** 2)) * on
        pair_theta = -p.k_eff * pr.propensity * hb
        hi, hj = h[pr.i], h[pr.j]
        v = (p.gamma_p + (p.gamma_w - p.gamma_p) * hi * hj) * pair_theta
        energy = float(v.sum())

        # Force on O_i along each distance; N_j/H_j take the opposite, shared by their atoms.
        g_no = (v * dr_no / (p.sigma_NO ** 2 * np.where(on, r_no, 1.0)))[:, None] * x_no
        g_ho = (v * dr_ho / (p.sigma_HO ** 2 * np.where(on, r_ho, 1.0)))[:, None] * x_ho
        np.add.at(forces, pr.o_i, g_no + g_ho)
        np.add.at(forces, pr.ca_jm1, -(p.an * g_no + p.ah * g_ho))
        np.add.at(forces, pr.ca_j, -(p.bn * g_no + p.bh * g_ho))
        np.add.at(forces, pr.o_jm1, -(p.cn * g_no + p.ch * g_ho))

        dgamma = (p.gamma_w - p.gamma_p) * pair_theta
        de_drho = np.bincount(pr.i, dgamma * dh[pr.i] * hj, minlength=n) + \
            np.bincount(pr.j, dgamma * hi * dh[pr.j], minlength=n)
        factor = de_drho[r] + de_drho[s]
        if self.delta_helix_xi > 0:
            factor = np.where(np.abs(factor) > self.delta_helix_xi, factor, 0.0)
        f_pair = (-factor * p.kappa * theta * (t_max - t_min) / dist)[:, None] * dx
        fb = np.zeros((n, 3))
        np.add.at(fb, r, f_pair)
        np.add.at(fb, s, -f_pair)
        np.add.at(forces, self.bead, fb)
        return energy, forces

    def energy(self, positions) -> float:
        return self.compute(positions)[0]


def numpy_energy_forces(positions, topology: HelixTopology, parameters: HelixParameters, **kwargs):
    """One-shot :class:`NumpyHelix` evaluation."""
    return NumpyHelix(topology, parameters, **kwargs).compute(positions)


# ----------------------------------------------------------------------------
# numba backend (numba is optional; the kernel is compiled on first use)
# ----------------------------------------------------------------------------
def _helix_kernel(x, bead, pair_r, pair_s, pi, pj, o_i, ca_jm1, ca_j, o_jm1, propensity, c, delta_xi):
    (k_helix, gamma_p, gamma_w, kappa, kappa_sigma, threshold, r_min, r_max, rmin_sq, rmax_sq, cutoff,
     r0_no, r0_ho, s_no, s_ho, an, bn, cn, ah, bh, ch) = c
    n = bead.shape[0]
    forces = np.zeros_like(x)
    rho = np.zeros(n)
    m = pair_r.shape[0]
    inside = np.empty(m, dtype=np.int64)
    prd = np.empty(m)
    n_inside = 0
    for k in range(m):
        r, s = pair_r[k], pair_s[k]
        br, bs = bead[r], bead[s]
        dx0 = x[br, 0] - x[bs, 0]
        dx1 = x[br, 1] - x[bs, 1]
        dx2 = x[br, 2] - x[bs, 2]
        rsq = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
        if rsq > rmin_sq and rsq < rmax_sq:
            d = np.sqrt(rsq)
            t_min = np.tanh(kappa * (d - r_min))
            t_max = np.tanh(kappa * (r_max - d))
            theta = 0.25 * (1.0 + t_min) * (1.0 + t_max)
            rho[r] += theta
            rho[s] += theta
            inside[n_inside] = k
            prd[n_inside] = kappa * theta * (t_max - t_min) / d
            n_inside += 1
    h = np.empty(n)
    dh = np.empty(n)
    for r in range(n):
        th = np.tanh(kappa_sigma * (rho[r] - threshold))
        h[r] = 0.5 * (1.0 - th)
        dh[r] = -0.5 * kappa_sigma * (1.0 - th * th)
    de_drho = np.zeros(n)
    energy = 0.0
    xno = np.empty(3)
    xho = np.empty(3)
    for k in range(pi.shape[0]):
        a, b, q, oo = ca_jm1[k], ca_j[k], o_jm1[k], o_i[k]
        for t in range(3):
            xno[t] = x[oo, t] - (an * x[a, t] + bn * x[b, t] + cn * x[q, t])
            xho[t] = x[oo, t] - (ah * x[a, t] + bh * x[b, t] + ch * x[q, t])
        r_no = np.sqrt(xno[0] ** 2 + xno[1] ** 2 + xno[2] ** 2)
        if not r_no < cutoff:
            continue
        r_ho = np.sqrt(xho[0] ** 2 + xho[1] ** 2 + xho[2] ** 2)
        dr_no = r_no - r0_no
        dr_ho = r_ho - r0_ho
        pair_theta = -k_helix * propensity[k] * np.exp(-0.5 * (dr_no * dr_no / s_no ** 2 + dr_ho * dr_ho / s_ho ** 2))
        i, j = pi[k], pj[k]
        v = (gamma_p + (gamma_w - gamma_p) * h[i] * h[j]) * pair_theta
        energy += v
        de_drho[i] += (gamma_w - gamma_p) * pair_theta * dh[i] * h[j]
        de_drho[j] += (gamma_w - gamma_p) * pair_theta * h[i] * dh[j]
        gno = v * dr_no / (s_no ** 2 * r_no)
        gho = v * dr_ho / (s_ho ** 2 * r_ho)
        for t in range(3):
            g1 = gno * xno[t]
            g2 = gho * xho[t]
            forces[oo, t] += g1 + g2
            forces[a, t] -= an * g1 + ah * g2
            forces[b, t] -= bn * g1 + bh * g2
            forces[q, t] -= cn * g1 + ch * g2
    for u in range(n_inside):
        k = inside[u]
        r, s = pair_r[k], pair_s[k]
        factor = de_drho[r] + de_drho[s]
        if not abs(factor) > delta_xi:
            continue
        br, bs = bead[r], bead[s]
        f = -factor * prd[u]
        for t in range(3):
            g = f * (x[br, t] - x[bs, t])
            forces[br, t] += g
            forces[bs, t] -= g
    return energy, forces


_COMPILED_KERNEL = []


def _numba_kernel():
    if not _COMPILED_KERNEL:
        import numba
        _COMPILED_KERNEL.append(numba.njit(cache=True)(_helix_kernel))
    return _COMPILED_KERNEL[0]


# rq-149bf633
class NumbaHelix:
    """Compiled energy and exact forces (Angstrom, kcal/mol); needs numba.

    Candidate density pairs are precomputed (``neighbors='dense'``) or found with a k-d
    tree each call (``'kdtree'``); ``'auto'`` takes the k-d tree above ``dense_max``
    residues. ``delta_helix_xi`` < 0 keeps every density force; 1e-6 mimics AWSEM-MD.
    """

    def __init__(self, topology: HelixTopology, parameters: HelixParameters, delta_helix_xi: float = -1.0,
                 neighbors: str = 'auto', dense_max: int = 300):
        p = parameters
        self.topology = topology
        self.parameters = parameters
        self._kernel = _numba_kernel()
        pairs = topology.pairs(p)
        rmin_theta, rmax_theta = p.theta_range
        self.rmax_theta = rmax_theta
        self.neighbors = _neighbor_mode(neighbors, topology.n_residues, dense_max, rmax_theta)
        if self.neighbors == 'dense':
            self._dense_pairs = tuple(np.ascontiguousarray(a) for a in topology.density_pairs())
        self._bead = np.ascontiguousarray(topology.density_bead)
        self._args = (
            pairs.i.copy(), pairs.j.copy(), pairs.o_i.copy(), pairs.ca_jm1.copy(), pairs.ca_j.copy(),
            pairs.o_jm1.copy(), pairs.propensity.copy(),
            (p.k_eff, p.gamma_p, p.gamma_w, p.kappa, p.kappa_sigma, p.threshold, p.r_min, p.r_max,
             rmin_theta ** 2, rmax_theta ** 2 if np.isfinite(rmax_theta) else np.inf, p.cutoff,
             p.r0_NO, p.r0_HO, p.sigma_NO, p.sigma_HO, p.an, p.bn, p.cn, p.ah, p.bh, p.ch),
            float(delta_helix_xi),
        )

    def compute(self, positions):
        x = np.ascontiguousarray(positions, dtype=np.float64)
        if self.neighbors == 'dense':
            r, s = self._dense_pairs
        else:
            r, s = _kdtree_pairs(self.topology, x[self._bead], self.rmax_theta)
        energy, forces = self._kernel(x, self._bead, r, s, *self._args)
        return float(energy), forces

    def energy(self, positions) -> float:
        return self.compute(positions)[0]


# ----------------------------------------------------------------------------
# PyTorch backend (torch is optional; the Module class is created on first use)
# ----------------------------------------------------------------------------
_HELIX_MODULE_CLASS = []


def _helix_module_class():
    if _HELIX_MODULE_CLASS:
        return _HELIX_MODULE_CLASS[0]
    import torch

    class HelixModule(torch.nn.Module):
        """Helix energy; ``forward(positions) -> energy`` (TorchScript-compatible)."""

        def __init__(self, topology: HelixTopology, parameters: HelixParameters, units: str = 'awsem',
                     energy_scale: Optional[float] = None, dtype=None, learnable: bool = False):
            super().__init__()
            if units not in ('awsem', 'openmm'):
                raise ValueError("units must be 'awsem' or 'openmm'")
            dtype = torch.float64 if dtype is None else dtype
            p = parameters
            pairs = topology.pairs(p)
            r, s = topology.density_pairs()
            rmin_theta, rmax_theta = p.theta_range

            def long(a):
                return torch.from_numpy(np.array(a, dtype=np.int64))
            self.register_buffer('bead', long(topology.density_bead))
            self.register_buffer('rho_r', long(r))
            self.register_buffer('rho_s', long(s))
            self.register_buffer('pi', long(pairs.i))
            self.register_buffer('pj', long(pairs.j))
            self.register_buffer('o_i', long(pairs.o_i))
            self.register_buffer('ca_jm1', long(pairs.ca_jm1))
            self.register_buffer('ca_j', long(pairs.ca_j))
            self.register_buffer('o_jm1', long(pairs.o_jm1))
            # 21-entry propensity table: the 20 h4prob values and the proline-acceptor value.
            seq = topology.sequence
            self.register_buffer('type_i', long([20 if (seq[a] == 'P' and p.pro_acceptor_flag) else AA.index(seq[a])
                                                 for a in pairs.i]))
            self.register_buffer('type_j', long([AA.index(seq[b]) for b in pairs.j]))
            values = {'h4': torch.tensor(list(p.h4prob) + [p.h4prob_pro_acceptor], dtype=dtype),
                      'k_helix': torch.tensor(p.k_helix, dtype=dtype),
                      'gamma_p': torch.tensor(p.gamma_p, dtype=dtype),
                      'gamma_w': torch.tensor(p.gamma_w, dtype=dtype)}
            for name, value in values.items():
                if learnable:
                    setattr(self, name, torch.nn.Parameter(value))
                else:
                    self.register_buffer(name, value)
            self.epsilon = float(p.epsilon)
            self.n_residues = int(topology.n_residues)
            self.length_scale = float(1.0 / ANGSTROM_TO_NM if units == 'openmm' else 1.0)
            if energy_scale is None:
                energy_scale = KCAL_TO_KJ if units == 'openmm' else 1.0
            self.energy_scale = float(energy_scale)
            self.kappa, self.kappa_sigma, self.threshold = float(p.kappa), float(p.kappa_sigma), float(p.threshold)
            self.r_min, self.r_max = float(p.r_min), float(p.r_max)
            self.rmin_theta = float(rmin_theta)
            self.rmax_theta = float(rmax_theta) if np.isfinite(rmax_theta) else 1e30
            self.cutoff = float(p.cutoff)
            self.r0_no, self.r0_ho = float(p.r0_NO), float(p.r0_HO)
            self.s_no, self.s_ho = float(p.sigma_NO), float(p.sigma_HO)
            self.an, self.bn, self.cn = float(p.an), float(p.bn), float(p.cn)
            self.ah, self.bh, self.ch = float(p.ah), float(p.bh), float(p.ch)

        def density(self, x: torch.Tensor) -> torch.Tensor:
            """Local density of every residue (positions in Angstrom)."""
            xb = x[self.bead]
            d = torch.linalg.norm(xb[self.rho_r] - xb[self.rho_s], dim=1)
            inside = (d > self.rmin_theta) & (d < self.rmax_theta)
            theta = 0.25 * (1.0 + torch.tanh(self.kappa * (d - self.r_min))) * \
                (1.0 + torch.tanh(self.kappa * (self.r_max - d)))
            theta = torch.where(inside, theta, torch.zeros_like(theta))
            rho = torch.zeros(self.n_residues, dtype=x.dtype, device=x.device)
            return rho.index_add(0, self.rho_r, theta).index_add(0, self.rho_s, theta)

        def forward(self, positions: torch.Tensor) -> torch.Tensor:
            x = positions.to(self.h4.dtype) * self.length_scale
            h = 0.5 * (1.0 - torch.tanh(self.kappa_sigma * (self.density(x) - self.threshold)))
            xo, xa, xc, xp = x[self.o_i], x[self.ca_jm1], x[self.ca_j], x[self.o_jm1]
            r_no = torch.linalg.norm(xo - (self.an * xa + self.bn * xc + self.cn * xp), dim=1)
            r_ho = torch.linalg.norm(xo - (self.ah * xa + self.bh * xc + self.ch * xp), dim=1)
            hb = torch.exp(-0.5 * ((r_no - self.r0_no) ** 2 / self.s_no ** 2 +
                                   (r_ho - self.r0_ho) ** 2 / self.s_ho ** 2))
            hb = torch.where(r_no < self.cutoff, hb, torch.zeros_like(hb))
            sigma_gamma = self.gamma_p + (self.gamma_w - self.gamma_p) * h[self.pi] * h[self.pj]
            propensity = self.h4[self.type_i] + self.h4[self.type_j]
            energy = -self.k_helix * self.epsilon * torch.sum(propensity * hb * sigma_gamma)
            return energy * self.energy_scale

    _HELIX_MODULE_CLASS.append(HelixModule)
    return HelixModule


# rq-64c066a5
def make_helix_module(topology: HelixTopology, parameters: HelixParameters, units: str = 'awsem',
                      energy_scale: Optional[float] = None, dtype=None, learnable: bool = False):
    """A ``torch.nn.Module`` computing the helix energy; needs PyTorch.

    ``units='awsem'``: positions in Angstrom, energy in kcal/mol. ``units='openmm'``:
    nm and kJ/mol (what TorchForce expects). With ``learnable=True``, ``k_helix``,
    ``gamma_p``, ``gamma_w`` and ``h4`` (20 propensities + the proline-acceptor value)
    are ``torch.nn.Parameter``s.
    """
    return _helix_module_class()(topology, parameters, units=units, energy_scale=energy_scale, dtype=dtype,
                                 learnable=learnable)


# rq-e978a51c
def torch_energy_forces(positions, topology: HelixTopology, parameters: HelixParameters, module=None):
    """Energy (kcal/mol) and autograd forces (kcal/mol/A) as NumPy."""
    import torch
    module = make_helix_module(topology, parameters) if module is None else module
    x = torch.tensor(np.asarray(positions, dtype=float), requires_grad=True)
    energy = module(x)
    (grad,) = torch.autograd.grad(energy, x)
    return float(energy.detach()), -grad.numpy()


# ----------------------------------------------------------------------------
# OpenMM force builders (topology indices, AWSEM-unit parameters)
# ----------------------------------------------------------------------------
def _nm(x):
    return x * ANGSTROM_TO_NM


def _hb_expression(p: HelixParameters, o=1, a=2, c=3, q=4) -> str:
    """Definitions computing ``hb``: the O-N/O-H Gaussian of O_{o} with N/H built from
    CA_{a} (j-1), CA_{c} (j) and O_{q} (j-1), zero for ``R_NO >= cutoff``."""
    def vsite(w1, w2, w3, axis):
        return f"{w1!r}*{axis}{a}+{w2!r}*{axis}{c}+{w3!r}*{axis}{q}"
    lines = [
        f"hb=exp(-(rNO-{_nm(p.r0_NO)!r})^2/{2 * _nm(p.sigma_NO) ** 2!r}"
        f"-(rHO-{_nm(p.r0_HO)!r})^2/{2 * _nm(p.sigma_HO) ** 2!r})*step({_nm(p.cutoff)!r}-rNO)",
        f"rNO=pointdistance(x{o},y{o},z{o},nx,ny,nz)",
        f"rHO=pointdistance(x{o},y{o},z{o},hx,hy,hz)",
    ]
    for axis in 'xyz':
        lines.append(f"n{axis}={vsite(p.an, p.bn, p.cn, axis)}")
        lines.append(f"h{axis}={vsite(p.ah, p.bh, p.ch, axis)}")
    return ';\n'.join(lines)


def _well_expression(p: HelixParameters, dist: str) -> str:
    """Density well theta(dist), dist in nm, with AWSEM-MD's hard range."""
    kappa = p.kappa / ANGSTROM_TO_NM
    expr = (f"0.25*(1+tanh({kappa!r}*({dist}-{_nm(p.r_min)!r})))"
            f"*(1+tanh({kappa!r}*({_nm(p.r_max)!r}-{dist})))")
    if p.theta_cutoff:
        rmin_theta, rmax_theta = p.theta_range
        expr += f"*step({dist}-{_nm(rmin_theta)!r})*step({_nm(rmax_theta)!r}-{dist})"
    return expr


def _switch(p: HelixParameters, rho: str) -> str:
    return f"0.5*(1-tanh({p.kappa_sigma!r}*({rho}-{p.threshold!r})))"


def _finish(forces, force_group):
    if force_group is not None:
        for f in forces:
            f.setForceGroup(force_group)
    return forces


def build_frozen_density(topology: HelixTopology, parameters: HelixParameters, energy_scale: float = KCAL_TO_KJ,
                         force_group: Optional[int] = None, gamma: Optional[float] = None, positions=None) -> List:
    """Not exact. One CustomCompoundBondForce over (O_i, CA_{j-1}, CA_j, O_{j-1}).

    The environment weight is ``gamma`` (default gamma_p: every pair buried, as
    :func:`hydrogenBondTerms.helical_term` assumes) or, given ``positions`` (Angstrom),
    each pair's ``gamma_p + (gamma_w - gamma_p) H(rho_i) H(rho_j)`` in that structure:
    exact energy there, no density forces.
    """
    p = parameters
    pairs = topology.pairs(p)
    if positions is not None:
        if gamma is not None:
            raise ValueError("give either gamma or positions, not both")
        rho = NumpyHelix(topology, p).density(positions)
        h = 0.5 * (1.0 - np.tanh(p.kappa_sigma * (rho - p.threshold)))
        weight = p.gamma_p + (p.gamma_w - p.gamma_p) * h[pairs.i] * h[pairs.j]
    else:
        weight = np.full(len(pairs), p.gamma_p if gamma is None else gamma)
    force = openmm.CustomCompoundBondForce(4, "-kp*hb;\n" + _hb_expression(p))
    force.addPerBondParameter('kp')
    for k in range(len(pairs)):
        force.addBond([int(pairs.o_i[k]), int(pairs.ca_jm1[k]), int(pairs.ca_j[k]), int(pairs.o_jm1[k])],
                      [p.k_eff * pairs.propensity[k] * weight[k] * energy_scale])
    force.setName('HelixFrozenDensity')
    return _finish([force], force_group)


def _slot_lists(topology: HelixTopology):
    """For each residue: the other residues, sequence neighbours first, and the 0/1
    weights of the first two slots (the only ones the exclusion rule can zero)."""
    n = topology.n_residues
    mask = topology.density_mask()
    slots, weights = [], []
    for r in range(n):
        first = [s for s in (r - 1, r + 1) if 0 <= s < n]
        order = first + [s for s in range(n) if s != r and s not in first]
        w = [1.0 if mask[r, s] else 0.0 for s in order[:2]]
        slots.append(order)
        weights.append(w + [1.0] * (2 - len(w)))
    return slots, weights


def build_compound(topology: HelixTopology, parameters: HelixParameters, energy_scale: float = KCAL_TO_KJ,
                   force_group: Optional[int] = None, max_residues: Optional[int] = 30) -> List:
    """Exact. One CustomCompoundBondForce whose bonds carry every density bead.

    Each bond has 2N + 4 particles and OpenMM differentiates the expression with respect
    to every one, so Context creation grows faster than N^2 (about 11 s at 20 residues,
    150 s at 50). ``max_residues`` (None to disable) guards against unusable builds.
    """
    p = parameters
    pairs = topology.pairs(p)
    n = topology.n_residues
    if max_residues is not None and n > max_residues:
        raise ValueError(f"The compound method is limited to {max_residues} residues (got {n}): its OpenMM "
                         f"compile time grows faster than N^2. Use 'cvforce', 'python' or 'torchforce', "
                         f"or pass max_residues=None.")
    m = n - 1
    slots, weights = _slot_lists(topology)
    first_i, first_j = 7, 7 + m  # 1-based particle numbers of the first rho_i / rho_j slot
    # Lepton definitions may only use names defined after them: energy, switches,
    # densities, wells, distances.
    lines = [f"-kp*hb*({p.gamma_p!r}+{p.gamma_w - p.gamma_p!r}*Hi*Hj)",
             f"Hi={_switch(p, 'rhoi')}", f"Hj={_switch(p, 'rhoj')}"]
    wells, distances = [], []
    for name, centre, first, wnames in (('rhoi', 5, first_i, ('wi0', 'wi1')), ('rhoj', 6, first_j, ('wj0', 'wj1'))):
        terms = []
        for slot in range(m):
            terms.append(f"{wnames[slot] + '*' if slot < 2 else ''}t{name}{slot}")
            wells.append(f"t{name}{slot}={_well_expression(p, f'd{name}{slot}')}")
            distances.append(f"d{name}{slot}=distance(p{centre},p{first + slot})")
        lines.append(f"{name}=" + ('+'.join(terms) if terms else '0'))
    force = openmm.CustomCompoundBondForce(6 + 2 * m, ';\n'.join(lines + wells + distances) + ';\n' +
                                           _hb_expression(p))
    for name in ('kp', 'wi0', 'wi1', 'wj0', 'wj1'):
        force.addPerBondParameter(name)
    bead = topology.density_bead
    for k in range(len(pairs)):
        i, j = int(pairs.i[k]), int(pairs.j[k])
        particles = [pairs.o_i[k], pairs.ca_jm1[k], pairs.ca_j[k], pairs.o_jm1[k], bead[i], bead[j]]
        particles += [bead[s] for s in slots[i]] + [bead[s] for s in slots[j]]
        force.addBond([int(a) for a in particles], [p.k_eff * pairs.propensity[k] * energy_scale] + weights[i] +
                      weights[j])
    force.setName('HelixCompound')
    return _finish([force], force_group)


def _cv_blocks(pairs, max_cvs: int):
    """Greedy blocks of consecutive pairs with (#densities + #pairs) <= max_cvs."""
    blocks, current, residues = [], [], set()
    for k in range(len(pairs)):
        needed = residues | {int(pairs.i[k]), int(pairs.j[k])}
        if current and len(needed) + len(current) + 1 > max_cvs:
            blocks.append(current)
            current, needed = [], {int(pairs.i[k]), int(pairs.j[k])}
        current.append(k)
        residues = needed
    if current:
        blocks.append(current)
    return blocks


def build_cvforce(topology: HelixTopology, parameters: HelixParameters, energy_scale: float = KCAL_TO_KJ,
                  force_group: Optional[int] = None, max_cvs: int = MAX_CVS) -> List:
    """Exact. The gamma_p part as CustomCompoundBondForce plus one CustomCVForce per block
    of pairs (densities as CustomBondForce and hydrogen-bond factors as one-bond
    CustomCompoundBondForce collective variables, at most ``max_cvs`` per block)."""
    if not 3 <= max_cvs <= MAX_CVS:
        raise ValueError(f"max_cvs must be between 3 and {MAX_CVS}")
    p = parameters
    pairs = topology.pairs(p)
    forces = build_frozen_density(topology, p, energy_scale=energy_scale, gamma=p.gamma_p)
    forces[0].setName('HelixCVConstant')
    if p.gamma_w == p.gamma_p or len(pairs) == 0:
        return _finish(forces, force_group)
    mask, bead = topology.density_mask(), topology.density_bead

    def rho_force(r):
        # A fresh Force per use: a CustomCVForce owns its variables.
        f = openmm.CustomBondForce(_well_expression(p, 'r'))
        for s in np.nonzero(mask[r])[0]:
            f.addBond(int(bead[r]), int(bead[s]))
        return f

    for b, block in enumerate(_cv_blocks(pairs, max_cvs)):
        residues = sorted({int(pairs.i[k]) for k in block} | {int(pairs.j[k]) for k in block})
        terms = [f"hb{k}*H{int(pairs.i[k])}*H{int(pairs.j[k])}" for k in block]
        defs = [f"H{r}={_switch(p, f'rho{r}')}" for r in residues]
        cv = openmm.CustomCVForce('+'.join(terms) + ';\n' + ';\n'.join(defs))
        for r in residues:
            cv.addCollectiveVariable(f'rho{r}', rho_force(r))
        for k in block:
            hb = openmm.CustomCompoundBondForce(4, "c*hb;\n" + _hb_expression(p))
            hb.addPerBondParameter('c')
            hb.addBond([int(pairs.o_i[k]), int(pairs.ca_jm1[k]), int(pairs.ca_j[k]), int(pairs.o_jm1[k])],
                       [-p.k_eff * pairs.propensity[k] * (p.gamma_w - p.gamma_p) * energy_scale])
            cv.addCollectiveVariable(f'hb{k}', hb)
        cv.setName(f'HelixCVDensity{b}')
        forces.append(cv)
    return _finish(forces, force_group)


class _PythonComputation:
    """Picklable PythonForce callback: positions (nm) -> energy (kJ/mol), forces (kJ/mol/nm)."""

    def __init__(self, model, energy_scale):
        self.model = model
        self.energy_scale = energy_scale

    def __call__(self, state):
        x = state.getPositions(asNumpy=True)._value / ANGSTROM_TO_NM
        energy, forces = self.model.compute(x)
        return energy * self.energy_scale, forces * (self.energy_scale / ANGSTROM_TO_NM)


class _TorchModel:
    """``compute(x) -> (energy, forces)`` over the torch module."""

    def __init__(self, topology, parameters):
        self.topology, self.parameters = topology, parameters
        self.module = make_helix_module(topology, parameters)

    def __getstate__(self):
        return {'topology': self.topology, 'parameters': self.parameters}

    def __setstate__(self, state):
        self.__init__(state['topology'], state['parameters'])

    def compute(self, x):
        return torch_energy_forces(x, self.topology, self.parameters, self.module)


class _NumbaModel(NumbaHelix):
    """Picklable NumbaHelix (the compiled kernel is rebuilt after unpickling)."""

    def __init__(self, topology, parameters, **options):
        self._options = options
        super().__init__(topology, parameters, **options)

    def __getstate__(self):
        return {'topology': self.topology, 'parameters': self.parameters, 'options': self._options}

    def __setstate__(self, state):
        self.__init__(state['topology'], state['parameters'], **state['options'])


def build_python(topology: HelixTopology, parameters: HelixParameters, energy_scale: float = KCAL_TO_KJ,
                 force_group: Optional[int] = None, backend: str = 'numba', **backend_options) -> List:
    """Exact. ``openmm.PythonForce`` (OpenMM >= 8.3) over the 'numba', 'numpy' or 'torch' backend."""
    if not hasattr(openmm, 'PythonForce'):
        raise ImportError("openmm.PythonForce needs OpenMM >= 8.3")
    if backend == 'numpy':
        model = NumpyHelix(topology, parameters, **backend_options)
    elif backend == 'numba':
        model = _NumbaModel(topology, parameters, **backend_options)
    elif backend == 'torch':
        model = _TorchModel(topology, parameters)
    else:
        raise ValueError("backend must be 'numba', 'numpy' or 'torch'")
    force = openmm.PythonForce(_PythonComputation(model, energy_scale))
    force.setName(f'HelixPython_{backend}')
    return _finish([force], force_group)


def build_torchforce(topology: HelixTopology, parameters: HelixParameters, energy_scale: float = KCAL_TO_KJ,
                     force_group: Optional[int] = None) -> List:
    """Exact. ``openmmtorch.TorchForce`` over the scripted torch module (needs openmm-torch)."""
    import torch
    from openmmtorch import TorchForce
    module = make_helix_module(topology, parameters, units='openmm', energy_scale=energy_scale)
    with warnings.catch_warnings():  # openmm-torch needs TorchScript, which recent torch deprecates
        warnings.simplefilter('ignore', DeprecationWarning)
        force = TorchForce(torch.jit.script(module))
    force.setUsesPeriodicBoundaryConditions(False)
    force.setName('HelixTorchForce')
    return _finish([force], force_group)


BUILDERS: Dict[str, Callable] = {
    'frozen_density': build_frozen_density,
    'compound': build_compound,
    'cvforce': build_cvforce,
    'python': build_python,
    'torchforce': build_torchforce,
}
EXACT = {'compound', 'cvforce', 'python', 'torchforce'}


# rq-51894fe8 rq-79acb080
def build_helix_forces(method: str, topology: HelixTopology, parameters: Optional[HelixParameters] = None,
                       **options) -> List:
    """The helical term as a list of OpenMM forces built with ``method`` (see :data:`BUILDERS`)."""
    if method not in BUILDERS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[method](topology, HelixParameters() if parameters is None else parameters, **options)


# rq-d22511e5
def combine_forces(forces: List, name: str = 'Helix'):
    """One Force equivalent to the sum of ``forces`` (nested CustomCVForces, <= 32 per level)."""
    level = list(forces)
    if not level:
        raise ValueError("no forces to combine")
    group = level[0].getForceGroup()
    while len(level) > 1:
        merged = []
        for k in range(0, len(level), MAX_CVS):
            chunk = level[k:k + MAX_CVS]
            cv = openmm.CustomCVForce('+'.join(f'f{i}' for i in range(len(chunk))))
            for i, f in enumerate(chunk):
                cv.addCollectiveVariable(f'f{i}', f)
            merged.append(cv)
        level = merged
    level[0].setName(name)
    level[0].setForceGroup(group)
    return level[0]


# rq-d6872219
class OpenMMEvaluator:
    """Energy (kcal/mol) and forces (kcal/mol/A) of forces in a bare System of ``n_particles``."""

    def __init__(self, n_particles: int, forces, platform: str = 'Reference', energy_scale: float = KCAL_TO_KJ,
                 properties: Optional[dict] = None):
        self.system = openmm.System()
        for _ in range(n_particles):
            self.system.addParticle(1.0)
        for f in forces:
            self.system.addForce(f)
        self.energy_scale = energy_scale
        self.context = openmm.Context(self.system, openmm.VerletIntegrator(0.001),
                                      openmm.Platform.getPlatformByName(platform), properties or {})

    def compute(self, positions_angstrom):
        self.context.setPositions(np.asarray(positions_angstrom, dtype=float) * ANGSTROM_TO_NM)
        state = self.context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy()._value / self.energy_scale
        forces = state.getForces(asNumpy=True)._value * (ANGSTROM_TO_NM / self.energy_scale)
        return energy, np.asarray(forces)


# ----------------------------------------------------------------------------
# OpenAWSEM terms
# ----------------------------------------------------------------------------
# rq-191d03cb rq-db316cce
def _setup(oa, k_helical, parameters):
    parameters = HelixParameters() if parameters is None else parameters
    if k_helical is not None:
        if isinstance(k_helical, openmm.unit.Quantity):
            k_helical = k_helical.value_in_unit(openmm.unit.kilojoule_per_mole)
        parameters = parameters.replace(k_helix=k_helical / KCAL_TO_KJ)
    if getattr(oa, 'periodic_box', None):
        warnings.warn("The helical terms ignore periodic images: densities and hydrogen bonds use plain "
                      "distances between the particles.")
    return HelixTopology.from_openawsem(oa), parameters, KCAL_TO_KJ * oa.k_awsem


def _term(method, oa, k_helical, parameters, forceGroup, **options):
    topology, parameters, scale = _setup(oa, k_helical, parameters)
    forces = build_helix_forces(method, topology, parameters, energy_scale=scale, force_group=forceGroup, **options)
    return combine_forces(forces, name=f'Helix_{method}') if len(forces) > 1 else forces[0]


# rq-fd58ca02
def density_dependent_helical_term(oa, k_helical=0.5 * KCAL_TO_KJ, method: str = 'cvforce',
                                   parameters: Optional[HelixParameters] = None, forceGroup: int = 29, **options):
    """AWSEM-MD's density-dependent helical term as one OpenMM Force.

    ``k_helical`` (kJ/mol, or an energy Quantity) replaces ``parameters.k_helix``; the
    default, 0.5 kcal/mol, is the AWSEM-MD 1R69 example and matches
    :func:`hydrogenBondTerms.helical_term` for buried residues. Pass ``k_helical=None``
    to keep ``parameters.k_helix`` (e.g. from
    ``HelixParameters.from_fix_backbone_coeff``). Energies are scaled by ``oa.k_awsem``.
    ``method`` is one of 'cvforce' (default), 'python', 'torchforce', 'compound' or
    'frozen_density'; ``options`` go to the matching ``helical_term_*`` function.
    """
    if method == 'frozen_density':
        return helical_term_frozen_density(oa, k_helical, parameters, forceGroup, **options)
    return _term(method, oa, k_helical, parameters, forceGroup, **options)


# rq-3a1f6e0b
def helical_term_cvforce(oa, k_helical=0.5 * KCAL_TO_KJ, parameters: Optional[HelixParameters] = None,
                         forceGroup: int = 29, max_cvs: int = MAX_CVS):
    """Exact helical term from CustomCVForce blocks (plain OpenMM forces, serializable)."""
    return _term('cvforce', oa, k_helical, parameters, forceGroup, max_cvs=max_cvs)


# rq-9d4c2b77
def helical_term_pythonforce(oa, k_helical=0.5 * KCAL_TO_KJ, parameters: Optional[HelixParameters] = None,
                             forceGroup: int = 29, backend: str = 'numba', **backend_options):
    """Exact helical term as ``openmm.PythonForce`` over the 'numba', 'numpy' or 'torch' backend."""
    return _term('python', oa, k_helical, parameters, forceGroup, backend=backend, **backend_options)


# rq-c80e5a19
def helical_term_torchforce(oa, k_helical=0.5 * KCAL_TO_KJ, parameters: Optional[HelixParameters] = None,
                            forceGroup: int = 29):
    """Exact helical term as ``openmmtorch.TorchForce`` (runs on GPU platforms)."""
    return _term('torchforce', oa, k_helical, parameters, forceGroup)


# rq-4b7f0d23
def helical_term_compound(oa, k_helical=0.5 * KCAL_TO_KJ, parameters: Optional[HelixParameters] = None,
                          forceGroup: int = 29, max_residues: Optional[int] = 30):
    """Exact helical term as one CustomCompoundBondForce; small systems only."""
    return _term('compound', oa, k_helical, parameters, forceGroup, max_residues=max_residues)


# rq-e1c95a60
def helical_term_frozen_density(oa, k_helical=0.5 * KCAL_TO_KJ, parameters: Optional[HelixParameters] = None,
                                forceGroup: int = 29, positions=None, gamma: Optional[float] = None):
    """Not exact: environment weights frozen at ``positions`` (Quantity or nm array) or at ``gamma``.

    With neither, every pair is treated as buried (gamma_p), like
    :func:`hydrogenBondTerms.helical_term`.
    """
    if positions is not None:
        if isinstance(positions, openmm.unit.Quantity):
            positions = positions.value_in_unit(openmm.unit.nanometer)
        positions = np.asarray(positions, dtype=float) / ANGSTROM_TO_NM
    return _term('frozen_density', oa, k_helical, parameters, forceGroup, positions=positions, gamma=gamma)
