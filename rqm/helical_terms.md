# Feature: AWSEM density-dependent helical hydrogen-bond terms <!-- rq-b25527fd -->

The helical term rewards (or penalizes) the i -> i+4 backbone hydrogen bond of an alpha
helix, weighted by the helix propensity of both residues and by how buried the two
residues are. The gold standard is the active `compute_pair` path of `FixBackbone` in
AWSEM-MD (`src/fix_backbone.cpp` of https://github.com/adavtyan/awsemmd, where
`pair_flag` is `#define`-d to 1). `openawsem/functionTerms/helicalTerms.py` provides
several interchangeable implementations of exactly that term (NumPy, numba, PyTorch
and five OpenMM strategies) behind OpenAWSEM term functions, so that the fastest correct
one can be chosen for a given platform and system size, and a test battery checks each
against energies and forces produced by LAMMPS/AWSEM-MD. The density-free
`hydrogenBondTerms.helical_term` is left unchanged.

## Definitions <!-- rq-eaad0b47 -->

All quantities are in AWSEM-MD units (Angstrom, kcal/mol) unless stated otherwise.

### Beads and virtual atoms <!-- rq-3d8754d8 -->

Every residue `r` has a CA bead, an O bead and, except glycine, a CB bead. The
density bead `X_r` is CB, or CA for glycine. The amide N and H of residue `j` are
virtual positions built from the previous residue:

```
N_j = an*CA_{j-1} + bn*CA_j + cn*O_{j-1}
H_j = ah*CA_{j-1} + bh*CA_j + ch*O_{j-1}
```

The coefficients come from the `[ABC]` section; the AWSEM-MD values sum to 1.0000364
(N) and 1.00007 (H), so the term is invariant under rotations about the coordinate
origin but not exactly under translations (nor rotations about another point), and
every implementation reproduces that.

### Helical pairs <!-- rq-51684635 -->

A helical pair `(i, j)` has `j = i + i_diff` with `i` and `j` in the same chain and
both O_i and O_{j-1} present. The last residue of a chain can be a donor `j`. The pair
contributes only if `R_NO = |O_i - N_j| < cutoff` (a hard cutoff).

### Local density <!-- rq-9cf0d859 -->

```
rho_r = sum over s of theta(|X_r - X_s|)
theta(d) = 0.25*(1 + tanh(kappa*(d - r_min)))*(1 + tanh(kappa*(r_max - d)))
```

The sum runs over beads `s` in another chain, or in the same chain with `|r - s| > 1`.
When `theta_cutoff` is true (AWSEM-MD's behaviour), `theta` is zero unless
`r_min - 8 ln(10)/kappa < d < r_max + 8 ln(10)/kappa`.

### Energy <!-- rq-44b5d559 -->

```
H(rho)  = 0.5*(1 - tanh(kappa_sigma*(rho - threshold)))
theta_p = exp(-(R_NO - r0_NO)^2/(2 sigma_NO^2) - (R_HO - r0_HO)^2/(2 sigma_HO^2))
P_p     = h4prob'(a_i) + h4prob(a_j)
E       = sum over pairs p of  -k_helix*epsilon * P_p * theta_p * (gamma_p + (gamma_w - gamma_p)*H(rho_i)*H(rho_j))
```

`h4prob'(a_i)` is `h4prob_pro_acceptor` when `pro_acceptor_flag` is set and residue `i`
is proline, and `h4prob(a_i)` otherwise.

### Forces <!-- rq-f3605e1e -->

Forces are `-dE/dx` on the CA_{j-1}, CA_j, O_{j-1} and O_i beads of every pair (through
N_j and H_j) and on every density bead (through rho). AWSEM-MD skips the density force
of a bead pair when `|dE/drho_r + dE/drho_s| <= 1e-6` (`delta_helix_xi`); the reference
port reproduces this truncation, and the other backends compute the exact gradient
unless asked otherwise.

## Parameters <!-- rq-825cdfb5 -->

`HelixParameters` holds every quantity of the `[Helix]` section (`k_helix`, `gamma_p`,
`gamma_w`, `kappa`, `kappa_sigma`, `threshold`, `i_diff`, `cutoff`, `r_min`, `r_max`, 20
`h4prob` values in ARNDCQEGHILKMFPSTWYV order, `pro_acceptor_flag`,
`h4prob_pro_acceptor`, `sigma_HO`, `sigma_NO`, `r0_HO`, `r0_NO`), the N/H coefficients
of `[ABC]`, `epsilon` from `[Epsilon]`, and `theta_cutoff`. Its defaults are those of
AWSEM-MD `parameters/fix_backbone_coeff.data`: `k_helix=1.5`, `gamma_p=2.0`,
`gamma_w=-1.0`, `kappa=kappa_sigma=7.0`, `threshold=3.0`, `i_diff=4`, `cutoff=6.0`,
`r_min=4.5`, `r_max=6.5`, proline `h4prob=-3.0`, `pro_acceptor_flag=False`,
`h4prob_pro_acceptor=-3.0`, `sigma_HO=0.76`, `sigma_NO=0.68`, `r0_HO=2.06`, `r0_NO=2.98`,
`[ABC]` N = (0.4831806, 0.703282, -0.1864262), H = (0.841, 0.89296, -0.73389),
`epsilon=1`. It is an immutable dataclass (no dependency beyond NumPy and OpenMM).

## Feature API <!-- rq-7effc863 -->

All items live in `openawsem.functionTerms.helicalTerms`. numba, PyTorch and openmm-torch
are optional and imported only by the items that need them.

### Parameters and topology <!-- rq-57ee71cc -->

- `HelixParameters(**fields)` — frozen dataclass of the parameters above. <!-- rq-68a3ed85 -->
  - Raises `ValueError` unless `h4prob` has 20 values, `i_diff` is an integer >= 1 and `sigma_HO`, `sigma_NO`, `kappa`, `kappa_sigma` and `cutoff` are positive; unknown fields raise `TypeError`.
  - `k_eff` is `k_helix*epsilon`; `theta_range` is `(rmin_theta, rmax_theta)` (`(0, inf)` without `theta_cutoff`).
  - `replace(**changes)` returns a validated copy; `as_dict()` returns plain values that `HelixParameters(**d)` accepts.
  - `propensity(aa, acceptor=False)` returns the propensity of a one-letter residue; raises `ValueError` for an unknown letter.
  - `to_section()` renders `[ABC]`, `[Epsilon]` and `[Helix]` in AWSEM-MD format.
  - `awsemmd_default()` equals `HelixParameters()`; `awsemmd_1r69_example()` is the 1R69 example of AWSEM-MD (`k_helix=0.5`, `cutoff=15`, `r0_HO=2.1558`, `r0_NO=2.9862`).
  - `from_fix_backbone_coeff(path, **overrides)` reads `[Helix]`, and `[ABC]`/`[Epsilon]` if present. A section whose header ends with `-` is disabled and ignored. A missing `[Helix]` or a truncated one raises `ValueError`.
- `HelixTopology(sequence, chain_index, ca, o, cb, n_particles)` — residue-to-particle map. <!-- rq-33497404 -->
  - Raises `ValueError` for an unknown residue letter, arrays of the wrong length, a residue without CA, a non-glycine residue without CB, an index outside the system, or non-contiguous chains.
  - `density_bead` is CB, or CA for glycine.
  - `pairs(parameters)` returns `HelixPairs` (`i`, `j`, `o_i`, `ca_jm1`, `ca_j`, `o_jm1`, `propensity`) following the pair rule above; pairs with a missing O are skipped.
  - `density_mask()` and `density_pairs()` give the bead pairs allowed in the density.
  - `from_openawsem(oa)` uses `oa.ca`, `oa.o`, `oa.cb`, `oa.seq`, `oa.chain_starts`, `oa.chain_ends`, `oa.nres` and `oa.natoms`.
  - `from_openmm(topology, sequence=None)` reads atoms named CA, O and CB, maps residue names (including OpenAWSEM's IGL and IPR) to one-letter codes, and takes `sequence` when names are not informative (OpenAWSEM's NGP).
  - `from_residues(sequence, chain_index, ca, o, cb, n_particles=None)` builds one from index arrays.

### Energy backends <!-- rq-6aa17030 -->

- `reference_energy_forces(positions, topology, parameters, delta_helix_xi=1e-6, return_terms=False)` — loop-by-loop port of `compute_pair`; returns `(energy, forces)` and, on request, `rho`, `sigma_h` and per-pair energies. <!-- rq-3bf0f6b6 -->
- `NumpyHelix(topology, parameters, neighbors='auto', dense_max=100, delta_helix_xi=0.0)` — vectorized energy and analytic forces; `compute(x)`, `energy(x)`, `density(x)`. `neighbors` is `'dense'`, `'kdtree'` (needs scipy and `theta_cutoff`) or `'auto'`. <!-- rq-cb6fa29b -->
- `NumbaHelix(topology, parameters, delta_helix_xi=-1.0, neighbors='auto', dense_max=300)` — compiled loops with the exact gradient over precomputed (`'dense'`) or k-d tree (`'kdtree'`) candidate pairs; `compute(x)`. <!-- rq-149bf633 -->
- `make_helix_module(topology, parameters, units='awsem', energy_scale=None, dtype=None, learnable=False)` — TorchScript-compatible `torch.nn.Module`; `forward(positions)` returns the energy. `units='openmm'` takes nm and returns kJ/mol. With `learnable=True`, `k_helix`, `gamma_p`, `gamma_w` and `h4` (the 20 propensities plus the proline-acceptor value) are `torch.nn.Parameter`s. <!-- rq-64c066a5 -->
- `torch_energy_forces(positions, topology, parameters, module=None)` — energy and autograd forces as NumPy. <!-- rq-e978a51c -->

### OpenMM builders <!-- rq-79acb080 -->

- `build_helix_forces(method, topology, parameters=None, **options)` — returns a list of `openmm.Force`; raises `ValueError` for an unknown method. Every builder accepts `energy_scale` (default 4.184 kJ/kcal) and `force_group`. <!-- rq-51894fe8 -->
  - `frozen_density`: CustomCompoundBondForce with the environment weight frozen at `gamma` (default `gamma_p`), or, given `positions` (Angstrom), at each pair's weight in that structure. Not exact.
  - `compound`: exact single CustomCompoundBondForce whose bonds carry every density bead. Refuses systems above `max_residues` (default 30), because OpenMM's expression compilation grows faster than quadratically with the number of particles per bond.
  - `cvforce`: exact; the constant part as CustomCompoundBondForce plus blocks of pairs as CustomCVForces whose collective variables are the needed densities and hydrogen-bond factors, at most `max_cvs` (<= 32) per block.
  - `python`: exact `openmm.PythonForce` (OpenMM >= 8.3) over the `numba` (default), `numpy` or `torch` backend; serializable with `XmlSerializer`.
  - `torchforce`: exact `openmmtorch.TorchForce` over the scripted torch module.
- `combine_forces(forces, name='Helix')` — one Force equal to the sum of `forces`, built from nested CustomCVForces with at most 32 variables each, keeping the force group of the first force; raises `ValueError` for an empty list. <!-- rq-d22511e5 -->
- `OpenMMEvaluator(n_particles, forces, platform='Reference', energy_scale=4.184, properties=None)` — `compute(x_angstrom)` returns energy (kcal/mol) and forces (kcal/mol/Angstrom). <!-- rq-d6872219 -->

### OpenAWSEM terms <!-- rq-e2d6cbae -->

Every term takes an `OpenMMAWSEMSystem` `oa`, returns one `openmm.Force` in `forceGroup`
(default 29), scales energies by `4.184*oa.k_awsem`, and warns when `oa.periodic_box` is
set (periodic images are ignored). `k_helical` (kJ/mol, or an energy Quantity; default
0.5 kcal/mol = 2.092 kJ/mol, the AWSEM-MD 1R69 example, which equals the buried limit of
`hydrogenBondTerms.helical_term`) replaces `parameters.k_helix`; `k_helical=None` keeps it.
`parameters` defaults to `HelixParameters()`.

- `density_dependent_helical_term(oa, k_helical=2.092, method='cvforce', parameters=None, forceGroup=29, **options)` — dispatches to the term of `method` (`cvforce`, `python`, `torchforce`, `compound` or `frozen_density`). <!-- rq-fd58ca02 -->
- `helical_term_cvforce(oa, k_helical, parameters, forceGroup, max_cvs=32)` — the `cvforce` blocks combined into one Force with `combine_forces`. <!-- rq-3a1f6e0b -->
- `helical_term_pythonforce(oa, k_helical, parameters, forceGroup, backend='numba', **backend_options)`. <!-- rq-9d4c2b77 -->
- `helical_term_torchforce(oa, k_helical, parameters, forceGroup)`. <!-- rq-c80e5a19 -->
- `helical_term_compound(oa, k_helical, parameters, forceGroup, max_residues=30)`. <!-- rq-4b7f0d23 -->
- `helical_term_frozen_density(oa, k_helical, parameters, forceGroup, positions=None, gamma=None)` — `positions` is a Quantity or an array in nm; with neither, every pair is buried (`gamma_p`). <!-- rq-e1c95a60 -->

### Gold-standard tooling <!-- rq-6b205020 -->

`devtools/helical_reference/` (development only, not installed):

- `awsemmd_reference.CoarseGrainedProtein` — CA/O/CB(HB) coordinates in AWSEM-MD bead order; `from_pdb(path, chains=None)` reads the first model and first alternate location, places glycine's HB at the ideal CB position, and raises `ValueError` for unknown residues or a residue without O (or without CB, unless glycine); `from_residues(sequences, ca, o, cb_or_hb, chain_names=None)`; `to_pdb(path)`. <!-- rq-b8026202 -->
- `awsemmd_reference.find_lammps()` — the `AWSEMMD_LMP` executable, else `lmp_serial`/`lmp` on `PATH`, else `None`. <!-- rq-843a60e6 -->
- `awsemmd_reference.run_lammps(protein, params, frames=None, lmp=None, workdir=None, keep=False)` — runs AWSEM-MD with only `[Helix]`, `[ABC]` and `[Epsilon]` and returns `(energies, forces)` per frame; raises `FileNotFoundError` without an executable and `RuntimeError` if LAMMPS fails. <!-- rq-e11f916a -->
- `make_reference.py` — regenerates `tests/data/helical_reference/*.npz` and `MANIFEST.json`. <!-- rq-9b29a1d1 -->
- `benchmark.py` and `plot_benchmark.py` — see Performance.

## Reference data <!-- rq-7c04f64f -->

`tests/data/helical_reference/` holds LAMMPS/AWSEM-MD (22 Jul 2025 update 6, AWSEM-MD
commit recorded in `MANIFEST.json`) energies and forces for:

- Helical proteins: 1R69, the 1CTA homodimer, the FIS homodimer, myoglobin (1MBN) and
  the GlpG membrane bundle (2XOV), each in six frames (native, Gaussian noise of 0.25,
  0.5 and 1.0 Angstrom, expanded x1.25, compressed x0.85).
- Controls: histone H4 tail (disordered), ubiquitin (alpha/beta) and Src SH3 (beta).
- Parameter variants: default, the 1R69 example, proline-acceptor flag on, `i_diff=3`,
  every parameter changed, and a 3.2 Angstrom cutoff.
- Edge cases: glycine/proline at acceptor, donor and neighbour positions; chains of 1, 2,
  4, 5 and 6 residues packed against each other; one-residue chains; O-N distances
  straddling the cutoff; compression below the well range; a fully extended chain.
- `openawsem/1mbn_movie.npz`: the 12 frames of `tests/data/1mbn-movie.dcd` on OpenAWSEM's
  1MBN model (`tests/data/1mbn-openmmawsem.pdb`), with AWSEM-MD energies and forces
  mapped onto its 900 atoms.

## Accuracy <!-- rq-e95e6f83 -->

- The reference port reproduces AWSEM-MD to 1e-9 kcal/mol and 1e-9 kcal/mol/Angstrom.
- Exact backends in double precision reproduce AWSEM-MD energies to 1e-6 kcal/mol and
  forces to 1e-5 kcal/mol/Angstrom (the difference is AWSEM-MD's force truncation), and
  each other to 1e-8.

## Performance <!-- rq-60f9eee2 -->

`devtools/helical_reference/benchmark.py` times setup and one energy+force evaluation of
every backend on the requested OpenMM platforms (with optional platform properties) for
helical systems from 20 to several thousand residues, skipping a backend once it exceeds
a time budget, and writes JSON; `plot_benchmark.py` turns it into Markdown tables and a
chart. The recorded results are in `devtools/helical_reference/results/`.

- The `python` method with the numba backend costs at most twice the density-free
  `frozen_density` force at every benchmarked CPU size, which is why both NumPy and numba
  switch to a k-d tree neighbour list (above 100 and 300 residues respectively).
- The `compound` method is limited by Context creation (minutes at 50 residues) and is
  guarded by `max_residues`.
- GPU platforms have not been benchmarked; `cvforce` is the default method until they are.

## Out of scope <!-- rq-7a3fa89f -->

- Membrane (z-dependent) helical propensities.
- Periodic images (the terms warn when `oa.periodic_box` is set).
- A compiled OpenMM plugin.
- Changing `hydrogenBondTerms.helical_term`.

## Gherkin Scenarios <!-- rq-08a5ee36 -->

```gherkin
Feature: AWSEM-MD agreement

  @rq-a261fb7b
  Scenario: The reference port reproduces AWSEM-MD
    Given every stored LAMMPS reference case and frame
    When reference_energy_forces is evaluated with delta_helix_xi = 1e-6
    Then the energy matches within 1e-9 kcal/mol
    And every force component matches within 1e-9 kcal/mol/Angstrom

  @rq-7c690172
  Scenario: Every exact backend reproduces AWSEM-MD energies and forces
    Given every stored reference case and an exact backend (numpy dense, numpy kdtree, numba dense, numba kdtree, torch)
    When the backend computes energy and forces
    Then the energy matches LAMMPS within 1e-6 kcal/mol
    And the forces match LAMMPS within 1e-5 kcal/mol/Angstrom

  @rq-c9509fd3
  Scenario: Every exact OpenMM method reproduces AWSEM-MD on each platform
    Given the helical proteins, the edge cases, and an exact OpenMM method (cvforce, cvforce combined into one Force, python with numpy, numba and torch, torchforce) on the Reference and CPU platforms
    When the energy and forces of each frame are computed in an OpenMM Context
    Then the energy matches LAMMPS within 1e-6 kcal/mol
    And the forces match LAMMPS within 1e-5 kcal/mol/Angstrom

  @rq-b42b3b34
  Scenario: The exact CustomCompoundBondForce reproduces AWSEM-MD on small systems
    Given the reference cases with at most 26 residues
    When the compound method is evaluated on the Reference platform
    Then energies and forces match LAMMPS within 1e-6 and 1e-5

  @rq-64514065
  Scenario: Truncated forces reproduce AWSEM-MD exactly
    Given NumpyHelix with delta_helix_xi = 1e-6
    When forces are computed for every reference frame
    Then they match LAMMPS within 1e-9 kcal/mol/Angstrom

  @rq-633ec7c1
  Scenario: Exact backends agree with each other
    Given a perturbed myoglobin frame and the 1R69-example parameters
    When every exact backend computes energy and forces
    Then all energies agree within 1e-8 kcal/mol and all forces within 1e-8 kcal/mol/Angstrom

  @rq-b721354e
  Scenario: Forces are the negative gradient of the energy
    Given a perturbed structure and an exact backend
    When forces are compared to central finite differences of the energy with step 1e-5 Angstrom
    Then they agree within 1e-5 kcal/mol/Angstrom

  @rq-566a9537
  Scenario: The density-free baseline is exact only without density dependence
    Given gamma_w equal to gamma_p
    When frozen_density and the reference are evaluated
    Then energies and forces agree within 1e-8
    But with the default gammas the baseline energy of the 1CTA dimer differs from AWSEM-MD by more than 1 kcal/mol

  @rq-e56cc5d7
  Scenario: The frozen-density term reproduces the energy of its build structure
    Given helical_term_frozen_density built with the native positions (in nm, as an array or a Quantity)
    When the native frame is evaluated
    Then the energy matches the exact energy within 1e-8 kcal/mol
    And without positions it equals the exact energy with gamma_w = gamma_p

Feature: Topology rules

  @rq-08e20c7f
  Scenario: Pairs stay inside chains
    Given chains of 1, 2, 4, 5, 6 and 22 residues
    When pairs are listed with i_diff = 4
    Then chains with fewer than 5 residues have no pairs
    And no pair crosses a chain boundary
    And the chain of 5 residues has exactly the pair (first, last)

  @rq-05e4cc4b
  Scenario: The last residue of a chain can be a donor
    Given a single chain of 10 residues
    When pairs are listed with i_diff = 4
    Then the pairs are (0,4), (1,5), (2,6), (3,7), (4,8) and (5,9)

  @rq-6f821ff5
  Scenario: Density neighbours across chains are kept
    Given two chains whose boundary residues have consecutive indices
    When the density mask is built
    Then the boundary residues count each other
    But same-chain residues with |r - s| <= 1 do not

  @rq-76746c8b
  Scenario: Glycine uses its CA as density bead
    Given a sequence containing glycine
    When the topology is built
    Then the density bead of each glycine is its CA

  @rq-4c990796
  Scenario: Invalid topologies are rejected
    Given a non-glycine residue without CB, an unknown residue letter, a residue without CA, arrays of the wrong length, an index outside the system, or a chain split in two
    When HelixTopology is constructed
    Then ValueError is raised

  @rq-2f586ec6
  Scenario: Pairs with a missing oxygen are skipped
    Given a residue without O
    When pairs are listed
    Then pairs needing that O as acceptor or as O_{j-1} are absent

  @rq-3fb15d40
  Scenario: Topology from an OpenMM topology
    Given an OpenMM topology with CA, O and CB atoms, an IGL glycine and an IPR proline in two chains
    When HelixTopology.from_openmm is called
    Then sequence, chains and indices match
    And a residue name that cannot be mapped needs an explicit sequence

  @rq-8c38f81c
  Scenario: Topology from an OpenAWSEM system
    Given the OpenMMAWSEMSystem of OpenAWSEM's 1MBN test model
    When HelixTopology.from_openawsem is called
    Then glycine CB indices are -1, the sequence and particle count match
    And the indices equal those of HelixTopology.from_openmm on its topology with the sequence

  @rq-2261a61a
  Scenario: Coarse-grained structure from a PDB file
    Given an all-atom PDB with an alternate location, a glycine and a second model
    When CoarseGrainedProtein.from_pdb is called
    Then only the first model and first alternate location are read
    And glycine gets an HB bead at the ideal CB position
    And writing and reading the beads as PDB keeps coordinates to 1e-3 Angstrom

  @rq-e3b558af
  Scenario: Malformed PDB residues are rejected
    Given a PDB residue without O, or a non-glycine residue without CB, or an unknown residue name
    When CoarseGrainedProtein.from_pdb is called
    Then ValueError is raised

Feature: Parameters

  @rq-6861a980
  Scenario: Defaults are AWSEM-MD's parameter file
    When HelixParameters() is created
    Then it equals the parameters read from AWSEM-MD parameters/fix_backbone_coeff.data

  @rq-562be003
  Scenario: Reading fix_backbone_coeff.data
    Given a file with [ABC], [Epsilon], a disabled [Water]- section and [Helix]
    When from_fix_backbone_coeff is called
    Then every [Helix], [ABC] and [Epsilon] value is read
    And to_section followed by from_fix_backbone_coeff returns equal parameters

  @rq-dd2d4354
  Scenario: Missing or broken [Helix] sections are rejected
    Given a file whose [Helix] is disabled with "[Helix]-", or absent, or truncated
    When from_fix_backbone_coeff is called
    Then ValueError is raised

  @rq-528445ef
  Scenario: Invalid parameter values are rejected
    Given 19 h4prob values, i_diff = 0, a non-positive width, kappa or cutoff, or an unknown field
    When HelixParameters is created, or replace is called with them
    Then ValueError (TypeError for an unknown field) is raised

  @rq-65db3ee4
  Scenario: Proline acceptor propensity
    Given pro_acceptor_flag on with h4prob_pro_acceptor = 0.4416
    Then a proline acceptor has propensity 0.4416
    And a proline donor keeps h4prob[P]
    And with the flag off a proline acceptor has h4prob[P]

  @rq-172adc62
  Scenario: Energy scales with k_helix and epsilon
    Given a structure
    When k_helix is doubled, or epsilon is set to 3
    Then the energy and forces scale by 2, or 3

Feature: Physics edge cases

  @rq-68243a10
  Scenario: The O-N cutoff switches a pair off
    Given a single helical pair whose O-N distance is just below and just above the cutoff
    When the energy is computed
    Then the pair contributes below the cutoff and contributes exactly zero energy and force above it

  @rq-08aff94c
  Scenario: The density well range
    Given two beads closer than rmin_theta, and two beads farther than rmax_theta
    When densities are computed with theta_cutoff on
    Then those bead pairs contribute zero
    And with theta_cutoff off they contribute the smooth tail of theta

  @rq-1c10e976
  Scenario: Exposed and buried limits
    Given an isolated helix whose densities are far below the threshold (rho < 3, threshold 6)
    Then each pair energy equals -k P theta gamma_w within 1e-6
    And when the same helix is surrounded by many beads (rho >> threshold) each pair energy equals -k P theta gamma_p within 1e-6

  @rq-219ab293
  Scenario: Rotation invariance
    Given a structure and a random rotation about the coordinate origin
    When the energy and forces are computed for both
    Then the energies agree within 1e-9 and the forces are the rotated forces within 1e-8
    And the net torque about the origin is zero within 1e-8

  @rq-b01002d9
  Scenario: Net force and torque follow the [ABC] coefficients
    Given [ABC] coefficients that sum to one
    Then the energy is translation invariant, and the net force and the net torque about the centroid are zero within 1e-9
    But with AWSEM-MD's coefficients (sums 1.0000364 and 1.00007) the net force is not zero
    And it equals minus the derivative of the energy under a rigid translation within 1e-7

  @rq-0aff8e03
  Scenario: Systems without pairs
    Given a protein with only one-residue chains, or an empty set of pairs
    When each backend and OpenMM method computes energy and forces
    Then the energy is zero and all forces are zero

  @rq-f82a7cac
  Scenario: Units of the OpenMM forces
    Given an exact OpenMM method
    When the raw OpenMM state is read
    Then the energy in kJ/mol is 4.184 times the AWSEM energy and forces in kJ/mol/nm are 41.84 times the AWSEM forces

Feature: OpenMM builders

  @rq-05f54fdc
  Scenario: Exact terms in a real OpenAWSEM system
    Given the OpenMMAWSEMSystem of OpenAWSEM's 1MBN test model and the 12 frames of its test trajectory
    When each exact term (cvforce, python with numpy and numba, torchforce) is added in group 29 and every frame is evaluated on the Reference and CPU platforms
    Then the group energies match LAMMPS within 1e-5 kJ/mol and forces within 1e-3 kJ/mol/nm
    And the N, H and C atoms receive no force
    But helical_term_frozen_density and hydrogenBondTerms.helical_term miss the LAMMPS energy by more than 1 kcal/mol on some frame

  @rq-b3e6f1d3
  Scenario: Force groups
    Given force_group = 7
    When any builder runs
    Then every returned force is in group 7

  @rq-c2b250e3
  Scenario: Unknown methods are rejected
    When build_helix_forces is called with method "gbforce"
    Then ValueError is raised

  @rq-5b6d5200
  Scenario: CustomCVForce blocks respect the 32-variable limit
    Given myoglobin and max_cvs of 32, 12 and 3
    When the cvforce method is built
    Then no CustomCVForce has more than max_cvs collective variables
    And every pair appears in exactly one block
    And a max_cvs outside 3..32 raises ValueError

  @rq-30eb15e1
  Scenario: The compound method refuses large systems
    Given a system larger than max_residues
    When the compound method is built
    Then ValueError is raised

  @rq-baae387a
  Scenario: PythonForce can be serialized
    Given a python force with the numpy backend
    When it is serialized and deserialized with XmlSerializer
    Then the restored force gives the same energy

  @rq-f7c6b614
  Scenario: The torch module round-trips through TorchScript
    Given HelixModule scripted with torch.jit.script and saved to a file
    When the file is loaded and evaluated
    Then the energy equals the eager module's

  @rq-541732b3
  Scenario: OpenAWSEM terms scale by k_awsem and k_helical
    Given a duck-typed OpenAWSEM system with k_awsem = 2
    When density_dependent_helical_term is built with k_helical = None
    Then its energy in kJ/mol is 2*4.184 times the AWSEM energy
    And with k_helical = 2.092 kJ/mol, or 0.5 kcal/mol as a Quantity, it is scaled by 0.5/k_helix

  @rq-191d03cb
  Scenario: Every OpenAWSEM term returns one Force
    Given a duck-typed OpenAWSEM system and forceGroup = 5
    When helical_term_cvforce, helical_term_compound, helical_term_pythonforce (numpy, numba) and helical_term_torchforce are built, and density_dependent_helical_term with each exact method
    Then each returns one Force in group 5 whose energy and forces match NumPy within 1e-8

  @rq-db316cce
  Scenario: Periodic boxes are reported
    Given an OpenAWSEM system with periodic_box set
    When a helical term is built
    Then a warning mentioning periodic images is issued

  @rq-75e522c6
  Scenario: Combining forces
    Given 1, 2, 32, 33 and 70 forces in force group 3
    When combine_forces merges them
    Then the result is one Force in group 3 whose energy is the sum of theirs
    And an empty list raises ValueError

Feature: Differentiable parameters

  @rq-9988b853
  Scenario: Gradients with respect to the parameters
    Given HelixModule with learnable = True
    When the energy is differentiated with respect to k_helix, gamma_p, gamma_w and the propensities
    Then the gradients match central finite differences within 1e-6

Feature: Gold-standard bridge

  @rq-7d184230
  Scenario: LAMMPS is run when available
    Given an AWSEM-MD LAMMPS executable
    When devtools run_lammps evaluates a perturbed 1R69 frame
    Then the reference port matches its energy and forces within 1e-9

  @rq-21a3480b
  Scenario: A missing LAMMPS executable is reported
    Given AWSEMMD_LMP pointing to a file that does not exist and no executable on PATH
    When run_lammps is called
    Then FileNotFoundError is raised

Feature: Benchmark

  @rq-f5817d31
  Scenario: The benchmark runs on a small system
    When the benchmark is run for 20 residues and one repetition
    Then it reports a setup time and an evaluation time for every available exact backend
```
