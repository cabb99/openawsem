"""OpenMM builders of the helical term: options, limits and serialization (rqm/helical_terms.md)."""
import types

import numpy as np
import pytest

import openmm

from openawsem.functionTerms.helicalTerms import (BUILDERS, EXACT, MAX_CVS, OpenMMEvaluator, build_helix_forces,
                                                  combine_forces, density_dependent_helical_term,
                                                  helical_term_compound, helical_term_cvforce,
                                                  helical_term_frozen_density, helical_term_pythonforce,
                                                  helical_term_torchforce, numpy_energy_forces)
from tests._helical_data import load_case


def _available(method, options=None):
    options = options or {}
    if method == 'python' and not hasattr(openmm, 'PythonForce'):
        return False
    if method == 'torchforce' or options.get('backend') == 'torch':
        try:
            import torch  # noqa: F401
            if method == 'torchforce':
                import openmmtorch  # noqa: F401
        except ImportError:
            return False
    return True


@pytest.mark.parametrize('method', sorted(BUILDERS))
def test_force_groups(method):  # rq-b3e6f1d3
    if not _available(method):
        pytest.skip(f'{method} not available')
    case = load_case('glypro_default')
    forces = build_helix_forces(method, case.topology, case.params, force_group=7)
    assert forces and all(f.getForceGroup() == 7 for f in forces)
    assert EXACT == {'compound', 'cvforce', 'python', 'torchforce'}


def test_unknown_methods_are_rejected():  # rq-c2b250e3
    case = load_case('glypro_default')
    with pytest.raises(ValueError):
        build_helix_forces('gbforce', case.topology, case.params)
    if hasattr(openmm, 'PythonForce'):
        with pytest.raises(ValueError):
            build_helix_forces('python', case.topology, case.params, backend='fortran')


@pytest.mark.parametrize('max_cvs', [32, 12, 3])
def test_cvforce_blocks_respect_the_limit(max_cvs):  # rq-5b6d5200
    case = load_case('1mbn_default')
    pairs = case.topology.pairs(case.params)
    forces = build_helix_forces('cvforce', case.topology, case.params, max_cvs=max_cvs)
    cv_forces = [f for f in forces if isinstance(f, openmm.CustomCVForce)]
    seen = []
    for f in cv_forces:
        names = [f.getCollectiveVariableName(k) for k in range(f.getNumCollectiveVariables())]
        assert len(names) <= max_cvs <= MAX_CVS
        seen += [int(n[2:]) for n in names if n.startswith('hb')]
    assert sorted(seen) == list(range(len(pairs)))
    if max_cvs == 3:  # one pair per block: rho_i, rho_j and the hydrogen-bond factor
        assert len(cv_forces) == len(pairs)
        e, f = OpenMMEvaluator(case.protein.n_particles, forces).compute(case.positions[1])
        e0, f0 = numpy_energy_forces(case.positions[1], case.topology, case.params)
        assert e == pytest.approx(e0, abs=1e-8) and np.abs(f - f0).max() < 1e-8
    for bad in (2, 33):
        with pytest.raises(ValueError):
            build_helix_forces('cvforce', case.topology, case.params, max_cvs=bad)


def test_compound_refuses_large_systems():  # rq-30eb15e1
    case = load_case('1r69_default')
    with pytest.raises(ValueError, match='max_residues'):
        build_helix_forces('compound', case.topology, case.params)
    small = load_case('single_residue_chains')
    assert build_helix_forces('compound', small.topology, small.params, max_residues=12)
    with pytest.raises(ValueError):
        build_helix_forces('compound', small.topology, small.params, max_residues=11)


@pytest.mark.parametrize('backend', ['numpy', 'numba', 'torch'])
def test_python_force_serialization(backend):  # rq-baae387a
    if not hasattr(openmm, 'PythonForce'):
        pytest.skip('openmm.PythonForce needs OpenMM >= 8.3')
    if backend in ('numba', 'torch'):
        pytest.importorskip(backend)
    case = load_case('1cta_default')
    x = case.positions[1]
    (force,) = build_helix_forces('python', case.topology, case.params, backend=backend)
    restored = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(force))
    e, f = OpenMMEvaluator(case.protein.n_particles, [restored]).compute(x)
    e0, f0 = numpy_energy_forces(x, case.topology, case.params)
    assert e == pytest.approx(e0, abs=1e-10) and np.abs(f - f0).max() < 1e-10


def _duck_openawsem(case, k_awsem=1.0, periodic_box=None):
    t = case.topology
    starts = [int(np.where(t.chain_index == c)[0][0]) for c in np.unique(t.chain_index)]
    ends = [int(np.where(t.chain_index == c)[0][-1]) for c in np.unique(t.chain_index)]
    return types.SimpleNamespace(nres=t.n_residues, natoms=t.n_particles, seq=t.sequence, chain_starts=starts,
                                 chain_ends=ends, ca=t.ca.tolist(), o=t.o.tolist(),
                                 cb=(np.arange(t.n_residues) * 3 + 2).tolist(), k_awsem=k_awsem,
                                 periodic_box=periodic_box)


def test_openawsem_terms_scale_by_k_awsem_and_k_helical():  # rq-541732b3
    case = load_case('1cta_default')
    oa = _duck_openawsem(case, k_awsem=2.0)
    raw_kj = dict(energy_scale=1.0)
    force = density_dependent_helical_term(oa, k_helical=None, method='cvforce', parameters=case.params,
                                           forceGroup=29)
    assert isinstance(force, openmm.Force) and force.getForceGroup() == 29
    e_kj, _ = OpenMMEvaluator(case.protein.n_particles, [force], **raw_kj).compute(case.positions[0])
    assert e_kj == pytest.approx(2 * 4.184 * case.energies[0], abs=1e-6)
    # k_helical (kJ/mol or a Quantity) replaces parameters.k_helix.
    k = 0.5 * openmm.unit.kilocalorie_per_mole
    for k_helical in (0.5 * 4.184, k):
        force = helical_term_cvforce(oa, k_helical=k_helical, parameters=case.params)
        e_kj, _ = OpenMMEvaluator(case.protein.n_particles, [force], **raw_kj).compute(case.positions[0])
        assert e_kj == pytest.approx(2 * 4.184 * case.energies[0] * 0.5 / case.params.k_helix, abs=1e-6)


def test_every_openawsem_term_returns_one_force():  # rq-191d03cb
    case = load_case('glypro_default')
    oa = _duck_openawsem(case)
    x = case.positions[2]
    terms = {'cvforce': lambda: helical_term_cvforce(oa, k_helical=None, parameters=case.params, forceGroup=5),
             'compound': lambda: helical_term_compound(oa, k_helical=None, parameters=case.params, forceGroup=5)}
    backends = ['numpy']
    try:
        import numba  # noqa: F401
        backends.append('numba')
    except ImportError:
        pass
    if hasattr(openmm, 'PythonForce'):
        for backend in backends:
            terms[f'python-{backend}'] = (lambda b=backend: helical_term_pythonforce(
                oa, k_helical=None, parameters=case.params, forceGroup=5, backend=b))
    try:
        import openmmtorch  # noqa: F401
        terms['torchforce'] = lambda: helical_term_torchforce(oa, k_helical=None, parameters=case.params, forceGroup=5)
    except ImportError:
        pass
    e0, f0 = numpy_energy_forces(x, case.topology, case.params)
    for name, make in terms.items():
        force = make()
        assert isinstance(force, openmm.Force) and force.getForceGroup() == 5, name
        e, f = OpenMMEvaluator(case.protein.n_particles, [force]).compute(x)
        assert e == pytest.approx(e0, abs=1e-8) and np.abs(f - f0).max() < 1e-8, name
    available = {'cvforce': True, 'compound': True, 'python': 'python-numba' in terms, 'torchforce': 'torchforce' in terms}
    for method in sorted(EXACT):
        if not available[method]:
            continue
        force = density_dependent_helical_term(oa, k_helical=None, method=method, parameters=case.params)
        e, _ = OpenMMEvaluator(case.protein.n_particles, [force]).compute(x)
        assert e == pytest.approx(e0, abs=1e-8), method


def test_frozen_density_term():  # rq-e56cc5d7
    case = load_case('fis_default')
    oa = _duck_openawsem(case)
    native = case.positions[0]
    force = helical_term_frozen_density(oa, k_helical=None, parameters=case.params, positions=native * 0.1)
    assert OpenMMEvaluator(case.protein.n_particles, [force]).compute(native)[0] == pytest.approx(case.energies[0],
                                                                                                abs=1e-8)
    force = density_dependent_helical_term(oa, k_helical=None, method='frozen_density', parameters=case.params,
                                           positions=native * 0.1 * openmm.unit.nanometer)
    assert OpenMMEvaluator(case.protein.n_particles, [force]).compute(native)[0] == pytest.approx(case.energies[0],
                                                                                                abs=1e-8)
    buried = helical_term_frozen_density(oa, k_helical=None, parameters=case.params)
    flat = case.params.replace(gamma_w=case.params.gamma_p)
    e_buried = OpenMMEvaluator(case.protein.n_particles, [buried]).compute(native)[0]
    assert e_buried == pytest.approx(numpy_energy_forces(native, case.topology, flat)[0], abs=1e-8)


def test_periodic_box_warning():  # rq-db316cce
    case = load_case('glypro_default')
    with pytest.warns(UserWarning, match='periodic'):
        helical_term_cvforce(_duck_openawsem(case, periodic_box=[10, 10, 10]), parameters=case.params)


@pytest.mark.parametrize('n_forces', [1, 2, 32, 33, 70])
def test_combine_forces(n_forces):  # rq-75e522c6
    case = load_case('glypro_default')
    x = case.positions[1]
    pairs = case.topology.pairs(case.params)
    singles = []
    for k in range(n_forces):  # one-bond copies of the first pair: energies simply add up
        f = openmm.CustomBondForce('r')
        f.addBond(int(pairs.o_i[0]), int(pairs.ca_j[0]))
        f.setForceGroup(3)
        singles.append(f)
    combined = combine_forces(singles, name='Sum')
    assert combined.getName() == 'Sum' and combined.getForceGroup() == 3
    e, _ = OpenMMEvaluator(case.protein.n_particles, [combined], energy_scale=1.0).compute(x)
    d = np.linalg.norm(x[pairs.o_i[0]] - x[pairs.ca_j[0]]) * 0.1
    assert e == pytest.approx(n_forces * d, rel=1e-12)
    with pytest.raises(ValueError):
        combine_forces([])
