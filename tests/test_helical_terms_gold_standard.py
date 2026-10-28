"""Agreement of every helical-term implementation with LAMMPS/AWSEM-MD (rqm/helical_terms.md)."""
import numpy as np
import openmm
import pytest

from openawsem.functionTerms.helicalTerms import (NumpyHelix, OpenMMEvaluator, build_helix_forces, combine_forces,
                                                  reference_energy_forces)
from tests._helical_data import CASES, EDGE, HELICAL, load_case

E_TOL, F_TOL = 1e-6, 1e-5          # exact gradient vs AWSEM-MD's truncated density forces
EXACT_TOL = 1e-9                   # reference port / truncated forces vs AWSEM-MD


def _assert_case(compute, case, e_tol=E_TOL, f_tol=F_TOL):
    for k, x in enumerate(case.positions):
        energy, forces = compute(x)
        assert energy == pytest.approx(case.energies[k], abs=e_tol), f"{case.name} frame {case.labels[k]}"
        err = np.abs(forces - case.forces[k]).max()
        assert err < f_tol, f"{case.name} frame {case.labels[k]}: max force error {err:.2e}"


@pytest.mark.parametrize('name', CASES)
def test_reference_port_reproduces_awsemmd(name):  # rq-a261fb7b
    case = load_case(name)
    _assert_case(lambda x: reference_energy_forces(x, case.topology, case.params), case, EXACT_TOL, EXACT_TOL)


def _numba(neighbors):
    def build(case):
        pytest.importorskip('numba')
        from openawsem.functionTerms.helicalTerms import NumbaHelix
        return NumbaHelix(case.topology, case.params, neighbors=neighbors).compute
    return build


def _torch(case):
    pytest.importorskip('torch')
    from openawsem.functionTerms.helicalTerms import make_helix_module, torch_energy_forces
    module = make_helix_module(case.topology, case.params)
    return lambda x: torch_energy_forces(x, case.topology, case.params, module)


BACKENDS = {
    'numpy-dense': lambda case: NumpyHelix(case.topology, case.params, neighbors='dense').compute,
    'numpy-kdtree': lambda case: NumpyHelix(case.topology, case.params, neighbors='kdtree').compute,
    'numba-dense': _numba('dense'),
    'numba-kdtree': _numba('kdtree'),
    'torch': _torch,
}


@pytest.mark.parametrize('backend', sorted(BACKENDS))
@pytest.mark.parametrize('name', CASES)
def test_exact_backends_reproduce_awsemmd(name, backend):  # rq-7c690172
    case = load_case(name)
    _assert_case(BACKENDS[backend](case), case)


@pytest.mark.parametrize('name', CASES)
def test_truncated_forces_reproduce_awsemmd(name):  # rq-64514065
    case = load_case(name)
    _assert_case(NumpyHelix(case.topology, case.params, delta_helix_xi=1e-6).compute, case, EXACT_TOL, EXACT_TOL)


OPENMM_METHODS = {
    'cvforce': ('cvforce', {}, False),
    'cvforce-combined': ('cvforce', {}, True),
    'python-numpy': ('python', {'backend': 'numpy'}, False),
    'python-numba': ('python', {'backend': 'numba'}, False),
    'python-torch': ('python', {'backend': 'torch'}, False),
    'torchforce': ('torchforce', {}, False),
}


def require(method, options):
    """Skip when the optional dependencies of a method are missing."""
    if method == 'torchforce' or options.get('backend') == 'torch':
        pytest.importorskip('torch')
    if method == 'torchforce':
        pytest.importorskip('openmmtorch')
    if options.get('backend') == 'numba':
        pytest.importorskip('numba')
    if method == 'python' and not hasattr(openmm, 'PythonForce'):
        pytest.skip('openmm.PythonForce needs OpenMM >= 8.3')


@pytest.mark.parametrize('platform', ['Reference', 'CPU'])
@pytest.mark.parametrize('label', sorted(OPENMM_METHODS))
@pytest.mark.parametrize('name', HELICAL + EDGE)
def test_openmm_methods_reproduce_awsemmd(name, label, platform):  # rq-c9509fd3
    method, options, combined = OPENMM_METHODS[label]
    require(method, options)
    case = load_case(name)
    forces = build_helix_forces(method, case.topology, case.params, **options)
    if combined:
        forces = [combine_forces(forces)]
    _assert_case(OpenMMEvaluator(case.protein.n_particles, forces, platform).compute, case)


@pytest.mark.parametrize('name', [n for n in CASES if len(load_case(n).protein.sequence) <= 26])
def test_compound_reproduces_awsemmd_on_small_systems(name):  # rq-b42b3b34
    case = load_case(name)
    evaluator = OpenMMEvaluator(case.protein.n_particles, build_helix_forces('compound', case.topology, case.params),
                                'Reference')
    _assert_case(evaluator.compute, case)
