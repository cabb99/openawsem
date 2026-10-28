"""Physics and edge cases of the helical term, checked across implementations (rqm/helical_terms.md)."""
import numpy as np
import pytest

from openawsem.functionTerms.helicalTerms import (HelixParameters, NumpyHelix, numpy_energy_forces,
                                                  reference_energy_forces)
from tests._helical_data import cg_protein, load_case


def exact_implementations(topology, params, openmm_methods=True):
    """(name, compute) for every exact implementation available here (OpenMM on Reference)."""
    impls = [
        ('reference', lambda x: reference_energy_forces(x, topology, params, delta_helix_xi=0.0)),
        ('numpy-dense', NumpyHelix(topology, params, neighbors='dense').compute),
    ]
    if params.theta_cutoff:
        impls.append(('numpy-kdtree', NumpyHelix(topology, params, neighbors='kdtree').compute))
    try:
        from openawsem.functionTerms.helicalTerms import NumbaHelix
        impls.append(('numba-dense', NumbaHelix(topology, params, neighbors='dense').compute))
        if params.theta_cutoff:
            impls.append(('numba-kdtree', NumbaHelix(topology, params, neighbors='kdtree').compute))
    except ImportError:
        pass
    try:
        from openawsem.functionTerms.helicalTerms import make_helix_module as HelixModule, torch_energy_forces
        module = HelixModule(topology, params)
        impls.append(('torch', lambda x: torch_energy_forces(x, topology, params, module)))
    except ImportError:
        pass
    if not openmm_methods:
        return impls
    try:
        import openmm
        from openawsem.functionTerms.helicalTerms import OpenMMEvaluator, build_helix_forces
    except ImportError:
        return impls
    methods = [('cvforce', {})]
    if hasattr(openmm, 'PythonForce'):
        methods.append(('python', {'backend': 'numpy'}))
    try:
        import openmmtorch  # noqa: F401
        methods.append(('torchforce', {}))
    except ImportError:
        pass
    if topology.n_residues <= 12:
        methods.append(('compound', {}))
    for method, options in methods:
        ev = OpenMMEvaluator(topology.n_particles, build_helix_forces(method, topology, params, **options))
        impls.append((f'openmm-{method}', ev.compute))
    return impls


def test_exact_backends_agree_with_each_other():  # rq-633ec7c1
    case = load_case('1mbn_1r69_example')
    x = case.positions[2]
    results = {name: compute(x) for name, compute in exact_implementations(case.topology, case.params)}
    e0, f0 = results['reference']
    assert len(results) >= 2
    for name, (e, f) in results.items():
        assert e == pytest.approx(e0, abs=1e-8), name
        assert np.abs(f - f0).max() < 1e-8, name


@pytest.mark.parametrize('case_name,frame', [('glypro_default', 2), ('1cta_default', 1), ('fis_modified', 3)])
def test_forces_are_negative_gradient(case_name, frame):  # rq-b721354e
    case = load_case(case_name)
    x = case.positions[frame]
    h = 1e-5
    rng = np.random.default_rng(0)
    atoms = rng.choice(len(x), size=min(len(x), 24), replace=False)
    for name, compute in exact_implementations(case.topology, case.params):
        _, forces = compute(x)
        for a in atoms:
            for c in range(3):
                xp, xm = x.copy(), x.copy()
                xp[a, c] += h
                xm[a, c] -= h
                fd = -(compute(xp)[0] - compute(xm)[0]) / (2 * h)
                assert fd == pytest.approx(forces[a, c], abs=1e-5), f"{name}: atom {a} axis {c}"


def test_nodensity_baseline():  # rq-566a9537
    pytest.importorskip('openmm')
    from openawsem.functionTerms.helicalTerms import OpenMMEvaluator, build_helix_forces
    case = load_case('1cta_default')
    flat = case.params.replace(**dict(gamma_w=case.params.gamma_p))
    ev = OpenMMEvaluator(case.protein.n_particles, build_helix_forces('frozen_density', case.topology, flat))
    for x in case.positions:
        e, f = ev.compute(x)
        e0, f0 = reference_energy_forces(x, case.topology, flat, delta_helix_xi=0.0)
        assert e == pytest.approx(e0, abs=1e-8)
        assert np.abs(f - f0).max() < 1e-8
    ev = OpenMMEvaluator(case.protein.n_particles, build_helix_forces('frozen_density', case.topology,
                                                                      case.params))
    assert abs(ev.compute(case.positions[0])[0] - case.energies[0]) > 1.0


def test_frozen_density_baseline():  # rq-e56cc5d7
    pytest.importorskip('openmm')
    from openawsem.functionTerms.helicalTerms import OpenMMEvaluator, build_helix_forces
    case = load_case('fis_default')
    native = case.positions[0]
    forces = build_helix_forces('frozen_density', case.topology, case.params, positions=native)
    e, _ = OpenMMEvaluator(case.protein.n_particles, forces).compute(native)
    assert e == pytest.approx(case.energies[0], abs=1e-8)
    with pytest.raises(ValueError):
        build_helix_forces('frozen_density', case.topology, case.params, positions=native, gamma=1.0)


def _five_residue_helix():
    case = load_case('glypro_default')
    x = case.positions[0]
    t = case.topology
    sl = slice(4, 9)  # 'LPKGP' -> acceptor L, donor P
    return cg_protein(['AAAAA'], x[t.ca][sl], x[t.o][sl], x[2::3][sl])


def _with_r_no(protein, params, r_no):
    """Move O_0 along the O-N direction so that |O_0 - N_4| = r_no."""
    x = protein.positions.copy()
    t = protein.topology
    n4 = params.an * x[t.ca[3]] + params.bn * x[t.ca[4]] + params.cn * x[t.o[3]]
    u = (x[t.o[0]] - n4) / np.linalg.norm(x[t.o[0]] - n4)
    x[t.o[0]] = n4 + r_no * u
    return x


def test_cutoff_switches_a_pair_off():  # rq-68243a10
    params = HelixParameters(cutoff=3.2)
    protein = _five_residue_helix()
    assert len(protein.topology.pairs(params)) == 1
    below, above = _with_r_no(protein, params, 3.2 - 1e-6), _with_r_no(protein, params, 3.2 + 1e-6)
    for name, compute in exact_implementations(protein.topology, params):
        e_below, f_below = compute(below)
        e_above, f_above = compute(above)
        assert abs(e_below) > 0.01 and np.abs(f_below).max() > 0.01, name
        assert e_above == 0.0, name
        assert np.abs(f_above).max() < 1e-12, name


def test_density_well_range():  # rq-08aff94c
    params = HelixParameters()
    rmin_theta, rmax_theta = params.theta_range
    seqs = ['A', 'A', 'A']
    for d in (rmin_theta - 0.05, rmax_theta + 0.05):
        cb = np.array([[0.0, 0, 0], [d, 0, 0], [50.0, 0, 0]])
        protein = cg_protein(seqs, cb + 1.0, cb - 1.0, cb)
        on = NumpyHelix(protein.topology, params).density(protein.positions)
        off_params = params.replace(**dict(theta_cutoff=False))
        off = NumpyHelix(protein.topology, off_params).density(protein.positions)
        _, _, terms = reference_energy_forces(protein.positions, protein.topology, params, return_terms=True)
        assert on.tolist() == [0.0, 0.0, 0.0] and terms['rho'].tolist() == [0.0, 0.0, 0.0]
        assert off[0] > 0 and off[0] == off[1]
        theta = 0.25 * (1 + np.tanh(7 * (d - 4.5))) * (1 + np.tanh(7 * (6.5 - d)))
        assert off[0] == pytest.approx(theta, rel=1e-6)
        try:
            import torch
            from openawsem.functionTerms.helicalTerms import make_helix_module as HelixModule
            x = torch.tensor(protein.positions)
            assert HelixModule(protein.topology, params).density(x).tolist() == [0.0, 0.0, 0.0]
            assert HelixModule(protein.topology, off_params).density(x)[0].item() == pytest.approx(off[0], rel=1e-12)
        except ImportError:
            pass


def _geometric_theta(x, t, p):
    n4 = p.an * x[t.ca[3]] + p.bn * x[t.ca[4]] + p.cn * x[t.o[3]]
    h4 = p.ah * x[t.ca[3]] + p.bh * x[t.ca[4]] + p.ch * x[t.o[3]]
    r_no, r_ho = np.linalg.norm(x[t.o[0]] - n4), np.linalg.norm(x[t.o[0]] - h4)
    return np.exp(-(r_no - p.r0_NO) ** 2 / (2 * p.sigma_NO ** 2) - (r_ho - p.r0_HO) ** 2 / (2 * p.sigma_HO ** 2))


def test_exposed_and_buried_limits():  # rq-1c10e976
    p = HelixParameters(cutoff=15.0)
    helix = _five_residue_helix()
    x = helix.positions
    t = helix.topology
    prefactor = -p.k_eff * (p.propensity('A', True) + p.propensity('A')) * _geometric_theta(x, t, p)
    # Exposed: an isolated five-residue helix (rho ~ 2) with the threshold raised to 6.
    exposed = p.replace(**dict(threshold=6.0))
    _, _, terms = reference_energy_forces(x, t, exposed, return_terms=True)
    assert terms['rho'].max() < 3
    for name, compute in exact_implementations(t, exposed, openmm_methods=False):
        assert compute(x)[0] == pytest.approx(prefactor * p.gamma_w, abs=1e-6), name
    # Buried: 24 single-residue chains with their CB 5.5 A from both CB_0 and CB_4.
    rng = np.random.default_rng(1)
    cb0, cb4 = x[t.cb[0]], x[t.cb[4]]
    shell = []
    for centre in (cb0, cb4):
        v = rng.normal(size=(12, 3))
        shell.append(centre + 5.5 * v / np.linalg.norm(v, axis=1, keepdims=True))
    shell = np.concatenate(shell)
    far = shell + 100.0  # CA/O of the shell residues, out of the way
    buried = cg_protein(['AAAAA'] + ['A'] * len(shell),
                                                np.concatenate([x[t.ca], far]), np.concatenate([x[t.o], far + 3]),
                                                np.concatenate([x[2::3], shell]))
    _, _, terms = reference_energy_forces(buried.positions, buried.topology, p, return_terms=True)
    assert terms['rho'][0] > 6 and terms['rho'][4] > 6
    for name, compute in exact_implementations(buried.topology, p, openmm_methods=False):
        assert compute(buried.positions)[0] == pytest.approx(prefactor * p.gamma_p, abs=1e-6), name


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_rotation_invariance():  # rq-219ab293
    case = load_case('1r69_1r69_example')
    x = case.positions[1]
    rot = _rotation(3)
    for name, compute in exact_implementations(case.topology, case.params, openmm_methods=False):
        e, f = compute(x)
        e_rot, f_rot = compute(x @ rot.T)
        assert e_rot == pytest.approx(e, abs=1e-9), name
        assert np.abs(f_rot - f @ rot.T).max() < 1e-8, name
        assert np.abs(np.cross(x, f).sum(axis=0)).max() < 1e-8, name


def test_net_force_and_torque():  # rq-b01002d9
    case = load_case('1cta_default')
    x = case.positions[2]
    p = case.params
    unit = p.replace(**dict(cn=1 - p.an - p.bn, ch=1 - p.ah - p.bh))
    shift = np.array([3.0, -2.0, 5.0])
    for name, compute in exact_implementations(case.topology, unit, openmm_methods=False):
        e, f = compute(x)
        assert compute(x + shift)[0] == pytest.approx(e, abs=1e-9), name
        assert np.abs(f.sum(axis=0)).max() < 1e-9, name
        assert np.abs(np.cross(x - x.mean(axis=0), f).sum(axis=0)).max() < 1e-9, name
    assert p.an + p.bn + p.cn == pytest.approx(1.0000364) and p.ah + p.bh + p.ch == pytest.approx(1.00007)
    h = 1e-4
    for name, compute in exact_implementations(case.topology, p, openmm_methods=False):
        _, f = compute(x)
        net = f.sum(axis=0)
        assert np.abs(net).max() > 1e-6, name
        for axis in range(3):
            u = np.zeros(3)
            u[axis] = h
            d_e = (compute(x + u)[0] - compute(x - u)[0]) / (2 * h)
            assert net[axis] == pytest.approx(-d_e, abs=1e-7), name


def test_systems_without_pairs():  # rq-0aff8e03
    case = load_case('single_residue_chains')
    assert len(case.topology.pairs(case.params)) == 0
    for name, compute in exact_implementations(case.topology, case.params):
        for x in case.positions:
            e, f = compute(x)
            assert e == 0.0 and np.abs(f).max() == 0.0, name
    try:
        from openawsem.functionTerms.helicalTerms import OpenMMEvaluator, build_helix_forces
        forces = build_helix_forces('frozen_density', case.topology, case.params)
        e, f = OpenMMEvaluator(case.protein.n_particles, forces).compute(case.positions[0])
        assert e == 0.0 and np.abs(f).max() == 0.0
    except ImportError:
        pass


def test_openmm_units():  # rq-f82a7cac
    openmm = pytest.importorskip('openmm')
    from openawsem.functionTerms.helicalTerms import OpenMMEvaluator, build_helix_forces
    case = load_case('1r69_default')
    x = case.positions[1]
    e0, f0 = numpy_energy_forces(x, case.topology, case.params)
    for method in ('cvforce', 'frozen_density'):
        params = case.params if method == 'cvforce' else case.params.replace(**dict(gamma_w=2.0))
        e0, f0 = numpy_energy_forces(x, case.topology, params)
        ev = OpenMMEvaluator(case.protein.n_particles, build_helix_forces(method, case.topology, params))
        ev.context.setPositions(x * 0.1)
        state = ev.context.getState(getEnergy=True, getForces=True)
        e_kj = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        f_kj = state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        assert e_kj == pytest.approx(4.184 * e0, abs=1e-8)
        assert np.abs(f_kj - 41.84 * f0).max() < 1e-7
