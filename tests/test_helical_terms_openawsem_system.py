"""Helical terms in a real OpenMMAWSEMSystem (OpenAWSEM's 1MBN test model and trajectory) vs AWSEM-MD."""
import json

import numpy as np
import openmm
import pytest

import openawsem
from openawsem.functionTerms import helicalTerms
from openawsem.functionTerms.helicalTerms import HelixParameters, HelixTopology
from tests._helical_data import DATA, REFERENCE_DIR

KJ_PER_KCAL, KJNM_PER_KCALA = 4.184, 41.84


@pytest.fixture(scope='module')
def system_and_reference():
    chain = openawsem.helperFunctions.myFunctions.getAllChains(DATA / '1mbn-crystal_structure.pdb')
    seq = openawsem.helperFunctions.myFunctions.read_fasta(DATA / '1mbn-crystal_structure.fasta')
    oa = openawsem.OpenMMAWSEMSystem(DATA / '1mbn-openmmawsem.pdb', chains=chain, k_awsem=1.0,
                                     xml_filename=openawsem.xml, seqFromPdb=seq, includeLigands=False)
    d = np.load(REFERENCE_DIR / 'openawsem' / '1mbn_movie.npz')
    params = HelixParameters(**json.loads(str(d['params'])))
    return oa, params, d['positions'], d['energies'], d['forces']


def _terms():
    terms = {'cvforce': (helicalTerms.helical_term_cvforce, {})}
    if hasattr(openmm, 'PythonForce'):
        terms['python-numpy'] = (helicalTerms.helical_term_pythonforce, {'backend': 'numpy'})
        try:
            import numba  # noqa: F401
            terms['python-numba'] = (helicalTerms.helical_term_pythonforce, {'backend': 'numba'})
        except ImportError:
            pass
    try:
        import openmmtorch  # noqa: F401
        terms['torchforce'] = (helicalTerms.helical_term_torchforce, {})
    except ImportError:
        pass
    return terms


def test_topology_of_an_openawsem_system(system_and_reference):  # rq-8c38f81c
    oa, *_ = system_and_reference
    t = HelixTopology.from_openawsem(oa)
    assert t.sequence == oa.seq and t.n_particles == oa.natoms
    assert np.all(t.cb[np.array(list(oa.seq)) == 'G'] == -1)
    same = HelixTopology.from_openmm(oa.pdb.topology, sequence=oa.seq)
    for name in ('chain_index', 'ca', 'o', 'cb', 'density_bead'):
        assert np.array_equal(getattr(t, name), getattr(same, name)), name


@pytest.mark.parametrize('platform', ['Reference', 'CPU'])
@pytest.mark.parametrize('label', sorted(_terms()))
def test_terms_in_an_openawsem_system(system_and_reference, label, platform):  # rq-05f54fdc
    oa, params, frames, energies, forces = system_and_reference
    make, options = _terms()[label]
    system = openmm.XmlSerializer.clone(oa.system)
    system.addForce(make(oa, k_helical=None, parameters=params, forceGroup=29, **options))
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName(platform))
    others = np.array([a.index for a in oa.pdb.topology.atoms() if a.name in ('N', 'H', 'C')])
    for k, x in enumerate(frames):
        context.setPositions(x * 0.1)
        state = context.getState(getEnergy=True, getForces=True, groups={29})
        e = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        f = state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        assert e == pytest.approx(KJ_PER_KCAL * energies[k], abs=1e-5), k
        assert np.abs(f - KJNM_PER_KCALA * forces[k]).max() < 1e-3, k
        assert np.all(f[others] == 0.0)


def test_density_free_terms_on_the_trajectory(system_and_reference):  # rq-05f54fdc
    oa, params, frames, energies, _ = system_and_reference
    for force in (helicalTerms.helical_term_frozen_density(oa, k_helical=None, parameters=params),
                  openawsem.functionTerms.hydrogenBondTerms.helical_term(oa)):
        system = openmm.System()
        for _ in range(oa.natoms):
            system.addParticle(1.0)
        system.addForce(force)
        context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName('Reference'))
        errors = []
        for x, e in zip(frames, energies):
            context.setPositions(x * 0.1)
            e_kj = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
            errors.append(abs(e_kj / KJ_PER_KCAL - e))
        assert max(errors) > 1.0
