"""PyTorch helical term: TorchScript round trip and parameter gradients (rqm/helical_terms.md)."""
import numpy as np
import pytest

from openawsem.functionTerms.helicalTerms import numpy_energy_forces
from tests._helical_data import load_case

torch = pytest.importorskip('torch')
from openawsem.functionTerms.helicalTerms import make_helix_module as HelixModule  # noqa: E402


def test_torchscript_round_trip(tmp_path):  # rq-f7c6b614
    case = load_case('1mbn_pro_acceptor')
    module = HelixModule(case.topology, case.params, units='openmm')
    scripted = torch.jit.script(module)
    path = tmp_path / 'helix.pt'
    scripted.save(str(path))
    loaded = torch.jit.load(str(path))
    x = torch.tensor(case.positions[3] * 0.1)
    assert loaded(x).item() == pytest.approx(module(x).item(), abs=1e-12)
    assert module(x).item() == pytest.approx(4.184 * case.energies[3], abs=1e-6)
    # Single-precision positions (as some OpenMM platforms provide) are promoted internally.
    assert loaded(x.float()).item() == pytest.approx(module(x).item(), rel=1e-5)


def test_parameter_gradients():  # rq-9988b853
    case = load_case('glypro_pro_acceptor')
    x = torch.tensor(case.positions[2])
    module = HelixModule(case.topology, case.params, learnable=True)
    names = ['k_helix', 'gamma_p', 'gamma_w', 'h4']
    assert {n for n, _ in module.named_parameters()} == set(names)
    energy = module(x)
    assert energy.item() == pytest.approx(case.energies[2], abs=1e-6)
    grads = torch.autograd.grad(energy, [getattr(module, n) for n in names])
    h = 1e-6
    for name, grad in zip(names, grads):
        param = getattr(module, name)
        flat = param.detach().reshape(-1)
        for k in range(flat.numel()):
            with torch.no_grad():
                original = flat[k].item()
                param.view(-1)[k] = original + h
                e_plus = module(x).item()
                param.view(-1)[k] = original - h
                e_minus = module(x).item()
                param.view(-1)[k] = original
            assert grad.reshape(-1)[k].item() == pytest.approx((e_plus - e_minus) / (2 * h), abs=1e-6), (name, k)
    # The proline-acceptor entry (index 20) is used because the flag is on and prolines accept.
    assert grads[3][20].item() != 0.0


def test_positions_gradient_matches_numpy():  # rq-64c066a5
    case = load_case('fis_modified')
    x = case.positions[4]
    module = HelixModule(case.topology, case.params)
    xt = torch.tensor(x, requires_grad=True)
    energy = module(xt)
    energy.backward()
    e0, f0 = numpy_energy_forces(x, case.topology, case.params)
    assert energy.item() == pytest.approx(e0, abs=1e-10)
    assert np.abs(-xt.grad.numpy() - f0).max() < 1e-10
