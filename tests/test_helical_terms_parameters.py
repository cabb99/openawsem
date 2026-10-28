"""HelixParameters: defaults, AWSEM-MD file parsing and validation (rqm/helical_terms.md)."""
import dataclasses

import numpy as np
import pytest

from openawsem.functionTerms.helicalTerms import HelixParameters, NumpyHelix
from tests._helical_data import load_case

# [Helix] of AWSEM-MD parameters/fix_backbone_coeff.data (commit 295fefe), including the
# trailing note that AWSEM-MD ignores.
AWSEMMD_PARAMETER_FILE = """\
[ABC]
0.4831806 0.703282 -0.1864262
0.4436538 0.2352006 0.3211455
0.841 0.89296 -0.73389

[Water]-
1.0
5.0 7.0

[Helix]
1.5
2.0 -1.0
7.0 7.0
3.0
4
6.0
4.5 6.5
0.77 0.68 0.07 0.15 0.23 0.33 0.27 0.0 0.06 0.23 0.62 0.65 0.50 0.41 -3.0 0.35 0.11 0.45 0.17 0.14
0 -3.0
0.76   0.68
2.06   2.98

NOTE: helix_cutoff can be 7.0 instead fo 15.0

#[AMH-Go]
kamhgo
"""


def test_defaults_are_awsemmd_parameter_file(tmp_path):  # rq-6861a980
    path = tmp_path / 'fix_backbone_coeff.data'
    path.write_text(AWSEMMD_PARAMETER_FILE)
    assert HelixParameters.from_fix_backbone_coeff(path) == HelixParameters()
    assert HelixParameters.awsemmd_default() == HelixParameters()


def test_reading_fix_backbone_coeff(tmp_path):  # rq-562be003
    text = """[ABC]
0.5 0.6 -0.1
0.4436538 0.2352006 0.3211455
0.8 0.9 -0.7

[Epsilon]
2.5

[Water]-
0.75
[Helix]
0.5
1.9 -0.9
6.0 5.0
2.5
3
15.0
4.0 7.0
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
1 0.4416
0.70 0.60
2.1558 2.9862
"""
    path = tmp_path / 'coeff.data'
    path.write_text(text)
    p = HelixParameters.from_fix_backbone_coeff(path)
    assert (p.k_helix, p.gamma_p, p.gamma_w, p.kappa, p.kappa_sigma) == (0.5, 1.9, -0.9, 6.0, 5.0)
    assert (p.threshold, p.i_diff, p.cutoff, p.r_min, p.r_max) == (2.5, 3, 15.0, 4.0, 7.0)
    assert p.h4prob == tuple(float(v) for v in range(1, 21))
    assert p.pro_acceptor_flag is True and p.h4prob_pro_acceptor == 0.4416
    assert (p.sigma_HO, p.sigma_NO, p.r0_HO, p.r0_NO) == (0.70, 0.60, 2.1558, 2.9862)
    assert (p.an, p.bn, p.cn, p.ah, p.bh, p.ch) == (0.5, 0.6, -0.1, 0.8, 0.9, -0.7)
    assert p.epsilon == 2.5 and p.k_eff == pytest.approx(1.25)
    # Round trip through the AWSEM-MD text format.
    again = tmp_path / 'again.data'
    again.write_text(p.to_section())
    assert HelixParameters.from_fix_backbone_coeff(again) == p
    # Overrides win over the file.
    assert HelixParameters.from_fix_backbone_coeff(path, cutoff=6.0).cutoff == 6.0


@pytest.mark.parametrize('text', [
    AWSEMMD_PARAMETER_FILE.replace('[Helix]\n', '[Helix]-\n'),        # disabled
    '[ABC]\n0.48 0.70 -0.18\n0.44 0.23 0.32\n0.84 0.89 -0.73\n',      # absent
    '[Helix]\n1.5\n2.0 -1.0\n7.0 7.0\n3.0\n4\n6.0\n4.5 6.5\n0.77 0.68\n',  # truncated
], ids=['disabled', 'absent', 'truncated'])
def test_missing_or_broken_helix_section(tmp_path, text):  # rq-dd2d4354
    path = tmp_path / 'coeff.data'
    path.write_text(text)
    with pytest.raises(ValueError):
        HelixParameters.from_fix_backbone_coeff(path)


@pytest.mark.parametrize('kwargs', [
    dict(h4prob=[0.1] * 19), dict(i_diff=0), dict(sigma_HO=0.0), dict(sigma_NO=-1.0), dict(kappa=0.0),
    dict(kappa_sigma=-7.0), dict(cutoff=0.0), dict(unknown_field=1.0),
], ids=lambda kw: next(iter(kw)))
def test_invalid_parameter_values(kwargs):  # rq-528445ef
    with pytest.raises((ValueError, TypeError)):
        HelixParameters(**kwargs)
    with pytest.raises((ValueError, TypeError)):
        HelixParameters().replace(**kwargs)


def test_parameters_are_immutable():  # rq-68a3ed85
    with pytest.raises(dataclasses.FrozenInstanceError):
        HelixParameters().k_helix = 2.0
    p = HelixParameters(h4prob=[0.5] * 20)
    assert p.h4prob == (0.5,) * 20 and HelixParameters(**p.as_dict()) == p


def test_proline_acceptor_propensity():  # rq-65db3ee4
    on = HelixParameters(pro_acceptor_flag=True, h4prob_pro_acceptor=0.4416)
    off = HelixParameters(h4prob_pro_acceptor=0.4416)
    assert on.propensity('P', acceptor=True) == 0.4416
    assert on.propensity('P', acceptor=False) == -3.0
    assert on.propensity('A', acceptor=True) == 0.77
    assert off.propensity('P', acceptor=True) == -3.0
    with pytest.raises(ValueError):
        on.propensity('X')


def test_theta_range():  # rq-9cf0d859
    p = HelixParameters()
    pad = 8 * 2.302585 / 7.0
    assert p.theta_range == pytest.approx((4.5 - pad, 6.5 + pad))
    assert HelixParameters(theta_cutoff=False).theta_range == (0.0, np.inf)


@pytest.mark.parametrize('change,factor', [(dict(k_helix=3.0), 2.0), (dict(epsilon=3.0), 3.0)],
                         ids=['k_helix', 'epsilon'])
def test_energy_scales_with_k_and_epsilon(change, factor):  # rq-172adc62
    case = load_case('1mbn_default')
    x = case.positions[2]
    e0, f0 = NumpyHelix(case.topology, case.params).compute(x)
    e1, f1 = NumpyHelix(case.topology, case.params.replace(**change)).compute(x)
    assert e1 == pytest.approx(factor * e0, rel=1e-12)
    np.testing.assert_allclose(f1, factor * f0, rtol=1e-10, atol=1e-12)
