import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from pmrf.frequency import Frequency
from pmrf.utils import unwrap

from scipy.constants import c, mu_0, epsilon_0

from pmrf.models import (
    PhaseLine, 
    RLGCLine, 
    LowLossLine,
    PhysicalLine, 
    DatasheetLine, 
    NominalEffectiveLineFormulation,
    FirstOrderEffectiveLineFormulation,
    GeneralizedLine,
    RLTerm,
    RCTerm,
    LCTerm,
    CoaxialLine, 
    MicrostripLine, 
    StriplineLine,
    FloatingLine
)
from pmrf.materials import (
    BulkConductor,
    ConductorProperties,
    ConstantDielectric,
    DielectricProperties,
    DjordjevicSarkarDielectric,
    RoughConductor,
)
from pmrf.materials.surface_impedance import HalfSpaceSurfaceImpedance, TescheTubeSurfaceImpedance
from pmrf.models.components.lines.coaxial import SchelkunoffCoaxialFormulation, TescheCoaxialFormulation
from pmrf.models.components.lines.microstrip import KirschningJansenMicrostripDispersion

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=10, unit='GHz')

def test_phase_line(basic_freq):
    """Test ideal, lossless phase line limits."""
    # Create a 50-ohm line, 90 degrees at 5 GHz
    line = PhaseLine(z0=50.0, theta=90.0, f0=5.0e9)
    s = line.s(basic_freq)
    
    assert s.shape == (10, 2, 2)
    
    # S11 should be 0 for a line matched to the system z0 (which defaults to 50)
    assert jnp.allclose(s[:, 0, 0], 0.0, atol=1e-5)
    
    # Magnitude of transmission should be perfectly 1.0 (lossless)
    assert jnp.allclose(jnp.abs(s[:, 1, 0]), 1.0, atol=1e-5)

def test_floating_line(basic_freq):
    """Test the wrapper that converts a 2-port to a 4-port floating line."""
    base_line = PhaseLine(z0=50.0, theta=90.0, f0=5e9)
    float_line = FloatingLine(floating=base_line)
    
    s = float_line.s(basic_freq)
    
    # Should automatically scale up to a 4-port parameter matrix
    assert s.shape == (10, 4, 4)
    assert not jnp.any(jnp.isnan(s))

def test_constant_rlgc_line(basic_freq):
    """Test a basic transmission line evaluated from per-unit-length components."""
    # Lossless 50 ohm line (L=250nH, C=100pF -> Zc = sqrt(L/C) = 50)
    line = RLGCLine(R=0.0, G=0.0, L=250e-9, C=100e-12, length=0.1)
    s = line.s(basic_freq)
    
    assert s.shape == (10, 2, 2)
    assert jnp.allclose(s[:, 0, 0], 0.0, atol=1e-5)
    assert jnp.allclose(jnp.abs(s[:, 1, 0]), 1.0, atol=1e-5)

@pytest.mark.parametrize("line_model", [
    LowLossLine(length=0.1),
    PhysicalLine(length=0.1),
    PhysicalLine(length=0.1, formulation=FirstOrderEffectiveLineFormulation()),
    DatasheetLine(length=0.1),
    DatasheetLine(length=0.1, formulation=FirstOrderEffectiveLineFormulation()),
    GeneralizedLine(length=0.1),
    CoaxialLine(length=0.1),
    MicrostripLine(length=0.1)
])
def test_physical_line_execution(line_model, basic_freq):
    """
    Ensure all physical models compile and evaluate correctly without 
    divide-by-zero or shape mismatch errors.
    """
    s = line_model.s(basic_freq)
    assert s.shape == (10, 2, 2)
    assert not jnp.any(jnp.isnan(s))


def test_physical_line_phase_delay(basic_freq):
    """Test that PhysicalLine calculates the correct phase delay based on permittivity."""
    # 15 mm line in a dielectric with Er = 4.0
    length = 0.015
    ep_eff = 4.0
    line = PhysicalLine(zn=50.0, length=length, ep_eff=ep_eff, A=0.0, tand=0.0)
    
    s = line.s(basic_freq)
    
    # Expected phase shift: beta * L = w * sqrt(ep_eff) * L / c
    # Since S21 = exp(-j * beta * L), angle(S21) = -beta * L
    expected_phase = -basic_freq.w * jnp.sqrt(ep_eff) * length / c
    actual_phase = jnp.unwrap(jnp.angle(s[:, 1, 0]))
    
    assert jnp.allclose(actual_phase, expected_phase, atol=1e-4)
    # Lossless matched line -> S11 should be exactly 0
    assert jnp.allclose(s[:, 0, 0], 0.0, atol=1e-5)

def test_datasheet_line_attenuation(basic_freq):
    """DatasheetLine attenuates by k1 dB per 100 m at 1 MHz, rising as sqrt(f)."""
    # Pure skin-effect loss (k1 in dB/(100 m sqrt(MHz))), no dielectric loss (k2).
    k1, length = 2.0, 10.0
    line = DatasheetLine(zn=50.0, vf=1.0, k1=k1, k2=0.0, length=length)

    s = line.s(basic_freq)

    expected_db = -k1 * jnp.sqrt(basic_freq.f / 1e6) / 100 * length
    actual_db = 20 * jnp.log10(jnp.abs(s[:, 1, 0]))

    # The nominal line departs from the separable attenuation at higher order in
    # R'/(wL') (at most 7e-3 here), and by the small mismatch of Zc to 50 ohm.
    # Observed 7e-5 dB of 20 dB.
    assert jnp.allclose(actual_db, expected_db, rtol=0, atol=2e-4)


@pytest.mark.parametrize("line_type", [DatasheetLine, PhysicalLine])
@pytest.mark.parametrize("length", [1.0, 10.0])
def test_empirical_line_matches_half_space_coax(line_type, length):
    """The first-order line is a half-space coax with a Djordjevic-Sarkar dielectric.

    `C'_n ep_DS / ep_eff` is the coax admittance and `(1 + j) R'_n` its half-space
    conductor impedance, so the match is exact.
    """
    d_in, d_out = 0.92e-3, 3.0e-3
    ep_eff, tand, sigma = 2.1, 1e-4, 5.8e7
    f_low, f_high, f_ref = 1e3, 1e12, 1e9
    a, b = d_in / 2, d_out / 2
    zn = np.sqrt(mu_0 / epsilon_0) * np.log(b / a) / (2 * np.pi * np.sqrt(ep_eff))
    k1 = (2000 / np.log(10)) * np.sqrt(np.pi * 1e6 * mu_0 / sigma) * (1 / a + 1 / b) / (4 * np.pi * zn)
    k2 = (2000 / np.log(10)) * np.pi * 1e6 * np.sqrt(ep_eff) * tand / c
    formulation = FirstOrderEffectiveLineFormulation(f_low=f_low, f_high=f_high)

    with jax.enable_x64():
        freq = Frequency(start=10, stop=500, npoints=491, unit='MHz')
        if line_type is DatasheetLine:
            line = DatasheetLine(
                length=length, zn=zn, vf=1 / np.sqrt(ep_eff), k1=k1, k2=k2,
                f_ref=f_ref, formulation=formulation,
            )
        else:
            A = k1 * np.sqrt(f_ref / 1e6) / 100
            line = PhysicalLine(
                length=length, zn=zn, ep_eff=ep_eff, A=A, tand=tand,
                f_ref=f_ref, formulation=formulation,
            )
        reference = CoaxialLine(
            d_in=d_in, d_out=d_out, length=length,
            dielectric=DjordjevicSarkarDielectric(
                ep_r=ep_eff, tand=tand, f_low=f_low, f_high=f_high, f_ref=f_ref,
            ),
            conductor=BulkConductor(sigma=sigma),
            formulation=SchelkunoffCoaxialFormulation(
                inner_impedance=HalfSpaceSurfaceImpedance(),
                shield_impedance=HalfSpaceSurfaceImpedance(),
            ),
        )

        difference = np.abs(np.asarray(line.s(freq) - reference.s(freq)))

    assert np.max(difference) < 1e-8


# -----------------------------------------------------------------------------
# Effective-value lines (#322). The suggested configuration of the issue, on a
# 1 MHz-1 GHz axis whose last point is f_ref.
# -----------------------------------------------------------------------------

_EFFECTIVE = dict(zn=50.0, ep_eff=2.1, A=0.05, tand=2e-4, f_ref=1e9)
_DS_BOUNDS = dict(f_low=1e3, f_high=1e12)


def _effective_freq():
    return Frequency(start=1, stop=1000, npoints=200, unit='MHz')


def _closed_form_gamma(w, zn, ep_eff, A, tand, f_ref):
    alpha_c = np.log(10) / 20 * A * np.sqrt(w / (2 * np.pi * f_ref))
    alpha_d = w * np.sqrt(ep_eff) * tand / (2 * c)
    return alpha_c + alpha_d + 1j * w * np.sqrt(ep_eff) / c


def test_low_loss_line_matches_closed_form():
    with jax.enable_x64():
        freq = _effective_freq()
        line = LowLossLine(length=2.0, **_EFFECTIVE)
        zc, gammaL = line.zc_and_gammaL(freq)
        expected = _closed_form_gamma(np.asarray(freq.w), **_EFFECTIVE)

        assert np.all(np.imag(np.asarray(zc)) == 0)
        assert np.all(np.real(np.asarray(zc)) == _EFFECTIVE["zn"])
        # Machine precision: a few ulps of the largest term.
        np.testing.assert_allclose(np.asarray(gammaL) / 2.0, expected, rtol=1e-14)


def test_effective_lines_match_skrf_defined_a_ep_tand_z0():
    """scikit-rf's `DefinedAEpTandZ0` (frequency-invariant) is the same effective-value line.

    Its gamma is always the low-loss one. Its Zc is `z0` when that is an array, which
    makes it LowLossLine, and the nominal RLGC `sqrt(Z'/Y')` when `z0` is a scalar, which
    pairs a nominal Zc with a low-loss gamma: a mixed model that ParamRF does not keep.
    So Zc for a scalar `z0` is checked against the nominal PhysicalLine alone.
    """
    from skrf.media import DefinedAEpTandZ0

    zn, ep_eff, A, tand, f_ref = (_EFFECTIVE[k] for k in ("zn", "ep_eff", "A", "tand", "f_ref"))
    with jax.enable_x64():
        freq = _effective_freq()
        n = freq.npoints
        skrf_low_loss = DefinedAEpTandZ0(
            freq.to_skrf(), A=A, f_A=f_ref, ep_r=ep_eff, tanD=tand, z0=np.full(n, zn),
        )
        skrf_nominal_zc = DefinedAEpTandZ0(
            freq.to_skrf(), A=A, f_A=f_ref, ep_r=ep_eff, tanD=tand, z0=zn,
        ).z0_characteristic
        zc, gammaL = LowLossLine(length=1.0, **_EFFECTIVE).zc_and_gammaL(freq)
        nominal_zc = PhysicalLine(length=1.0, **_EFFECTIVE).zc(freq)

    # Both evaluate the same closed forms in float64: a few ulps apart.
    np.testing.assert_allclose(np.asarray(gammaL), skrf_low_loss.gamma, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(zc), skrf_low_loss.z0_characteristic, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(nominal_zc), skrf_nominal_zc, rtol=1e-14)


def test_nominal_line_reduces_to_low_loss_quadratically():
    """The nominal gamma departs from the low-loss one at second order in the losses."""
    def relative_difference(scale):
        values = dict(_EFFECTIVE, A=_EFFECTIVE["A"] * scale, tand=_EFFECTIVE["tand"] * scale)
        nominal = PhysicalLine(length=1.0, **values).gammaL(freq)
        low_loss = LowLossLine(length=1.0, **values).gammaL(freq)
        return np.max(np.abs(np.asarray(nominal - low_loss)) / np.abs(np.asarray(low_loss)))

    with jax.enable_x64():
        freq = _effective_freq()
        coarse, fine = relative_difference(1.0), relative_difference(0.1)

    assert coarse < 1e-4
    # Scaling the losses by 1/10 scales a second-order difference by 1/100. The
    # higher-order terms move the observed ratio, 99.995, by 5e-5.
    np.testing.assert_allclose(coarse / fine, 100.0, rtol=1e-3)


@pytest.mark.parametrize("make", [
    lambda: PhysicalLine(length=1.0, **_EFFECTIVE),
    lambda: DatasheetLine(length=1.0, zn=50.0, vf=0.69, k1=0.2, k2=0.01),
], ids=["PhysicalLine", "DatasheetLine"])
def test_first_order_admittance_is_djordjevic_sarkar(make):
    with jax.enable_x64():
        freq = _effective_freq()
        line = make()
        values = unwrap(line).effective_values()
        first_order = eqx.tree_at(
            lambda m: m.formulation, line, FirstOrderEffectiveLineFormulation(**_DS_BOUNDS)
        )
        Y = np.asarray(first_order.immittance(freq).Y)
        Y_nominal = np.asarray(line.immittance(freq).Y)

        dielectric = DjordjevicSarkarDielectric(
            ep_r=values.ep_eff, tand=values.tand, f_ref=values.f_ref, **_DS_BOUNDS
        )
        C = np.sqrt(values.ep_eff) / (values.zn * c)
        expected = 1j * np.asarray(freq.w) * C * np.asarray(dielectric.properties(freq).ep_r) / values.ep_eff

        np.testing.assert_allclose(Y, expected, rtol=1e-13)
        # The last point is f_ref, where chi_DS = 1.
        np.testing.assert_allclose(Y[-1], Y_nominal[-1], rtol=1e-13)
        assert not np.allclose(Y[:-1], Y_nominal[:-1], rtol=1e-6)


@pytest.mark.parametrize("formulation", [
    NominalEffectiveLineFormulation(), FirstOrderEffectiveLineFormulation(**_DS_BOUNDS),
], ids=["nominal", "first-order"])
def test_datasheet_line_maps_onto_physical_line(formulation):
    zn, vf, k1, k2, f_ref = 50.0, 0.69, 0.2, 0.01, 1e9
    with jax.enable_x64():
        freq = _effective_freq()
        datasheet = DatasheetLine(
            length=1.0, zn=zn, vf=vf, k1=k1, k2=k2, f_ref=f_ref, formulation=formulation,
        )
        physical = PhysicalLine(
            length=1.0, zn=zn, ep_eff=1 / vf**2, A=k1 * np.sqrt(f_ref / 1e6) / 100,
            tand=k2 * vf * c * np.log(10) / (2 * np.pi * 1e9), f_ref=f_ref,
            formulation=formulation,
        )
        expected = physical.immittance(freq)
        actual = datasheet.immittance(freq)

        np.testing.assert_allclose(np.asarray(actual.Z), np.asarray(expected.Z), rtol=1e-13)
        np.testing.assert_allclose(np.asarray(actual.Y), np.asarray(expected.Y), rtol=1e-13)


@pytest.mark.parametrize("line_type", [PhysicalLine, DatasheetLine])
def test_effective_lines_default_to_nominal_at_1_ghz(line_type):
    line = line_type(length=1.0)
    assert type(line.formulation) is NominalEffectiveLineFormulation
    assert line.f_ref.value == 1e9


# -----------------------------------------------------------------------------
# GeneralizedLine (#324): the first-order line plus RL, RC and LC terms. The same
# effective values on the same 1 MHz-1 GHz axis.
# -----------------------------------------------------------------------------

def _first_order(**kwargs):
    return PhysicalLine(
        length=1.0, formulation=FirstOrderEffectiveLineFormulation(**_DS_BOUNDS), **kwargs,
    )


def _generalized(**kwargs):
    return GeneralizedLine(length=1.0, **_DS_BOUNDS, **{**_EFFECTIVE, **kwargs})


def _nominal_rlc(freq, zn=_EFFECTIVE["zn"], ep_eff=_EFFECTIVE["ep_eff"], A=_EFFECTIVE["A"],
                 f_ref=_EFFECTIVE["f_ref"]):
    """The nominal R'_n(w), L'_n and C'_n of EffectiveValues, in closed form."""
    w = np.asarray(freq.w)
    R = 2 * zn * np.log(10) / 20 * A * np.sqrt(w / (2 * np.pi * f_ref))
    return w, R, zn * np.sqrt(ep_eff) / c, np.sqrt(ep_eff) / (zn * c)


def test_generalized_line_reduces_to_first_order():
    with jax.enable_x64():
        freq = _effective_freq()
        expected = _first_order(**_EFFECTIVE).immittance(freq)
        actual = _generalized().immittance(freq)

    # Built from the same operations: equal to the last ulp.
    np.testing.assert_allclose(np.asarray(actual.Z), np.asarray(expected.Z), rtol=1e-15, atol=0)
    np.testing.assert_allclose(np.asarray(actual.Y), np.asarray(expected.Y), rtol=1e-15, atol=0)


def test_generalized_line_onset_error_falls_as_root_f_s():
    """With a tiny f_s the onset is (1 + j) R'_n up to (2/pi) sqrt(f_s / f).

    arctan(z) = pi/2 - 1/z + O(z^-3), and (1 + j) / sqrt(j) = sqrt(2), so the leading
    error is real: Re Z' carries all of it, with no j w L'_n to cancel against.
    """
    f_s = 1e-6
    with jax.enable_x64():
        freq = _effective_freq()
        reference = np.real(np.asarray(_first_order(**_EFFECTIVE).immittance(freq).Z))
        onset = np.real(np.asarray(_generalized(f_s=f_s).immittance(freq).Z))
    _, R, _, _ = _nominal_rlc(freq)

    relative_error = (reference - onset) / np.abs((1 + 1j) * R)
    # The next term is relative O(f_s / f) <= 1e-12; the difference loses ~1e-10.
    np.testing.assert_allclose(relative_error, 2 / np.pi * np.sqrt(f_s / np.asarray(freq.f)), rtol=1e-8)


def test_generalized_line_onset_matches_quadrature():
    """The arctan term is the closed form of the truncated RL distribution."""
    from scipy.integrate import quad

    f_s = 1e6
    zn, A, f_ref = _EFFECTIVE["zn"], _EFFECTIVE["A"], _EFFECTIVE["f_ref"]
    freq = Frequency.from_f([1e5, 1e6, 3e6, 1e8, 1e9], unit='Hz')
    with jax.enable_x64():
        Z = np.asarray(_generalized(f_s=f_s).immittance(freq).Z)
    w, _, L, _ = _nominal_rlc(freq)
    w_s, w_ref = 2 * np.pi * f_s, 2 * np.pi * f_ref
    R_ref = 2 * zn * np.log(10) / 20 * A

    for wk, Zk in zip(w, Z):
        # x = u^2 turns the integrand into 2 j w / (u^2 + j w) over [sqrt(w_s), inf).
        u0 = np.sqrt(w_s)
        re = quad(lambda u: 2 * wk**2 / (u**4 + wk**2), u0, np.inf, epsabs=0, epsrel=1e-13)[0]
        im = quad(lambda u: 2 * wk * u**2 / (u**4 + wk**2), u0, np.inf, epsabs=0, epsrel=1e-13)[0]
        expected = np.sqrt(2) * R_ref / (np.pi * np.sqrt(w_ref)) * (re + 1j * im)
        # Subtracting j w L'_n (up to 1500 ohm/m) from Z' leaves ~1e-13 ohm/m of rounding.
        np.testing.assert_allclose(Zk - 1j * wk * L, expected, rtol=1e-10)


def test_generalized_line_rc_term():
    tand_i, f_i = 1e-3, 10e6
    with jax.enable_x64():
        freq = _effective_freq()
        base = _generalized().immittance(freq)
        line = _generalized(dielectric_terms=[(tand_i, f_i)]).immittance(freq)
    w, _, _, C = _nominal_rlc(freq)

    expected = 1j * w * C * 2 * tand_i / (1 + 1j * w / (2 * np.pi * f_i))
    # The term is ~1e-3 of Y', so subtracting the base leaves ~1e-13 relative rounding.
    np.testing.assert_allclose(np.asarray(line.Y - base.Y), expected, rtol=1e-10)


def test_generalized_line_rc_term_peak_and_static_capacitance():
    """On a lossless base the Debye pole is the whole loss: a peak tand_i at f_i.

    With strength 2 ep_eff tand_i, eps'' peaks at ep_eff tand_i and eps' there is
    ep_eff (1 + tand_i), so tan(delta) = tand_i / (1 + tand_i): about tand_i.
    """
    tand_i, f_i = 1e-3, 10e6
    with jax.enable_x64():
        freq = Frequency.from_f(np.logspace(0, 9, 1001), unit='Hz')
        freq_peak = Frequency.from_f([1.0, f_i], unit='Hz')
        line = _generalized(tand=0.0, dielectric_terms=[(tand_i, f_i)])
        Y = np.asarray(line.immittance(freq).Y)
        Y_peak = np.asarray(line.immittance(freq_peak).Y)
    _, _, _, C = _nominal_rlc(freq)

    def tand(Y):
        return np.real(Y) / np.imag(Y)

    np.testing.assert_allclose(tand(Y_peak[1]), tand_i / (1 + tand_i), rtol=1e-12)
    assert np.max(tand(Y)) <= tand(Y_peak[1])
    # At 1 Hz the pole is static to 1e-14: C' is 2 tand_i C'_n above C'_n.
    C_static = np.imag(Y_peak[0]) / (2 * np.pi)
    np.testing.assert_allclose(C_static, C * (1 + 2 * tand_i), rtol=1e-12)


@pytest.mark.parametrize("family", ["inductive_dispersion", "capacitive_dispersion"])
def test_generalized_line_lc_term(family):
    a, f_i = 1e-2, 5e9
    with jax.enable_x64():
        freq = _effective_freq()
        base = _generalized().immittance(freq)
        line = _generalized(**{family: [(a, f_i)]}).immittance(freq)
    w, _, L, C = _nominal_rlc(freq)
    w_i = 2 * np.pi * f_i

    resonance = a * w_i**2 / (w_i**2 - w**2)
    # The term is ~1e-2 of Z' or Y', so subtracting the base leaves ~1e-14 relative rounding.
    if family == "inductive_dispersion":
        np.testing.assert_allclose(np.asarray(line.Z - base.Z), 1j * w * L * resonance, rtol=1e-10)
        np.testing.assert_array_equal(np.asarray(line.Y), np.asarray(base.Y))
    else:
        np.testing.assert_allclose(np.asarray(line.Y - base.Y), 1j * w * C * resonance, rtol=1e-10)
        np.testing.assert_array_equal(np.asarray(line.Z), np.asarray(base.Z))


@pytest.mark.parametrize("family", ["inductive_dispersion", "capacitive_dispersion"])
def test_generalized_line_undamped_lc_term_in_band_raises(family):
    line = _generalized(**{family: [(1e-2, 500e6)]})
    with pytest.raises(Exception, match="must resonate outside the evaluated band"):
        line.s(_effective_freq())


def test_generalized_line_rl_term_is_parallel_rl():
    A_i, f_i = 0.02, 1e6
    with jax.enable_x64():
        freq = _effective_freq()
        base = _generalized().immittance(freq)
        line = _generalized(conductor_terms=[(A_i, f_i)]).immittance(freq)
    w = np.asarray(freq.w)
    R_i = 2 * _EFFECTIVE["zn"] * np.log(10) / 20 * A_i
    L_i = R_i / (2 * np.pi * f_i)

    # dZ is down to ~1e-4 ohm/m against j w L'_n up to 1500 ohm/m: ~1e-12 relative rounding.
    dZ = np.asarray(line.Z - base.Z)
    np.testing.assert_allclose(dZ, 1 / (1 / R_i + 1 / (1j * w * L_i)), rtol=1e-10)
    # Re dZ = R_i / (1 + (f_i / f)^2): 1e-6 below R_i at 1 GHz.
    np.testing.assert_allclose(np.real(dZ[-1]), R_i, rtol=2e-6)


def test_generalized_line_terms_coerce_from_tuples():
    line = GeneralizedLine(
        length=1.0,
        conductor_terms=[(0.01, 1e6)],
        dielectric_terms=[(1e-4, 1e7)],
        inductive_dispersion=[(1e-3, 5e9)],
        capacitive_dispersion=[(1e-3, 5e9, 0.1), LCTerm(a=1e-3, f=6e9)],
    )
    assert line.conductor_terms == (RLTerm(A=0.01, f=1e6),)
    assert line.dielectric_terms == (RCTerm(tand=1e-4, f=1e7),)
    assert line.inductive_dispersion == (LCTerm(a=1e-3, f=5e9),)
    assert line.capacitive_dispersion == (LCTerm(a=1e-3, f=5e9, zeta=0.1), LCTerm(a=1e-3, f=6e9))


def test_generalized_line_default_is_first_order_physical_line():
    with jax.enable_x64():
        freq = _effective_freq()
        line = GeneralizedLine(length=1.0)
        expected = PhysicalLine(length=1.0, formulation=FirstOrderEffectiveLineFormulation())
        np.testing.assert_array_equal(np.asarray(line.s(freq)), np.asarray(expected.s(freq)))
    # Always causal: there is no formulation to choose.
    assert not hasattr(line, "formulation")
    assert line.f_s is None


def test_generalized_line_is_passive():
    """Every term family populated, the undamped poles above the axis."""
    with jax.enable_x64():
        freq = Frequency.from_f(np.logspace(0, 11, 2001), unit='Hz')
        line = _generalized(
            R_dc=0.01,
            f_s=1e5,
            conductor_terms=[(0.02, 1e6), (0.01, 1e9)],
            dielectric_terms=[(1e-3, 1e7), (5e-4, 3e9)],
            inductive_dispersion=[(1e-2, 300e9), (5e-3, 2e9, 0.2)],
            capacitive_dispersion=[(1e-2, 400e9), (5e-3, 5e9, 0.1)],
        )
        immittance = line.immittance(freq)

    assert np.all(np.real(np.asarray(immittance.Z)) >= 0)
    assert np.all(np.real(np.asarray(immittance.Y)) >= 0)


def test_coaxial_line_impedance(basic_freq):
    """Verify CoaxialLine matches the analytical Zc formula for a lossless coax."""
    # RG-58 dimensions roughly: ep_r=2.25, d_in=0.9mm, d_out=2.95mm
    line = CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=ConstantDielectric(ep_r=2.25, tand=0.0),
        conductor=BulkConductor(sigma=jnp.inf),
        length=1.0,
    )
    
    # Extract Zc directly from the internal method
    zc, _ = line.zc_and_gammaL(basic_freq)
    
    # Analytical Zc = (1 / 2*pi) * sqrt(mu / eps) * ln(b/a)
    # (Note: 1/(2*pi) * sqrt(mu_0/eps_0) is approx 59.958 ohms)
    expected_zc = (1 / (2 * jnp.pi)) * jnp.sqrt(mu_0 / (epsilon_0 * 2.25)) * jnp.log(2.95 / 0.9)
    
    # Real part of Zc should match the theoretical lossless impedance
    assert jnp.allclose(jnp.real(zc), expected_zc, rtol=1e-3)


def test_coaxial_static_conductivity_has_analytic_dc_conductance():
    freq = Frequency.from_f(jnp.array([0.0, 1.0, 1e3]))
    sigma = 0.01
    d_in, d_out = 0.9e-3, 2.95e-3
    line = CoaxialLine(
        d_in=d_in,
        d_out=d_out,
        dielectric=ConstantDielectric(ep_r=2.25, sigma=sigma),
        conductor=BulkConductor(sigma=jnp.inf),
        length=1.0,
    )

    expected = 2 * jnp.pi * sigma / jnp.log(d_out / d_in)
    assert jnp.allclose(line.immittance(freq).G, expected)


def test_coaxial_rejects_reversed_diameters():
    line = CoaxialLine(d_in=2e-3, d_out=1e-3, length=0.1)
    with pytest.raises(Exception, match="d_out must exceed d_in"):
        line.immittance(Frequency.from_f(jnp.array([1e9])))

def test_coaxial_permeability_comes_from_the_dielectric(basic_freq):
    """A magnetic filling is a property of the material, not of the geometry.

    Zc = sqrt(mu/eps) * ln(b/a) / 2pi, so mu_r = 4 raises it by exactly 2.
    """
    def coax(dielectric):
        return CoaxialLine(
            d_in=0.9e-3,
            d_out=2.95e-3,
            dielectric=dielectric,
            conductor=BulkConductor(sigma=jnp.inf),
            length=1.0,
        )

    plain = coax(ConstantDielectric(ep_r=2.25, tand=0.0))
    magnetic = coax(ConstantDielectric(ep_r=2.25, tand=0.0, mu_r=4.0))

    zc_plain, _ = plain.zc_and_gammaL(basic_freq)
    zc_magnetic, _ = magnetic.zc_and_gammaL(basic_freq)

    assert jnp.allclose(jnp.real(zc_magnetic), 2.0 * jnp.real(zc_plain), rtol=1e-6)


def test_coaxial_magnetic_loss_enters_the_series_resistance(basic_freq):
    """A complex mu_r is not restricted at the interface; it becomes loss.

    With a lossless conductor and a lossless dielectric the only remaining loss
    channel is magnetic: Im(mu_r) < 0 adds w*mu''*ln(b/a)/2pi to Re(Z).
    """
    class LossyMagnetic(ConstantDielectric):
        def properties(self, freq):
            properties = super().properties(freq)
            return eqx.tree_at(
                lambda p: p.mu_r,
                properties,
                (4.0 - 0.5j) * jnp.ones(freq.npoints, dtype=complex),
            )

    line = CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=LossyMagnetic(ep_r=2.25, tand=0.0),
        conductor=BulkConductor(sigma=jnp.inf),
        length=1.0,
    )
    immittance = line.immittance(basic_freq)

    expected_R = (
        basic_freq.w * 0.5 * mu_0 * jnp.log(2.95 / 0.9) / (2 * jnp.pi)
    )
    assert jnp.allclose(immittance.R, expected_R, rtol=1e-6)
    # A passive magnetic loss must not show up as a negative resistance.
    assert jnp.all(immittance.R > 0)


def test_microstrip_line_default_construction_has_conductor_loss(basic_freq):
    """A default-constructed line has nonzero, sigma-sensitive attenuation.

    Regression for the bug where Wheeler's conductor-loss correction was
    guarded on `substrate.t is not None`, and `t` defaults to `None`: a
    default-constructed line (dispersion=KirschningJansenMicrostripDispersion,
    t=None) had `sigma` completely inert.
    """
    def alpha(sigma):
        line = MicrostripLine(
            dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
            conductor=BulkConductor(sigma=sigma),
            length=0.1,
        )
        _, gamma_length = line.zc_and_gammaL(basic_freq)
        return jnp.real(gamma_length / line.length)

    alpha_lossless = alpha(jnp.inf)
    alpha_lossy = alpha(1 / 1.68e-8)

    assert jnp.all(alpha_lossless == 0.0)
    assert jnp.all(alpha_lossy > alpha_lossless)


def test_microstrip_line_impedance(basic_freq):
    """Verify MicrostripLine evaluates Wheeler's formula correctly."""
    # Standard 50-ohm trace on 1.6mm FR4 (Er=4.3) is roughly 3.0mm wide
    line = MicrostripLine(
        w=3.0e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
        conductor=BulkConductor(sigma=jnp.inf),
        dispersion=None,
        length=0.1,
    )
    
    zc, _ = line.zc_and_gammaL(basic_freq)
    
    # The Wheeler approximation for this geometry should yield an impedance near 50 ohms
    zc_real = jnp.real(zc)
    assert jnp.all(zc_real > 48.0)
    assert jnp.all(zc_real < 52.0)

@pytest.mark.parametrize("dispersion", [None, KirschningJansenMicrostripDispersion()])
def test_microstrip_line_ep_eff_and_w_eff_match_immittance_pipeline(basic_freq, dispersion):
    """`ep_eff`/`w_eff` must report exactly what `immittance` uses internally.

    Regression for #80: these were previously reconstructed by hand in the
    scikit-rf validation matrix rather than exposed on the model, so a
    refactor of `immittance` could silently drift from the recomputed value.
    Recomputes the quasi-static (+ optional dispersion) pipeline independently
    here, rather than calling the model's own helper, so the test does not
    just check the method against itself.
    """
    w, h, ep_r = 3.0e-3, 1.6e-3, 4.3
    line = MicrostripLine(
        w=w,
        h=h,
        dielectric=ConstantDielectric(ep_r=ep_r, tand=0.01),
        conductor=BulkConductor(sigma=1 / 1.68e-8),
        dispersion=dispersion,
        length=0.1,
    )

    dielectric = line.substrate.dielectric.properties(basic_freq)
    quasi_static = line.formulation.quasi_static(w=w, h=h, t=None, ep_r=dielectric.ep_r)
    if dispersion is None:
        expected_ep_eff, expected_w_eff = quasi_static.ep_eff, quasi_static.w_eff
    else:
        expected_ep_eff, _ = dispersion.disperse(
            basic_freq, ep_eff_0=quasi_static.ep_eff, zc_0=quasi_static.zc,
            ep_r=dielectric.ep_r, w_eff=quasi_static.w_eff, h=h,
        )
        expected_w_eff = quasi_static.w_eff

    assert jnp.allclose(line.ep_eff(basic_freq), expected_ep_eff)
    assert jnp.allclose(line.w_eff(basic_freq), expected_w_eff)
    assert jnp.iscomplexobj(line.ep_eff(basic_freq))
    assert jnp.any(jnp.imag(line.ep_eff(basic_freq)) != 0.0)


def test_microstrip_finite_thickness_has_dc_resistance_floor():
    """#84: a finite-t line's R settles at rho/(W*t) at dc instead of 0.

    The sheet model alone gives R -> 0 as f -> 0, wrong for a trace of known
    thickness. A finite t gets a floor; t=None gets none, since it asserts
    skin effect in operation at every frequency including dc, leaving no dc
    regime for a floor to describe (matches ADS, which floors nothing).
    """
    from pmrf.models import HammerstadJensenMicrostripFormulation

    w, h, t, rho = 3.0e-3, 1.6e-3, 35e-6, 1.68e-8
    dc = Frequency.from_f(jnp.array([0.0]))

    floored = MicrostripLine(
        w=w, h=h, t=t,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
        conductor=BulkConductor(sigma=1 / rho),
        formulation=HammerstadJensenMicrostripFormulation(),
        length=0.1,
    )
    r_dc_expected = rho / (w * t)
    assert jnp.allclose(floored.immittance(dc).R, r_dc_expected)

    unfloored = MicrostripLine(
        w=w, h=h, t=None,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
        conductor=BulkConductor(sigma=1 / rho),
        length=0.1,
    )
    assert jnp.allclose(unfloored.immittance(dc).R, 0.0)


def test_material_coercion():
    """Scalars and tuples coerce into the corresponding material modules."""
    line = CoaxialLine(dielectric=(2.25, 0.001), conductor=5.8e7, length=0.1)

    assert isinstance(line.dielectric, ConstantDielectric)
    assert isinstance(line.conductor, BulkConductor)
    assert jnp.allclose(line.dielectric.ep_r.value, 2.25)


def test_coaxial_formulation_takes_plain_arrays(basic_freq):
    """A coaxial formulation is callable without ParamRF objects."""
    npoints = basic_freq.npoints
    result = TescheCoaxialFormulation().immittance(
        basic_freq,
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=DielectricProperties(
            np.full(npoints, 2.25 - 0.00225j), np.ones(npoints), np.zeros(npoints)
        ),
        conductor=ConductorProperties(
            np.full(npoints, 0.01 + 0.01j),
            np.full(npoints, 1 / 1.72e-8),
            np.ones(npoints),
        ),
    )
    assert result.Z.shape == (npoints,)
    assert result.Y.shape == (npoints,)


def test_coaxial_defaults_reuse_inner_material_and_infinite_wall(basic_freq):
    """Unspecified shield options preserve the simple coaxial construction."""
    line = CoaxialLine(length=0.1)
    explicit = CoaxialLine(
        length=0.1, outer_conductor=line.conductor, shield_thickness=None
    )
    assert jnp.allclose(line.immittance(basic_freq).Z, explicit.immittance(basic_freq).Z)


def test_coaxial_supports_dissimilar_finite_shield(basic_freq):
    """The shield can be a separate metal and a finite wall changes loss."""
    inner = BulkConductor(sigma=5.8e7)
    outer = BulkConductor(sigma=3.5e7)
    infinite = CoaxialLine(
        conductor=inner, outer_conductor=outer, length=0.1
    )
    finite = CoaxialLine(
        conductor=inner, outer_conductor=outer, shield_thickness=12.5e-6,
        length=0.1,
    )
    assert jnp.all(finite.immittance(basic_freq).R > 0)
    assert not jnp.allclose(
        finite.immittance(basic_freq).R, infinite.immittance(basic_freq).R
    )


def test_coaxial_internal_impedance_tube():
    """Tube arithmetic follows Tesche's equation (13).

    Kept here rather than folded into the wideband scikit-rf matrix in
    ``test_lines_skrf_matrix.py``: it pins one formula on its own, so a failure
    points straight at the internal impedance instead of at whichever line
    happens to use it.
    """
    skrf = pytest.importorskip("skrf")
    from skrf.media import Coaxial

    freq = Frequency.from_f(jnp.array([1e9]))
    sigma, mu = jnp.array([1e7]), jnp.ones(1)
    zs = jnp.sqrt(1j * freq.w * mu_0 / sigma)
    radius, thickness = 1e-3, 0.2e-3
    reference = Coaxial(
        freq.to_skrf(), Dint=2 * radius, Dout=4 * radius,
        sigma=float(sigma[0]), tout=thickness, model="tesche",
    )
    expected = reference._conductor_impedance(radius, thickness, None)
    conductor = ConductorProperties(zs, sigma, mu)
    zs_sq = TescheTubeSurfaceImpedance().impedance(freq.w, conductor, a=radius, t=thickness)
    actual = zs_sq / (2 * jnp.pi * radius)
    assert jnp.allclose(actual, expected)


def test_rough_conductor_is_passed_to_coaxial_formulation():
    freq = Frequency.from_f(jnp.array([1e6, 1e9]))
    smooth = CoaxialLine(conductor=BulkConductor(1 / 1.68e-8), length=0.1)
    rough = CoaxialLine(
        conductor=RoughConductor(1 / 1.68e-8, roughness=1e-6), length=0.1
    )
    assert jnp.all(rough.immittance(freq).R > smooth.immittance(freq).R)


def test_immittance_rlgc_roundtrip(basic_freq):
    """R, L, G, C are exact derived views on Z and Y."""
    line = RLGCLine(R=0.5, G=1e-4, L=250e-9, C=100e-12, length=0.1)
    imm = line.immittance(basic_freq)

    assert jnp.allclose(imm.R, 0.5)
    assert jnp.allclose(imm.L, 250e-9)
    assert jnp.allclose(imm.G, 1e-4)
    assert jnp.allclose(imm.C, 100e-12)


def test_immittance_rlgc_at_dc():
    """L and C stay finite when the axis includes DC."""
    freq = Frequency(start=0.0, stop=1.0, npoints=3, unit='GHz')
    line = RLGCLine(R=0.5, G=1e-4, L=250e-9, C=100e-12, length=0.1)
    imm = line.immittance(freq)

    assert jnp.all(jnp.isfinite(imm.L))
    assert jnp.all(jnp.isfinite(imm.C))
    assert jnp.allclose(imm.L, 250e-9)
    assert jnp.allclose(imm.C, 100e-12)


def test_immittance_inversion_rejects_non_passive_result():
    """Exact (Zc, gamma) inversion surfaces non-physical empirical output."""
    from pmrf.models import ImmittanceResult

    with pytest.raises(eqx.EquinoxRuntimeError, match=r"Re\(Z\) < 0"):
        ImmittanceResult.from_zc_gamma(
            zc=jnp.array([50.0]), gamma=jnp.array([-1.0 + 1j]), omega=jnp.array([1.0])
        )


def test_dispersive_dielectric_is_grid_independent():
    """
    A dispersive material is a function of frequency alone, so the same line
    evaluated on two different grids agrees at the frequencies they share.
    """
    line = CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=DjordjevicSarkarDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.68e-8),
        length=0.1,
    )

    coarse = Frequency(start=1.0, stop=5.0, npoints=5, unit='GHz')
    fine = Frequency(start=1.0, stop=9.0, npoints=9, unit='GHz')

    s_coarse = line.s(coarse)
    s_fine = line.s(fine)

    # The coarse grid is 1, 2, 3, 4, 5 GHz; the fine grid's first five points.
    assert jnp.allclose(s_coarse, s_fine[:5], atol=1e-12)


def test_microstrip_formulation_takes_plain_arrays(basic_freq):
    """A formulation is pure numerics, callable with no ParamRF objects."""
    from pmrf.models import WheelerMicrostripFormulation

    npoints = basic_freq.npoints
    ep_r = np.full(npoints, 4.3 - 0.086j)
    zs = np.full(npoints, 0.01 + 0.01j)

    result = WheelerMicrostripFormulation().quasi_static(
        w=3.0e-3, h=1.6e-3, t=None, ep_r=ep_r
    )

    assert result.ep_eff.shape == (npoints,)
    assert jnp.all(jnp.real(result.zc) > 40.0)
    assert jnp.allclose(result.w_eff, 3.0e-3)


def test_dispersive_microstrip_routes_through_planar_quasi_static_to_immittance():
    """#83: the dispersion path reports true RLGC Zc, not modal K-J Zc.

    Inverting the dispersed modal ``(zc, gamma)`` through ``from_zc_gamma``
    made ``Zc = sqrt(Z/Y)`` tautologically reproduce Kirschning-Jansen's own
    modal Zc, so the conductor never genuinely entered it. The dispersion
    path must instead build a fresh
    :class:`~pmrf.models.components.lines.planar.PlanarQuasiStaticResult`
    at the dispersed ``(ep_eff, zc)`` and route it through
    :meth:`~pmrf.models.components.lines.planar.PlanarQuasiStaticResult.to_immittance`,
    exactly like the quasi-static path, so ``line.immittance`` matches that
    construction bit for bit.
    """
    from pmrf.models import HammerstadJensenMicrostripFormulation, KirschningJansenMicrostripDispersion
    from pmrf.models.components.lines.microstrip import MicrostripCrossSection
    from pmrf.models.components.lines.planar import PlanarQuasiStaticResult

    freq = Frequency(start=1.0, stop=50.0, npoints=51, unit="GHz")
    formulation = HammerstadJensenMicrostripFormulation()
    dispersion = KirschningJansenMicrostripDispersion()
    line = MicrostripLine(
        w=0.2e-3,
        h=0.5e-3,
        t=18e-6,
        dielectric=ConstantDielectric(ep_r=10.0, tand=0.05),
        conductor=BulkConductor(sigma=1e6),
        formulation=formulation,
        dispersion=dispersion,
        length=0.1,
    )

    dielectric = line.substrate.dielectric.properties(freq)
    conductor = line.substrate.conductor.properties(freq)
    quasi_static = formulation.quasi_static(
        w=line.w, h=line.substrate.h, t=line.substrate.t, ep_r=dielectric.ep_r
    )
    ep_eff, zc = dispersion.disperse(
        freq,
        ep_eff_0=quasi_static.ep_eff,
        zc_0=quasi_static.zc,
        ep_r=dielectric.ep_r,
        w_eff=quasi_static.w_eff,
        h=line.substrate.h,
    )
    expected = PlanarQuasiStaticResult(
        ep_eff=ep_eff,
        zc=zc,
        w_eff=quasi_static.w_eff,
        shunt_conductance_factor=quasi_static.shunt_conductance_factor,
    ).to_immittance(
        freq, dielectric, conductor,
        current_distribution=line.current_distribution,
        cross_section=MicrostripCrossSection(
            w=line.w, h=line.substrate.h, t=line.substrate.t
        ),
    )

    actual = line.immittance(freq)
    assert jnp.array_equal(actual.Z, expected.Z)
    assert jnp.array_equal(actual.Y, expected.Y)

    # The dispersed zc no longer equals sqrt(Z/Y) tautologically: the
    # conductor genuinely enters the exact RLGC ratio, so the two differ.
    exact_zc = jnp.sqrt(actual.Z / actual.Y)
    assert not jnp.allclose(exact_zc, zc, rtol=1e-6, atol=1e-6)


def test_microstrip_without_dispersion_preserves_quasi_static_immittance():
    """Disabling stage three reproduces the pre-dispersion pipeline exactly."""
    freq = Frequency(start=1.0, stop=20.0, npoints=21, unit="GHz")
    line = MicrostripLine(
        w=3e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        dispersion=None,
        length=0.1,
    )

    from pmrf.models.components.lines.microstrip import MicrostripCrossSection

    ep_r = line.substrate.dielectric.properties(freq).ep_r
    zs = line.substrate.conductor.properties(freq).zs
    quasi_static = line.formulation.quasi_static(
        w=line.w, h=line.substrate.h, t=line.substrate.t, ep_r=ep_r
    )

    actual = line.immittance(freq)
    expected = quasi_static.to_immittance(
        freq,
        line.substrate.dielectric.properties(freq),
        line.substrate.conductor.properties(freq),
        current_distribution=line.current_distribution,
        cross_section=MicrostripCrossSection(
            w=line.w, h=line.substrate.h, t=line.substrate.t
        ),
    )
    assert jnp.array_equal(actual.Z, expected.Z)
    assert jnp.array_equal(actual.Y, expected.Y)


def test_microstrip_dispersion_is_a_pure_dispersion_toggle():
    """Turning dispersion off must not change *which* conductor loss applies.

    Both the quasi-static path (``PlanarQuasiStaticResult.to_immittance``) and
    the dispersion path charge Wheeler's incremental-inductance rule, so
    disabling dispersion with an identity dispersion formulation (one that
    hands the quasi-static ep_eff/zc straight through, unchanged) reproduces
    the quasi-static attenuation to the accuracy of the quasi-static path's
    own low-loss linearisation -- not to the ~9% gap that opened when the two
    paths charged different conductor-loss forms (issue #82).
    """
    from pmrf.models import HammerstadJensenMicrostripFormulation
    from pmrf.models.components.lines.microstrip import AbstractMicrostripDispersion

    class IdentityDispersion(AbstractMicrostripDispersion):
        """Hands the quasi-static ep_eff/zc through unchanged."""

        def disperse(self, freq, *, ep_eff_0, zc_0, ep_r, w_eff, h):
            return ep_eff_0, zc_0

    freq = Frequency(start=1.0, stop=20.0, npoints=21, unit="GHz")

    def alpha(dispersion):
        line = MicrostripLine(
            w=3e-3,
            h=1.6e-3,
            t=35e-6,
            dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
            conductor=BulkConductor(sigma=1 / 1.72e-8),
            formulation=HammerstadJensenMicrostripFormulation(),
            dispersion=dispersion,
            length=0.1,
        )
        _, gamma_length = line.zc_and_gammaL(freq)
        return jnp.real(gamma_length / line.length)

    alpha_quasi_static = alpha(None)
    alpha_identity_dispersion = alpha(IdentityDispersion())

    assert jnp.allclose(alpha_quasi_static, alpha_identity_dispersion, rtol=1e-2)


def test_microstrip_defaults_to_hammerstad_jensen():
    """#117: the thickness-aware formulation is the microstrip default.

    Hammerstad-Jensen quotes better than 0.2% on effective permittivity and
    0.01-0.03% on impedance, and the default Kirschning-Jansen dispersion was
    fitted against its effective width.
    """
    from pmrf.models import HammerstadJensenMicrostripFormulation

    assert isinstance(
        MicrostripLine(length=0.1).formulation, HammerstadJensenMicrostripFormulation
    )


def test_default_microstrip_finite_thickness_has_dc_resistance_floor():
    """#117: thickness reaches conductor loss on a default-constructed line.

    The issue's worked number: 35 um copper under a 1.55 mm trace gives
    1/(sigma*W*t) = 0.31781 ohm/m, with no formulation passed by the user.
    """
    w, t, sigma = 1.55e-3, 35e-6, 5.8e7
    line = MicrostripLine(
        w=w, h=0.8e-3, t=t, conductor=BulkConductor(sigma=sigma), length=0.1
    )
    dc = Frequency.from_f(jnp.array([0.0]))

    assert jnp.allclose(line.immittance(dc).R, 1 / (sigma * w * t), rtol=1e-9)


def test_wheeler_formulation_accepts_and_ignores_thickness():
    """#117: a zero-thickness-derived formulation declines t silently.

    Wheeler 1977 is derived for a zero-thickness strip, so t enters neither
    ep_eff nor zc; rejecting the input would stop the *loss* model from
    seeing a thickness it can use, so the formulation ignores it instead.
    """
    from pmrf.models import WheelerMicrostripFormulation

    ep_r = np.full(4, 4.3 - 0.086j)
    kwargs = dict(w=3.0e-3, h=1.6e-3, ep_r=ep_r)

    thin = WheelerMicrostripFormulation().quasi_static(t=None, **kwargs)
    thick = WheelerMicrostripFormulation().quasi_static(t=35e-6, **kwargs)

    assert jnp.allclose(thin.ep_eff, thick.ep_eff)
    assert jnp.allclose(thin.zc, thick.zc)
    assert jnp.allclose(thin.w_eff, thick.w_eff)


def test_wheeler_microstrip_line_with_thickness_still_gets_dc_floor():
    """#117: the loss floor is available whatever quasi-static choice is made."""
    from pmrf.models import WheelerMicrostripFormulation

    w, t, sigma = 1.55e-3, 35e-6, 5.8e7
    line = MicrostripLine(
        w=w, h=0.8e-3, t=t, conductor=BulkConductor(sigma=sigma),
        formulation=WheelerMicrostripFormulation(), length=0.1,
    )
    dc = Frequency.from_f(jnp.array([0.0]))

    assert jnp.allclose(line.immittance(dc).R, 1 / (sigma * w * t), rtol=1e-9)


def test_microstrip_defaults_to_kirschning_jansen():
    """Modal dispersion is the accuracy-oriented microstrip default."""
    from pmrf.models import KirschningJansenMicrostripDispersion

    assert isinstance(MicrostripLine(length=0.1).dispersion, KirschningJansenMicrostripDispersion)


def test_hammerstad_jensen_finite_thickness_matches_skrf():
    """The thickness-aware quasi-static formulation agrees with scikit-rf.

    Kept as a local unit test for the same reason as
    ``test_coaxial_internal_impedance_tube``: it exercises the quasi-static
    solution alone, with no dispersion, loss or conversion on top of it. The
    wideband end-to-end comparisons live in ``test_lines_skrf_matrix.py``.
    """
    from skrf.media import MLine
    from pmrf.models import HammerstadJensenMicrostripFormulation

    freq = Frequency(start=1.0, stop=10.0, npoints=10, unit="GHz")
    formulation = HammerstadJensenMicrostripFormulation()
    ep_r = jnp.full(freq.npoints, 4.3)
    result = formulation.quasi_static(w=1e-3, h=1e-3, t=35e-6, ep_r=ep_r)
    media = MLine(
        freq.to_skrf(),
        w=1e-3,
        h=1e-3,
        t=35e-6,
        ep_r=4.3,
        tand=0.0,
        rho=1.68e-8,
        rough=0.0,
        model="hammerstadjensen",
        disp="none",
        diel="frequencyinvariant",
    )

    assert jnp.allclose(result.zc, media.z0_characteristic, rtol=1e-12)
    assert jnp.allclose(result.ep_eff, media.ep_reff, rtol=1e-12)
    assert jnp.allclose(result.w_eff, media.w_eff, rtol=1e-12)


def test_microstrip_near_air_has_finite_conductance_and_gradient():
    """The old filling-factor singularity cannot return at εr approaching one."""
    freq = Frequency(start=1.0, stop=5.0, npoints=5, unit="GHz")

    def response(ep_r):
        line = MicrostripLine(
            w=1e-3,
            h=1e-3,
            dielectric=ConstantDielectric(ep_r=ep_r, tand=0.02),
            conductor=BulkConductor(sigma=jnp.inf),
            length=0.1,
        )
        return jnp.real(line.s(freq)[-1, 0, 0])

    near_air = MicrostripLine(
        w=1e-3,
        h=1e-3,
        dielectric=ConstantDielectric(ep_r=1.0 + 1e-12, tand=0.02),
        conductor=BulkConductor(sigma=jnp.inf),
        length=0.1,
    )
    near_air_g = near_air.immittance(freq).G
    slightly_above = MicrostripLine(
        w=1e-3,
        h=1e-3,
        dielectric=ConstantDielectric(ep_r=1.0 + 2e-12, tand=0.02),
        conductor=BulkConductor(sigma=jnp.inf),
        length=0.1,
    ).immittance(freq).G
    assert jnp.all(jnp.isfinite(near_air_g))
    assert jnp.allclose(near_air_g, slightly_above, rtol=1e-10, atol=1e-12)

    gradients = jax.vmap(jax.grad(response))(jnp.linspace(1.0 + 1e-12, 12.0, 25))
    assert jnp.all(jnp.isfinite(gradients))


def test_coaxial_line_has_no_modal_dispersion_field():
    """Homogeneously filled coax has no modal-dispersion stage."""
    assert not hasattr(CoaxialLine(length=0.1), "dispersion")


# -----------------------------------------------------------------------------
# Stripline
#
# scikit-rf has no stripline media, so these compare against the published
# closed-form results (Pozar, Microwave Engineering 4th ed., Section 3.7)
# and against the exact identities a homogeneously-filled line must satisfy.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("line", [
    MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, sigma=0.01), length=0.1
    ),
    StriplineLine(
        dielectric=ConstantDielectric(ep_r=2.2, sigma=0.01), length=0.1
    ),
])
def test_planar_static_conductivity_is_finite_and_continuous_at_dc(line):
    freq = Frequency.from_f(jnp.array([0.0, 1e-3, 1.0]))
    conductance = line.immittance(freq).G
    assert jnp.all(jnp.isfinite(conductance))
    assert jnp.all(conductance > 0)
    assert jnp.allclose(conductance[0], conductance[1], rtol=1e-12)


def test_microstrip_rejects_magnetic_substrate():
    line = MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, mu_r=2.0), length=0.1
    )
    with pytest.raises(Exception, match="nonmagnetic substrate"):
        line.immittance(Frequency.from_f(jnp.array([1e9])))


def test_stripline_rejects_thickness_not_below_ground_spacing():
    line = StriplineLine(b=1e-3, t=1e-3, length=0.1)
    with pytest.raises(Exception, match="0 < t < b"):
        line.immittance(Frequency.from_f(jnp.array([1e9])))

def test_stripline_effective_permittivity_equals_relative_permittivity():
    """Stripline is homogeneously filled, so there is no filling factor at all."""
    from pmrf.models import CohnStriplineFormulation, StriplineLine

    freq = Frequency(start=1.0, stop=20.0, npoints=11, unit="GHz")
    dielectric = ConstantDielectric(ep_r=2.2, tand=0.001)
    line = StriplineLine(w=2.655e-3, b=3.2e-3, dielectric=dielectric, length=0.1)

    quasi_static = CohnStriplineFormulation().quasi_static(
        w=line.w,
        b=line.b,
        t=line.t,
        ep_r=dielectric.properties(freq).ep_r,
    )

    assert jnp.array_equal(quasi_static.ep_eff, dielectric.properties(freq).ep_r)


def test_stripline_impedance_matches_pozar_design_example():
    """Pozar Example 3.7: er=2.20, b=0.32 cm, W/b=0.830 designs a 50 ohm line."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    b = 0.32e-2
    w_over_b = 30 * jnp.pi / (jnp.sqrt(2.2) * 50.0) - 0.441
    line = StriplineLine(
        w=w_over_b * b,
        b=b,
        dielectric=2.2,
        conductor=BulkConductor(sigma=jnp.inf),
        length=0.1,
    )

    zc = jnp.real(line.zc(freq))
    assert jnp.allclose(zc, 50.0, rtol=2e-3)


def test_stripline_dielectric_loss_is_the_homogeneous_limit():
    """A homogeneous line has alpha_d = k tan(delta) / 2 exactly."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    tand = 0.001
    line = StriplineLine(
        w=2.655e-3,
        b=3.2e-3,
        dielectric=ConstantDielectric(ep_r=2.2, tand=tand),
        conductor=BulkConductor(sigma=jnp.inf),
        length=1.0,
    )

    alpha = jnp.real(line.gammaL(freq))
    k = freq.w * jnp.sqrt(2.2) / c
    assert jnp.allclose(alpha, k * tand / 2, rtol=1e-3)


def test_stripline_attenuation_matches_pozar_worked_example():
    """Pozar Example 3.7 publishes alpha_d = 0.155 Np/m and alpha_c = 0.122 Np/m.

    Geometry: er=2.20, tand=0.001, b=0.32 cm, t=0.01 mm, copper, Z0=50 ohm,
    at 10 GHz.
    """
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    geometry = dict(w=2.655e-3, b=3.2e-3, t=1e-5, length=1.0)
    copper = BulkConductor(sigma=1 / 1.72e-8)

    lossless = StriplineLine(
        dielectric=2.2, conductor=BulkConductor(sigma=jnp.inf), **geometry
    )
    conductor_only = StriplineLine(dielectric=2.2, conductor=copper, **geometry)
    dielectric_only = StriplineLine(
        dielectric=ConstantDielectric(ep_r=2.2, tand=0.001),
        conductor=BulkConductor(sigma=jnp.inf),
        **geometry,
    )

    assert jnp.allclose(jnp.real(lossless.zc(freq)), 50.0, rtol=1e-3)
    assert jnp.allclose(jnp.real(lossless.gammaL(freq)), 0.0, atol=1e-12)
    # Pozar rounds to three digits, so match to that.
    assert jnp.allclose(jnp.real(conductor_only.gammaL(freq)), 0.122, atol=5e-4)
    assert jnp.allclose(jnp.real(dielectric_only.gammaL(freq)), 0.155, atol=5e-4)


def test_stripline_conductor_loss_scales_with_root_frequency():
    """Conductor loss enters through the surface impedance, so it follows sqrt(f)."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=1.0, stop=100.0, npoints=3, unit="GHz")
    line = StriplineLine(
        w=2.655e-3,
        b=3.2e-3,
        t=35e-6,
        dielectric=2.2,
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=1.0,
    )

    alpha = jnp.real(line.gammaL(freq))
    assert jnp.all(alpha > 0)
    # A hundredfold in frequency is a tenfold in skin-effect attenuation.
    assert jnp.allclose(alpha[2] / alpha[0], 10.0, rtol=1e-2)


def test_stripline_accepts_a_dispersive_dielectric():
    """A dispersive material needs no stripline-specific code: eps_eff tracks it."""
    from pmrf.models import CohnStriplineFormulation, StriplineLine

    freq = Frequency(start=1.0, stop=20.0, npoints=11, unit="GHz")
    dielectric = DjordjevicSarkarDielectric(ep_r=3.9, tand=0.02, f_ref=1e9)
    line = StriplineLine(w=2.655e-3, b=3.2e-3, dielectric=dielectric, length=0.1)

    ep_r = dielectric.properties(freq).ep_r
    quasi_static = CohnStriplineFormulation().quasi_static(
        w=line.w,
        b=line.b,
        t=line.t,
        ep_r=ep_r,
    )

    assert jnp.array_equal(quasi_static.ep_eff, ep_r)
    # The dielectric actually disperses over the band, so this is not vacuous.
    assert jnp.abs(jnp.real(ep_r[0]) - jnp.real(ep_r[-1])) > 1e-3

    s = line.s(freq)
    assert jnp.all(jnp.isfinite(s))


def test_stripline_narrow_strip_uses_the_fringing_correction():
    """Below W/b = 0.35 Cohn's effective width picks up the fringing term."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    b = 3.2e-3
    narrow = StriplineLine(
        w=0.2 * b, b=b, dielectric=2.2, conductor=BulkConductor(sigma=jnp.inf), length=0.1
    )

    w_e = 0.2 * b - (0.35 - 0.2) ** 2 * b
    expected = 30 * jnp.pi / jnp.sqrt(2.2) * b / (w_e + 0.441 * b)
    assert jnp.allclose(jnp.real(narrow.zc(freq)), expected, rtol=1e-6)


def test_stripline_high_impedance_branch_is_selected():
    """A narrow strip crosses sqrt(er)*Zc = 120 onto Cohn's other loss fit."""
    from pmrf.models import StriplineLine

    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit="GHz")
    geometry = dict(b=3.2e-3, t=35e-6, dielectric=2.2, length=1.0)
    copper = BulkConductor(sigma=1 / 1.72e-8)

    narrow = StriplineLine(w=0.2e-3, conductor=copper, **geometry)
    wide = StriplineLine(w=2.655e-3, conductor=copper, **geometry)

    assert jnp.sqrt(2.2) * jnp.real(narrow.zc(freq)) > 120
    assert jnp.sqrt(2.2) * jnp.real(wide.zc(freq)) < 120
    # A narrower strip concentrates the current, so it loses more.
    assert jnp.real(narrow.gammaL(freq)) > jnp.real(wide.gammaL(freq)) > 0


# -----------------------------------------------------------------------------
# Loss parameters on the coefficient-driven lines are NonNegative: a lossless
# line is on a closed bound, so it is in the model's domain (ADR-0007).
# -----------------------------------------------------------------------------

_ZERO_VALID_LINE = {
    "RLGCLine.R": (lambda x: RLGCLine(R=x, length=0.1), lambda line: line.R),
    "RLGCLine.G": (lambda x: RLGCLine(G=x, length=0.1), lambda line: line.G),
    "LowLossLine.A": (lambda x: LowLossLine(A=x, length=0.1), lambda line: line.A),
    "LowLossLine.tand": (lambda x: LowLossLine(tand=x, length=0.1), lambda line: line.tand),
    "PhysicalLine.A": (lambda x: PhysicalLine(A=x, length=0.1), lambda line: line.A),
    "PhysicalLine.tand": (lambda x: PhysicalLine(tand=x, length=0.1), lambda line: line.tand),
    "DatasheetLine.k1": (lambda x: DatasheetLine(k1=x, length=0.1), lambda line: line.k1),
    "DatasheetLine.k2": (lambda x: DatasheetLine(k2=x, length=0.1), lambda line: line.k2),
    "GeneralizedLine.R_dc": (lambda x: GeneralizedLine(R_dc=x, length=0.1), lambda line: line.R_dc),
}


def test_coefficient_lines_construct_with_defaults(basic_freq):
    for line in (RLGCLine(length=0.1), LowLossLine(length=0.1), PhysicalLine(length=0.1), DatasheetLine(length=0.1)):
        assert jnp.all(jnp.isfinite(line.s(basic_freq)))


@pytest.mark.parametrize("make, where", _ZERO_VALID_LINE.values(), ids=_ZERO_VALID_LINE.keys())
def test_zero_loss_line_has_finite_s_and_gradient(make, where, basic_freq):
    line = make(0.0)
    assert jnp.all(jnp.isfinite(line.s(basic_freq)))

    # Differentiate the model, not the parameter's raw map, which is -inf at 0.
    def power(x):
        return jnp.sum(jnp.abs(eqx.tree_at(where, line, x).s(basic_freq)) ** 2)

    assert jnp.isfinite(jax.grad(power)(0.0))


@pytest.mark.parametrize("make, where", _ZERO_VALID_LINE.values(), ids=_ZERO_VALID_LINE.keys())
def test_negative_line_loss_is_rejected(make, where):
    with pytest.raises(Exception, match="is not in NonNegative"):
        make(-1e-3)


@pytest.mark.parametrize("make", [
    lambda: RLGCLine(L=0.0, length=0.1),
    lambda: RLGCLine(C=0.0, length=0.1),
    lambda: RLGCLine(length=0.0),
    lambda: LowLossLine(zn=0.0, length=0.1),
    lambda: LowLossLine(f_ref=0.0, length=0.1),
    lambda: PhysicalLine(zn=0.0, length=0.1),
    lambda: PhysicalLine(f_ref=0.0, length=0.1),
    lambda: DatasheetLine(zn=0.0, length=0.1),
    lambda: DatasheetLine(vf=0.0, length=0.1),
    lambda: DatasheetLine(f_ref=0.0, length=0.1),
    lambda: FirstOrderEffectiveLineFormulation(f_low=0.0),
    lambda: FirstOrderEffectiveLineFormulation(f_high=0.0),
    lambda: GeneralizedLine(f_s=0.0, length=0.1),
], ids=[
    "RLGCLine.L", "RLGCLine.C", "RLGCLine.length",
    "LowLossLine.zn", "LowLossLine.f_ref",
    "PhysicalLine.zn", "PhysicalLine.f_ref",
    "DatasheetLine.zn", "DatasheetLine.vf", "DatasheetLine.f_ref",
    "FirstOrderEffectiveLineFormulation.f_low", "FirstOrderEffectiveLineFormulation.f_high",
    "GeneralizedLine.f_s",
])
def test_line_parameters_that_cannot_be_zero_reject_it(make):
    with pytest.raises(Exception, match="is not in Positive"):
        make()


def test_hammerstad_jensen_at_air_has_finite_gradient():
    """`sech(sqrt(ep_r - 1))` gave a NaN d/dep_r at ep_r = 1, even at t = 0 (#224).

    scikit-rf divides by ep_r - 1 in its loss analysis, so it is evaluated just above 1:
    the value at 1 + 1e-12 (observed 7e-13 relative), and the gradient as a forward
    difference between 1 + h and 1 + 2h with h = 1e-6, whose truncation is of order h
    (observed 2e-6 relative).
    """
    from skrf.media import MLine
    from pmrf.models import HammerstadJensenMicrostripFormulation

    freq = Frequency(start=0.1, stop=1.0, npoints=3, unit="GHz")
    formulation = HammerstadJensenMicrostripFormulation()

    def ours(ep_r):
        result = formulation.quasi_static(w=1e-3, h=1e-3, t=35e-6, ep_r=jnp.full(freq.npoints, ep_r))
        return jnp.stack([jnp.broadcast_to(x, (freq.npoints,)) for x in (result.zc, result.ep_eff, result.w_eff)])

    def skrf_(ep_r):
        media = MLine(
            freq.to_skrf(), w=1e-3, h=1e-3, t=35e-6, ep_r=ep_r, tand=0.0, rho=1.68e-8,
            rough=0.0, model="hammerstadjensen", disp="none", diel="frequencyinvariant",
        )
        return np.stack([np.broadcast_to(x, (freq.npoints,)) for x in (media.z0_characteristic, media.ep_reff, media.w_eff)])

    value, grad = jax.jvp(ours, (1.0,), (1.0,))
    h = 1e-6
    fd = (skrf_(1 + 2 * h) - skrf_(1 + h)) / h

    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(value, skrf_(1 + 1e-12), rtol=1e-11)
    for g, ref in zip(grad, fd):
        np.testing.assert_allclose(g, ref, rtol=0, atol=1e-5 * np.max(np.abs(ref)))
