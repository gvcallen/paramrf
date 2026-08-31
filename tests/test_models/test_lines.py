import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from pmrf.frequency import Frequency

from scipy.constants import c, mu_0, epsilon_0

from pmrf.models import (
    PhaseLine, 
    RLGCLine, 
    PhysicalLine, 
    DatasheetLine, 
    CoaxialLine, 
    MicrostripLine, 
    FloatingLine
)
from pmrf.materials import BulkConductor, ConstantDielectric, DjordjevicSarkar

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
    PhysicalLine(length=0.1),
    DatasheetLine(length=0.1),
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
    ep_r = 4.0
    line = PhysicalLine(zn=50.0, length=length, ep_r=ep_r, A=0.0, tand=0.0)
    
    s = line.s(basic_freq)
    
    # Expected phase shift: beta * L = w * sqrt(ep_r) * L / c
    # Since S21 = exp(-j * beta * L), angle(S21) = -beta * L
    expected_phase = -basic_freq.w * jnp.sqrt(ep_r) * length / c
    actual_phase = jnp.unwrap(jnp.angle(s[:, 1, 0]))
    
    assert jnp.allclose(actual_phase, expected_phase, atol=1e-4)
    # Lossless matched line -> S11 should be exactly 0
    assert jnp.allclose(s[:, 0, 0], 0.0, atol=1e-5)

def test_datasheet_line_attenuation(basic_freq):
    """Test that the DatasheetLine applies the correct frequency-scaled attenuation."""
    # Pure skin-effect loss (k1), no dielectric loss (k2), normalized coefficients
    line = DatasheetLine(
        zn=50.0, vf=1.0, k1=2.0, k2=0.0, length=10.0, loss_coeffs_normalized=True
    )
    
    s = line.s(basic_freq)
    
    # alpha_c = k1_norm * ln(10)/20 * sqrt(w)
    # |S21| = exp(-alpha_c * L)
    alpha_c = 2.0 * (jnp.log(10) / 20) * jnp.sqrt(basic_freq.w)
    expected_mag = jnp.exp(-alpha_c * 10.0)
    
    assert jnp.allclose(jnp.abs(s[:, 1, 0]), expected_mag, atol=1e-4)

def test_coaxial_line_impedance(basic_freq):
    """Verify CoaxialLine matches the analytical Zc formula for a lossless coax."""
    # RG-58 dimensions roughly: ep_r=2.25, d_in=0.9mm, d_out=2.95mm
    line = CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=ConstantDielectric(ep_r=2.25, tand=0.0),
        conductor=BulkConductor(rho=0.0),
        length=1.0,
    )
    
    # Extract Zc directly from the internal method
    zc, _ = line.zc_and_gammaL(basic_freq)
    
    # Analytical Zc = (1 / 2*pi) * sqrt(mu / eps) * ln(b/a)
    # (Note: 1/(2*pi) * sqrt(mu_0/eps_0) is approx 59.958 ohms)
    expected_zc = (1 / (2 * jnp.pi)) * jnp.sqrt(mu_0 / (epsilon_0 * 2.25)) * jnp.log(2.95 / 0.9)
    
    # Real part of Zc should match the theoretical lossless impedance
    assert jnp.allclose(jnp.real(zc), expected_zc, rtol=1e-3)

def test_microstrip_line_impedance(basic_freq):
    """Verify MicrostripLine evaluates Wheeler's formula correctly."""
    # Standard 50-ohm trace on 1.6mm FR4 (Er=4.3) is roughly 3.0mm wide
    line = MicrostripLine(
        w=3.0e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
        conductor=BulkConductor(rho=0.0),
        dispersion=None,
        length=0.1,
    )
    
    zc, _ = line.zc_and_gammaL(basic_freq)
    
    # The Wheeler approximation for this geometry should yield an impedance near 50 ohms
    zc_real = jnp.real(zc)
    assert jnp.all(zc_real > 48.0)
    assert jnp.all(zc_real < 52.0)

def test_material_coercion():
    """Scalars and tuples coerce into the corresponding material modules."""
    line = CoaxialLine(dielectric=(2.25, 0.001), conductor=1.68e-8, length=0.1)

    assert isinstance(line.dielectric, ConstantDielectric)
    assert isinstance(line.conductor, BulkConductor)
    assert jnp.allclose(line.dielectric.ep_r.value, 2.25)


def test_coaxial_line_matches_skrf(basic_freq):
    """Validate the coaxial immittance against scikit-rf's Tesche coaxial media."""
    skrf = pytest.importorskip("skrf")
    from skrf.media import Coaxial

    d_in, d_out, ep_r, tand, rho = 0.9e-3, 2.95e-3, 2.25, 1e-3, 1.72e-8
    line = CoaxialLine(
        d_in=d_in,
        d_out=d_out,
        dielectric=ConstantDielectric(ep_r=ep_r, tand=tand),
        conductor=BulkConductor(rho=rho),
        length=0.5,
    )
    media = Coaxial(
        basic_freq.to_skrf(),
        Dint=d_in, Dout=d_out, epsilon_r=ep_r, tan_delta=tand, sigma=1 / rho,
        model='tesche',
    )

    imm = line.immittance(basic_freq)

    # G and C agree to machine precision, L to 1e-6. R is looser for a known
    # reason: scikit-rf implements Tesche's equivalent circuit, eq. (14),
    # Z = Rdc + Zhf/(1 + Zhf/(jw*Lint)), which carries the DC resistance and the
    # internal inductance of the rod. ParamRF implements only the Zhf term, the
    # high-frequency asymptote, so the two converge as frequency rises and
    # diverge without bound below the skin-depth transition. scikit-rf is the
    # more complete model here; see the note on issue #63.
    assert jnp.allclose(imm.R, media.R, rtol=1e-2)
    assert jnp.allclose(imm.L, media.L, rtol=1e-6)
    assert jnp.allclose(imm.G, media.G, rtol=1e-12)
    assert jnp.allclose(imm.C, media.C, rtol=1e-12)

    zc, gammaL = line.zc_and_gammaL(basic_freq)
    assert jnp.allclose(zc, media.z0_characteristic, rtol=1e-4)
    assert jnp.allclose(gammaL / 0.5, media.gamma, rtol=1e-4)


def test_microstrip_line_matches_skrf(basic_freq):
    """Validate the microstrip line against scikit-rf's Wheeler microstrip media."""
    skrf = pytest.importorskip("skrf")
    from skrf.media import MLine

    W, H, ep_r = 3.0e-3, 1.6e-3, 4.3
    line = MicrostripLine(
        w=W,
        h=H,
        dielectric=ConstantDielectric(ep_r=ep_r, tand=0.0),
        conductor=BulkConductor(rho=0.0),
        dispersion=None,
        length=0.1,
    )
    media = MLine(
        basic_freq.to_skrf(),
        w=W, h=H, ep_r=ep_r, tand=0.0, rho=0.0, rough=0.0,
        model='wheeler', disp='none', diel='frequencyinvariant',
    )

    zc, gammaL = line.zc_and_gammaL(basic_freq)

    # scikit-rf implements Wheeler's own closed form, where ParamRF uses the
    # Hammerstad simplification of it. The two agree on the impedance to half a
    # percent and on the effective permittivity, hence the phase constant, to
    # under two percent.
    assert jnp.allclose(zc, media.z0_characteristic, rtol=1e-2)
    assert jnp.allclose(gammaL / 0.1, media.gamma, rtol=2e-2)


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
            zc=jnp.array([50.0]), gamma=jnp.array([-1.0 + 1j]), w=jnp.array([1.0])
        )


def test_dispersive_dielectric_is_grid_independent():
    """
    A dispersive material is a function of frequency alone, so the same line
    evaluated on two different grids agrees at the frequencies they share.
    """
    line = CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=DjordjevicSarkar(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(rho=1.68e-8),
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
        basic_freq, w=3.0e-3, h=1.6e-3, t=None, ep_r=ep_r, zs=zs
    )

    assert result.ep_eff.shape == (npoints,)
    assert jnp.all(jnp.real(result.zc) > 40.0)
    assert jnp.allclose(result.w_eff, 3.0e-3)


def test_lossy_microstrip_preserves_dispersed_zc_and_phase():
    """The dispersed modal solution survives exact immittance inversion."""
    from pmrf.models import HammerstadJensenMicrostripFormulation, KirschningJansen

    freq = Frequency(start=1.0, stop=50.0, npoints=51, unit="GHz")
    formulation = HammerstadJensenMicrostripFormulation()
    dispersion = KirschningJansen()
    line = MicrostripLine(
        w=0.2e-3,
        h=0.5e-3,
        t=18e-6,
        dielectric=ConstantDielectric(ep_r=10.0, tand=0.05),
        conductor=BulkConductor(rho=1e-6),
        formulation=formulation,
        dispersion=dispersion,
        length=0.1,
    )

    ep_r = line.dielectric.epsilon_r(freq)
    zs = line.conductor.surface_impedance(freq)
    quasi_static = formulation.quasi_static(
        freq, w=line.w, h=line.h, t=line.t, ep_r=ep_r, zs=zs
    )
    ep_eff, expected_zc = dispersion.disperse(
        freq,
        ep_eff_0=quasi_static.ep_eff,
        zc_0=quasi_static.zc,
        ep_r=ep_r,
        w=line.w,
        w_eff=quasi_static.w_eff,
        h=line.h,
        t=line.t,
    )

    zc, gamma_length = line.zc_and_gammaL(freq)
    gamma = gamma_length / line.length

    assert jnp.allclose(zc, expected_zc, rtol=1e-12, atol=1e-12)
    expected_beta = jnp.imag(1j * freq.w * jnp.sqrt(ep_eff) / c)
    assert jnp.allclose(jnp.imag(gamma), expected_beta, rtol=1e-12, atol=1e-12)


def test_microstrip_without_dispersion_preserves_quasi_static_immittance():
    """Disabling stage three reproduces the pre-dispersion pipeline exactly."""
    freq = Frequency(start=1.0, stop=20.0, npoints=21, unit="GHz")
    line = MicrostripLine(
        w=3e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(rho=1.72e-8),
        dispersion=None,
        length=0.1,
    )

    ep_r = line.dielectric.epsilon_r(freq)
    zs = line.conductor.surface_impedance(freq)
    quasi_static = line.formulation.quasi_static(
        freq, w=line.w, h=line.h, t=line.t, ep_r=ep_r, zs=zs
    )

    actual = line.immittance(freq)
    expected = quasi_static.to_immittance(freq, zs)
    assert jnp.array_equal(actual.Z, expected.Z)
    assert jnp.array_equal(actual.Y, expected.Y)


def test_microstrip_defaults_to_kirschning_jansen():
    """Modal dispersion is the accuracy-oriented microstrip default."""
    from pmrf.models import KirschningJansen

    assert isinstance(MicrostripLine(length=0.1).dispersion, KirschningJansen)


def test_hammerstad_jensen_finite_thickness_matches_skrf():
    """The thickness-aware quasi-static formulation agrees with scikit-rf."""
    from skrf.media import MLine
    from pmrf.models import HammerstadJensenMicrostripFormulation

    freq = Frequency(start=1.0, stop=10.0, npoints=10, unit="GHz")
    formulation = HammerstadJensenMicrostripFormulation()
    ep_r = jnp.full(freq.npoints, 4.3)
    result = formulation.quasi_static(
        freq, w=1e-3, h=1e-3, t=35e-6, ep_r=ep_r, zs=jnp.zeros(freq.npoints)
    )
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


@pytest.mark.parametrize("ep_r", [2.2, 4.3, 10.0])
@pytest.mark.parametrize("width_height", [0.1, 1.0, 10.0])
def test_kirschning_jansen_matches_skrf(ep_r, width_height):
    """Kirschning--Jansen agrees with its independent scikit-rf implementation."""
    from skrf.media import MLine
    from pmrf.models import HammerstadJensenMicrostripFormulation

    freq = Frequency(start=0.1, stop=50.0, npoints=101, unit="GHz")
    h = 1e-3
    line = MicrostripLine(
        w=width_height * h,
        h=h,
        dielectric=ConstantDielectric(ep_r=ep_r, tand=0.0),
        conductor=BulkConductor(rho=0.0),
        formulation=HammerstadJensenMicrostripFormulation(),
        length=0.1,
    )
    media = MLine(
        freq.to_skrf(),
        w=width_height * h,
        h=h,
        ep_r=ep_r,
        tand=0.0,
        rho=0.0,
        rough=0.0,
        model="hammerstadjensen",
        disp="kirschningjansen",
        diel="frequencyinvariant",
        compatibility_mode=None,
    )

    zc, gamma_length = line.zc_and_gammaL(freq)
    assert jnp.allclose(zc, media.z0_characteristic, rtol=1e-3)
    assert jnp.allclose(gamma_length / line.length, media.gamma, rtol=1e-3)


@pytest.mark.parametrize(
    ("convention", "compatibility_mode"),
    [("complex", None), ("real", "qucs")],
)
def test_microstrip_epsilon_convention_matches_skrf(convention, compatibility_mode):
    """Both documented permittivity conventions match their scikit-rf modes."""
    from skrf.media import MLine
    from pmrf.models import HammerstadJensenMicrostripFormulation

    freq = Frequency(start=1.0, stop=50.0, npoints=51, unit="GHz")
    line = MicrostripLine(
        w=1e-3,
        h=1e-3,
        t=35e-6,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(rho=1.68e-8),
        formulation=HammerstadJensenMicrostripFormulation(),
        epsilon_convention=convention,
        length=0.1,
    )
    media = MLine(
        freq.to_skrf(),
        w=1e-3,
        h=1e-3,
        t=35e-6,
        ep_r=4.3,
        tand=0.02,
        rho=1.68e-8,
        rough=0.0,
        model="hammerstadjensen",
        disp="kirschningjansen",
        diel="frequencyinvariant",
        compatibility_mode=compatibility_mode,
    )

    zc, gamma_length = line.zc_and_gammaL(freq)
    assert jnp.allclose(zc, media.z0_characteristic, rtol=1e-3)
    assert jnp.allclose(gamma_length / line.length, media.gamma, rtol=1e-3)


def test_real_epsilon_convention_retains_loss_without_modal_dispersion():
    """The convention switch is independent of enabling modal dispersion."""
    freq = Frequency(start=1.0, stop=10.0, npoints=10, unit="GHz")
    line = MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(rho=0.0),
        dispersion=None,
        epsilon_convention="real",
        length=0.1,
    )

    assert jnp.all(line.immittance(freq).G > 0)


def test_microstrip_rejects_unknown_epsilon_convention():
    with pytest.raises(ValueError, match="epsilon_convention"):
        MicrostripLine(epsilon_convention="hybrid", length=0.1)


def test_microstrip_near_air_has_finite_conductance_and_gradient():
    """The old filling-factor singularity cannot return at εr approaching one."""
    freq = Frequency(start=1.0, stop=5.0, npoints=5, unit="GHz")

    def response(ep_r):
        line = MicrostripLine(
            w=1e-3,
            h=1e-3,
            dielectric=ConstantDielectric(ep_r=ep_r, tand=0.02),
            conductor=BulkConductor(rho=0.0),
            length=0.1,
        )
        return jnp.real(line.s(freq)[-1, 0, 0])

    near_air = MicrostripLine(
        w=1e-3,
        h=1e-3,
        dielectric=ConstantDielectric(ep_r=1.0 + 1e-12, tand=0.02),
        conductor=BulkConductor(rho=0.0),
        length=0.1,
    )
    near_air_g = near_air.immittance(freq).G
    slightly_above = MicrostripLine(
        w=1e-3,
        h=1e-3,
        dielectric=ConstantDielectric(ep_r=1.0 + 2e-12, tand=0.02),
        conductor=BulkConductor(rho=0.0),
        length=0.1,
    ).immittance(freq).G
    assert jnp.all(jnp.isfinite(near_air_g))
    assert jnp.allclose(near_air_g, slightly_above, rtol=1e-10, atol=1e-12)

    gradients = jax.vmap(jax.grad(response))(jnp.linspace(1.0 + 1e-12, 12.0, 25))
    assert jnp.all(jnp.isfinite(gradients))


def test_coaxial_line_has_no_modal_dispersion_field():
    """Homogeneously filled coax has no modal-dispersion stage."""
    assert not hasattr(CoaxialLine(length=0.1), "dispersion")
