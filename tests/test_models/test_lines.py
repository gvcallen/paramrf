import pytest
import numpy as np
import jax.numpy as jnp
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
    epr = 4.0
    line = PhysicalLine(zn=50.0, length=length, epr=epr, A=0.0, tand=0.0)
    
    s = line.s(basic_freq)
    
    # Expected phase shift: beta * L = w * sqrt(epr) * L / c
    # Since S21 = exp(-j * beta * L), angle(S21) = -beta * L
    expected_phase = -basic_freq.w * jnp.sqrt(epr) * length / c
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
    # RG-58 dimensions roughly: epr=2.25, din=0.9mm, dout=2.95mm
    line = CoaxialLine(
        din=0.9e-3,
        dout=2.95e-3,
        dielectric=ConstantDielectric(epr=2.25, tand=0.0),
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
        dielectric=ConstantDielectric(epr=4.3, tand=0.0),
        conductor=BulkConductor(rho=0.0),
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
    assert jnp.allclose(line.dielectric.epr.value, 2.25)


def test_coaxial_line_matches_skrf(basic_freq):
    """Validate the coaxial immittance against scikit-rf's Tesche coaxial media."""
    skrf = pytest.importorskip("skrf")
    from skrf.media import Coaxial

    din, dout, epr, tand, rho = 0.9e-3, 2.95e-3, 2.25, 1e-3, 1.72e-8
    line = CoaxialLine(
        din=din,
        dout=dout,
        dielectric=ConstantDielectric(epr=epr, tand=tand),
        conductor=BulkConductor(rho=rho),
        length=0.5,
    )
    media = Coaxial(
        basic_freq.to_skrf(),
        Dint=din, Dout=dout, epsilon_r=epr, tan_delta=tand, sigma=1 / rho,
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

    W, H, epr = 3.0e-3, 1.6e-3, 4.3
    line = MicrostripLine(
        w=W,
        h=H,
        dielectric=ConstantDielectric(epr=epr, tand=0.0),
        conductor=BulkConductor(rho=0.0),
        length=0.1,
    )
    media = MLine(
        basic_freq.to_skrf(),
        w=W, h=H, ep_r=epr, tand=0.0, rho=0.0, rough=0.0,
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


def test_dispersive_dielectric_is_grid_independent():
    """
    A dispersive material is a function of frequency alone, so the same line
    evaluated on two different grids agrees at the frequencies they share.
    """
    line = CoaxialLine(
        din=0.9e-3,
        dout=2.95e-3,
        dielectric=DjordjevicSarkar(epr=4.3, tand=0.02),
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
    eps_r = np.full(npoints, 4.3 - 0.086j)
    zs = np.full(npoints, 0.01 + 0.01j)

    result = WheelerMicrostripFormulation().quasi_static(
        basic_freq, w=3.0e-3, h=1.6e-3, t=None, eps_r=eps_r, zs=zs
    )

    assert result.eps_eff.shape == (npoints,)
    assert jnp.all(jnp.real(result.zc) > 40.0)
    assert jnp.allclose(result.w_eff, 3.0e-3)
