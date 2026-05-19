import pytest
import jax.numpy as jnp
from pmrf.frequency import Frequency

from scipy.constants import c, mu_0, epsilon_0

from pmrf.models import (
    PhaseLine, 
    ConstantRLGCLine, 
    PhysicalLine, 
    DatasheetLine, 
    CoaxialLine, 
    MicrostripLine, 
    FloatingLine
)

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=10, unit='GHz')

def test_phase_line(basic_freq):
    """Test ideal, lossless phase line limits."""
    # Create a 50-ohm line, 90 degrees at 5 GHz
    line = PhaseLine(zc=50.0, theta=90.0, f0=5.0e9)
    s = line.s(basic_freq)
    
    assert s.shape == (10, 2, 2)
    
    # S11 should be 0 for a line matched to the system z0 (which defaults to 50)
    assert jnp.allclose(s[:, 0, 0], 0.0, atol=1e-5)
    
    # Magnitude of transmission should be perfectly 1.0 (lossless)
    assert jnp.allclose(jnp.abs(s[:, 1, 0]), 1.0, atol=1e-5)

def test_floating_line(basic_freq):
    """Test the wrapper that converts a 2-port to a 4-port floating line."""
    base_line = PhaseLine(zc=50.0, theta=90.0, f0=5e9)
    float_line = FloatingLine(floating=base_line)
    
    s = float_line.s(basic_freq)
    
    # Should automatically scale up to a 4-port parameter matrix
    assert s.shape == (10, 4, 4)
    assert not jnp.any(jnp.isnan(s))

def test_constant_rlgc_line(basic_freq):
    """Test a basic transmission line evaluated from per-unit-length components."""
    # Lossless 50 ohm line (L=250nH, C=100pF -> Zc = sqrt(L/C) = 50)
    line = ConstantRLGCLine(R=0.0, G=0.0, L=250e-9, C=100e-12, length=0.1)
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
        din=0.9e-3, dout=2.95e-3, epr=2.25, length=1.0, tand=0.0, rho=0.0
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
    line = MicrostripLine(w=3.0e-3, h=1.6e-3, epr=4.3, length=0.1, tand=0.0, rho=0.0)
    
    zc, _ = line.zc_and_gammaL(basic_freq)
    
    # The Wheeler approximation for this geometry should yield an impedance near 50 ohms
    zc_real = jnp.real(zc)
    assert jnp.all(zc_real > 48.0)
    assert jnp.all(zc_real < 52.0)