# tests/test_models/test_topologies.py
import pytest
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models import PiCLC, BoxCLCC, TeeLCL, LSectionLC

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

# ---------------------------------------------------------
# Pi-CLC Tests
# ---------------------------------------------------------

def test_piclc_general(basic_freq):
    """Test standard Pi-CLC execution and reciprocity."""
    model = PiCLC(C1=1e-12, L=1e-9, C2=2e-12)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))
    
    # Passive lumped network should be strictly reciprocal (S21 == S12)
    assert jnp.allclose(s[:, 1, 0], s[:, 0, 1], atol=1e-6)

def test_piclc_zero_inductance(basic_freq):
    """Test the edge case where L = 0."""
    # When L=0, the network is just C1 and C2 in parallel to ground
    model = PiCLC(C1=1e-12, L=0.0, C2=2e-12)
    a_mat = model.a(basic_freq)
    
    # ABCD for a shunt admittance Y is [[1, 0], [Y, 1]]
    Y_total = 1j * basic_freq.w * (1e-12 + 2e-12)
    
    assert jnp.allclose(a_mat[:, 0, 0], 1.0)
    assert jnp.allclose(a_mat[:, 0, 1], 0.0)
    assert jnp.allclose(a_mat[:, 1, 0], Y_total)
    assert jnp.allclose(a_mat[:, 1, 1], 1.0)

# ---------------------------------------------------------
# Box-CLCC Tests
# ---------------------------------------------------------

def test_boxclcc_general(basic_freq):
    """Test standard Box-CLCC execution (4-port)."""
    model = BoxCLCC(C1=1e-12, L=1e-9, C2=1e-12, C3=0.5e-12)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 4, 4)
    assert not jnp.any(jnp.isnan(s))

def test_boxclcc_zero_inductance(basic_freq):
    """Test the edge case where L is approx 0.0."""
    model = BoxCLCC(L=0.0, C1=1e-12, C2=1e-12, C3=1e-12)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 4, 4)
    assert not jnp.any(jnp.isnan(s))

# ---------------------------------------------------------
# Tee-LCL Tests
# ---------------------------------------------------------

def test_teelcl_general(basic_freq):
    """Test standard Tee-LCL execution and reciprocity."""
    model = TeeLCL(L1=1e-9, C=1e-12, L2=2e-9)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))
    assert jnp.allclose(s[:, 1, 0], s[:, 0, 1], atol=1e-6)

def test_teelcl_zero_capacitance(basic_freq):
    """Test the jax.lax.cond edge case where C = 0."""
    # When C=0, the network is just L1 and L2 in series
    model = TeeLCL(L1=1e-9, C=0.0, L2=2e-9)
    a_mat = model.a(basic_freq)
    
    # ABCD for a series impedance Z is [[1, Z], [0, 1]]
    Z_total = 1j * basic_freq.w * (1e-9 + 2e-9)
    
    assert jnp.allclose(a_mat[:, 0, 0], 1.0)
    assert jnp.allclose(a_mat[:, 0, 1], Z_total)
    assert jnp.allclose(a_mat[:, 1, 0], 0.0)
    assert jnp.allclose(a_mat[:, 1, 1], 1.0)

# ---------------------------------------------------------
# L-Section Tests
# ---------------------------------------------------------

def test_lsection_general(basic_freq):
    """Test standard L-Section execution."""
    model = LSectionLC(L=1e-9, C=1e-12)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))

def test_lsection_thru(basic_freq):
    """Test the ideal thru case where L=0 and C=0."""
    model = LSectionLC(L=0.0, C=0.0)
    s = model.s(basic_freq)
    
    # An ideal thru-line has S11=0 and S21=1
    assert jnp.allclose(s[:, 0, 0], 0.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s[:, 1, 0], 1.0 + 0.0j, atol=1e-6)