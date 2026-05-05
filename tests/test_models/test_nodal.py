# tests/test_models/test_nodal.py
import pytest
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.rf import s2y

# Adjust imports based on your actual module structures
from pmrf.models import (
    GroundLifted, GroundExposed, Shunt, 
    Resistor, Short, Open
)

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

# ---------------------------------------------------------
# GroundLifted Tests
# ---------------------------------------------------------

def test_ground_lifted(basic_freq):
    """
    Test lifting the ground of an N-port network.
    The signal paths should map cleanly to the even indices.
    """
    base_model = Resistor(R=50.0) # 2-port network
    lifted_model = GroundLifted(model=base_model)
    
    s_lifted = lifted_model.s(basic_freq)
    
    # A grounded 2-port becomes a 4-port (2 signals, 2 returns)
    assert s_lifted.shape == (5, 4, 4)
    assert not jnp.any(jnp.isnan(s_lifted))
    
    # The signal matrix (even ports 0 and 2) should exactly equal the original matrix
    s_base = base_model.s(basic_freq)
    assert jnp.allclose(s_lifted[:, 0::2, 0::2], s_base)

# ---------------------------------------------------------
# GroundExposed Tests
# ---------------------------------------------------------

def test_ground_exposed_iam_property(basic_freq):
    """
    Test exposing the ground as the N+1 port.
    By definition of the Indefinite Admittance Matrix (IAM), the sum of 
    all rows and all columns in the resulting Y-matrix must equal zero.
    """
    base_model = Resistor(R=50.0) # 2-port network
    exposed_model = GroundExposed(model=base_model)
    
    s_exposed = exposed_model.s(basic_freq)
    
    # A grounded 2-port becomes a 3-port
    assert s_exposed.shape == (5, 3, 3)
    assert not jnp.any(jnp.isnan(s_exposed))
    
    # Convert back to Y parameters to check the IAM math
    # Use standard 50-ohm characteristic impedance
    y_exposed = s2y(s_exposed, z0=50.0)
    
    # Sum across the columns (axis=-1) for each row should be effectively zero
    row_sums = jnp.sum(y_exposed, axis=-1)
    assert jnp.allclose(row_sums, 0.0 + 0.0j, atol=1e-12)
    
    # Sum across the rows (axis=-2) for each column should be effectively zero
    col_sums = jnp.sum(y_exposed, axis=-2)
    assert jnp.allclose(col_sums, 0.0 + 0.0j, atol=1e-12)

# ---------------------------------------------------------
# Shunt Tests
# ---------------------------------------------------------

def test_shunt_open_circuit(basic_freq):
    """
    Test shunting an Open circuit (Gamma = 1).
    Placing an open circuit in parallel with a line should have absolutely 
    no effect, acting as a perfect, lossless thru-line.
    """
    open_model = Open()
    shunt_open = Shunt(model=open_model)
    s_thru = shunt_open.s(basic_freq)
    
    assert s_thru.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s_thru))
    
    # S11 and S22 should be 0 (no reflection)
    assert jnp.allclose(s_thru[:, 0, 0], 0.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s_thru[:, 1, 1], 0.0 + 0.0j, atol=1e-6)
    
    # S21 and S12 should be 1 (perfect transmission)
    assert jnp.allclose(s_thru[:, 1, 0], 1.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s_thru[:, 0, 1], 1.0 + 0.0j, atol=1e-6)

def test_shunt_short_circuit(basic_freq):
    """
    Test shunting a Short circuit (Gamma = -1).
    Placing a dead short to ground across a transmission line should 
    fully reflect all power and block all transmission.
    """
    short_model = Short()
    shunt_short = Shunt(model=short_model)
    s_block = shunt_short.s(basic_freq)
    
    assert s_block.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s_block))
    
    # S11 and S22 should be -1 (full reflection, out of phase)
    assert jnp.allclose(s_block[:, 0, 0], -1.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s_block[:, 1, 1], -1.0 + 0.0j, atol=1e-6)
    
    # S21 and S12 should be 0 (zero transmission)
    assert jnp.allclose(s_block[:, 1, 0], 0.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s_block[:, 0, 1], 0.0 + 0.0j, atol=1e-6)

def test_shunt_invalid_port_count():
    """Ensure Shunt raises an error if passed a multi-port network."""
    res_model = Resistor() # 2-port
    
    with pytest.raises(ValueError, match="Shunt requires a 1-port model"):
        Shunt(model=res_model)