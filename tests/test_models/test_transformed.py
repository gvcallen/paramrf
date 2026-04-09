# tests/test_models/test_transformed.py
import pytest
import jax.numpy as jnp

from pmrf.core import Frequency
from pmrf.rf import s2y

# Adjust imports based on your actual module structures
from pmrf.models import (
    Renumbered, Flipped, Stacked, GroundLifted, GroundExposed,
    LSectionLC, Resistor, Short
)

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

# ---------------------------------------------------------
# Renumbered & Flipped Tests
# ---------------------------------------------------------

def test_renumbered_general(basic_freq):
    """Test explicit port renumbering using an asymmetric L-section."""
    # L-Section is asymmetrical: S11 != S22
    base_model = LSectionLC(L=1e-9, C=1e-12)
    s_base = base_model.s(basic_freq)
    
    # Manually swap port 0 and port 1
    renum_model = Renumbered(model=base_model, from_ports=(0, 1), to_ports=(1, 0))
    s_renum = renum_model.s(basic_freq)
    
    # S11 of renumbered should equal S22 of base
    assert jnp.allclose(s_renum[:, 0, 0], s_base[:, 1, 1])
    # S22 of renumbered should equal S11 of base
    assert jnp.allclose(s_renum[:, 1, 1], s_base[:, 0, 0])
    
def test_flipped_general(basic_freq):
    """Test the Flipped convenience wrapper."""
    base_model = LSectionLC(L=1e-9, C=1e-12)
    s_base = base_model.s(basic_freq)
    
    flipped_model = Flipped(model=base_model)
    s_flipped = flipped_model.s(basic_freq)
    
    # Flipped on a 2-port should exactly match swapping 0 and 1
    assert jnp.allclose(s_flipped[:, 0, 0], s_base[:, 1, 1])
    assert jnp.allclose(s_flipped[:, 1, 1], s_base[:, 0, 0])

def test_flipped_odd_ports_error():
    """Ensure Flipped raises an error for models with an odd number of ports."""
    # Short is a 1-port network
    with pytest.raises(ValueError, match="You can only flip multiple-of-two-port"):
        Flipped(model=Short())

# ---------------------------------------------------------
# Stacked Tests
# ---------------------------------------------------------

def test_stacked_block_diagonal(basic_freq):
    """Ensure Stacked correctly combines S-matrices along the diagonal."""
    res_model = Resistor(R=50.0)    # 2-port
    short_model = Short()           # 1-port
    
    stacked_model = Stacked(models=(res_model, short_model))
    s_stacked = stacked_model.s(basic_freq)
    
    assert s_stacked.shape == (5, 3, 3)
    
    # Check that the top-left 2x2 block exactly matches the Resistor
    s_res = res_model.s(basic_freq)
    assert jnp.allclose(s_stacked[:, 0:2, 0:2], s_res)
    
    # Check that the bottom-right 1x1 block exactly matches the Short
    s_short = short_model.s(basic_freq)
    assert jnp.allclose(s_stacked[:, 2:3, 2:3], s_short)
    
    # Check that off-diagonal blocks (uncoupled ports) are exactly zero
    assert jnp.allclose(s_stacked[:, 0:2, 2], 0.0 + 0.0j)
    assert jnp.allclose(s_stacked[:, 2, 0:2], 0.0 + 0.0j)
