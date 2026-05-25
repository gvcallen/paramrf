import pytest
import jax.numpy as jnp
from pmrf.frequency import Frequency

from pmrf.models import (
    Short, Open, Match,
    Resistor, Capacitor, Inductor,
    CapacitorQ, InductorQ
)

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

def test_ideal_loads(basic_freq):
    """Test 1-port ideal static loads (gamma representations)."""
    # Short circuit (Gamma = -1)
    s_short = Short().build().s(basic_freq)
    assert s_short.shape == (5, 1, 1)
    assert jnp.allclose(s_short, -1.0 + 0.0j)

    # Open circuit (Gamma = +1)
    s_open = Open().build().s(basic_freq)
    assert jnp.allclose(s_open, 1.0 + 0.0j)

    # Matched load (Gamma = 0)
    s_match = Match().build().s(basic_freq)
    assert jnp.allclose(s_match, 0.0 + 0.0j)

def test_series_resistor(basic_freq):
    """Test known RF limits for a series resistor."""
    res = Resistor(R=50.0)
    s = res.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    
    # In a 50 ohm system, a 50 ohm series resistor yields:
    # S21 = 2*Z0 / (R + 2*Z0) = 100 / 150 = 2/3
    assert jnp.allclose(s[:, 1, 0], 2.0/3.0, atol=1e-5)
    # S11 = R / (R + 2*Z0) = 50 / 150 = 1/3
    assert jnp.allclose(s[:, 0, 0], 1.0/3.0, atol=1e-5)

@pytest.mark.parametrize("model_class, param_kwargs", [
    (Capacitor, {'C': 1e-12}),
    (Inductor, {'L': 1e-9}),
])
def test_reactive_elements_execution(model_class, param_kwargs, basic_freq):
    """Ensure reactive lumped elements evaluate properly without shape or NaN errors."""
    model = model_class(**param_kwargs)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))

def test_q_components_execution(basic_freq):
    """Ensure lumped elements with finite Quality Factor evaluate properly."""
    cap_q = CapacitorQ(C=1e-12, Q=50.0)
    ind_q = InductorQ(L=1e-9, Q=50.0)
    
    s_c = cap_q.s(basic_freq)
    s_i = ind_q.s(basic_freq)
    
    assert s_c.shape == (5, 2, 2)
    assert s_i.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s_c))
    assert not jnp.any(jnp.isnan(s_i))