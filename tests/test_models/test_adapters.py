# tests/test_adapters/test_adapters.py
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import parax as prx

from pmrf.core import Frequency
from pmrf.models import (
    Discrete, SingleProperty, SingleDiscreteProperty,
    Host, ContinuousCallable, DiscreteCallable, Measured
)
from pmrf.network_collection import NetworkCollection

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def coarse_freq():
    # 3 points: 1, 2, 3 GHz
    return Frequency(start=1.0, stop=3.0, npoints=3, unit='GHz')

@pytest.fixture
def fine_freq():
    # 5 points: 1, 1.5, 2.0, 2.5, 3.0 GHz
    return Frequency(start=1.0, stop=3.0, npoints=5, unit='GHz')

# ---------------------------------------------------------
# Abstract Adapter Dummies & Tests
# ---------------------------------------------------------

class DummyDiscrete(Discrete):
    """A 1-port discrete model with tabulated S-parameters."""
    # Define as a class attribute so Equinox handles __init__ automatically
    frequency: Frequency

    def s_discrete(self) -> jnp.ndarray:
        vals = jnp.array([1.0, 2.0, 3.0], dtype=complex)
        return vals.reshape(-1, 1, 1)

def test_abstract_discrete_interpolation(coarse_freq, fine_freq):
    """Ensure the discrete model interpolates correctly when queried at new frequencies."""
    model = DummyDiscrete(frequency=coarse_freq)
    
    # Query at the new, finer frequency grid
    s_interp = model.s(fine_freq)
    
    assert s_interp.shape == (5, 1, 1)
    # The middle point (1.5 GHz) should interpolate perfectly to 1.5
    assert jnp.allclose(s_interp[1, 0, 0], 1.5 + 0.0j)
    
class DummySinglePropertyY(SingleProperty):
    """A model that only natively knows its Y-parameters."""
    prop: str = 'y'
    def output(self, freq: Frequency) -> jnp.ndarray:
        return jnp.ones((freq.npoints, 1, 1), dtype=complex) * 0.02 # 50 ohm admittance

def test_single_property_routing(fine_freq):
    """Test that specifying property='y' correctly routes and triggers conversions."""
    model = DummySinglePropertyY()
    
    # Querying Y should hit output() directly
    y_mat = model.y(fine_freq)
    assert jnp.allclose(y_mat, 0.02)
    
    # Querying S should trigger y2s conversion natively
    s_mat = model.s(fine_freq)
    # 50 ohm admittance in a 50 ohm system -> matched (S11 = 0)
    assert jnp.allclose(s_mat, 0.0 + 0.0j)

# ---------------------------------------------------------
# Host Model Tests
# ---------------------------------------------------------

class DummyHostModel(Host):
    """A dummy host model representing an external simulator."""
    val: prx.Parameter = 10.0
    
    @property
    def primary_property(self): 
        return 's'
    
    @property
    def number_of_ports(self): 
        return 1
    
    def compute(self, freq: Frequency) -> np.ndarray:
        # Safely extract scalar from val (handles Parameters, JAX arrays, and Numpy arrays)
        v = float(np.array(getattr(self.val, 'value', self.val)).item())
        nf = freq.npoints
        return np.ones((nf, 1, 1), dtype=complex) * v

def test_host_model_single_execution(fine_freq):
    """Test standard single-thread execution of a Host model."""
    model = DummyHostModel(val=5.0)
    s = model.s(fine_freq)
    
    assert s.shape == (5, 1, 1)
    assert jnp.allclose(s, 5.0 + 0.0j)

def test_host_model_vmap_multithreading(fine_freq):
    """Test that Host models successfully map batched parameters using the ThreadPool."""
    # Create a batch of 3 parameter values
    batched_val = prx.Parameter(jnp.array([1.0, 2.0, 3.0]))
    model = DummyHostModel(val=batched_val)
    
    # VMAP across the parameter dimension!
    @jax.vmap
    def run_batch(m):
        return m.s(fine_freq)
        
    s_batch = run_batch(model)
    
    # Output should be (batch=3, nfreq=5, nports=1, nports=1)
    assert s_batch.shape == (3, 5, 1, 1)
    assert jnp.allclose(s_batch[0, 0, 0, 0], 1.0 + 0.0j)
    assert jnp.allclose(s_batch[2, 0, 0, 0], 3.0 + 0.0j)

# ---------------------------------------------------------
# Callable Adapter Tests
# ---------------------------------------------------------

def test_continuous_callable(fine_freq):
    """Test wrapping a standard mathematical python function."""
    def dummy_fn(theta, f_scaled):
        # theta is a (1,) array, f_scaled is (5,)
        # Create a dummy S11 matching the frequency
        return (theta[0] * f_scaled).reshape(-1, 1, 1)
        
    model = ContinuousCallable(
        theta=[prx.Parameter(2.0)], 
        fn=dummy_fn
    )
    
    s = model.s(fine_freq)
    # At 1.5 GHz, 2.0 * 1.5 = 3.0
    assert jnp.allclose(s[1, 0, 0], 3.0 + 0.0j)

# ---------------------------------------------------------
# Measured & NetworkCollection Tests
# ---------------------------------------------------------

def test_measured_skrf_interpolation(coarse_freq, fine_freq):
    """Test wrapping a scikit-rf Network and interpolating its data."""
    skrf = pytest.importorskip("skrf")
    
    # Create a dummy scikit-rf network at coarse frequencies
    skrf_freq = coarse_freq.to_skrf()
    s_data = np.array([1.0, 2.0, 3.0]).reshape(-1, 1, 1)
    ntwk = skrf.Network(frequency=skrf_freq, s=s_data, z0=50)
    
    # Wrap in Measured adapter
    measured_model = Measured(data=ntwk)
    
    # Query at the fine frequency
    s_interp = measured_model.s(fine_freq)
    
    assert s_interp.shape == (5, 1, 1)
    assert jnp.allclose(s_interp[1, 0, 0], 1.5 + 0.0j)

def test_measured_network_collection_getattr(coarse_freq):
    """Test dynamic attribute access for NetworkCollections."""
    skrf = pytest.importorskip("skrf")
    skrf_f = coarse_freq.to_skrf()
    
    ntwk1 = skrf.Network(frequency=skrf_f, s=np.ones((3,1,1)), z0=50)
    ntwk1.name = 'thru'  # Explicitly set the scikit-rf network name
    
    ntwk2 = skrf.Network(frequency=skrf_f, s=np.ones((3,1,1))*2, z0=50)
    ntwk2.name = 'line'  # Explicitly set the scikit-rf network name
    
    # NetworkCollection expects an iterable of Networks, NOT a dictionary!
    nc = NetworkCollection([ntwk1, ntwk2])
    measured_collection = Measured(data=nc)
    
    sub_model = measured_collection.thru
    assert isinstance(sub_model, Measured)
    
    # Verify the underlying data is correct
    assert np.allclose(sub_model.data.s, ntwk1.s)
    
    # Calling s() directly on the collection wrapper should fail
    with pytest.raises(Exception, match="Cannot call s\\(\\) on a Measured model that contains a NetworkCollection"):
        measured_collection.s(coarse_freq)