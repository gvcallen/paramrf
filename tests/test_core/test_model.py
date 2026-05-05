# tests/test_core/test_model.py
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.models.model import Model

# ---------------------------------------------------------
# Dummy Concrete Models for Testing
# ---------------------------------------------------------

class DummyModelS(Model):
    """A simple 1-port model defining only S-parameters."""
    def s(self, freq: Frequency) -> jnp.ndarray:
        # Return a 1-port reflection of 0 (matched)
        nf = freq.npoints
        return jnp.zeros((nf, 1, 1), dtype=complex)

class DummyModelZ(Model):
    """A simple 1-port model defining only Z-parameters."""
    def z(self, freq: Frequency) -> jnp.ndarray:
        # Return a 1-port impedance of 50 ohms
        nf = freq.npoints
        return jnp.ones((nf, 1, 1), dtype=complex) * 50.0

class DummyModelS2Port(Model):
    """A simple 2-port model defining only S-parameters for cascade testing."""
    def s(self, freq: Frequency) -> jnp.ndarray:
        nf = freq.npoints
        return jnp.zeros((nf, 2, 2), dtype=complex)

class DummyCompositionalModel(Model):
    """A compositional model that delegates to another model via __call__."""
    def __call__(self) -> Model:
        return DummyModelS()

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=10, unit='GHz')

@pytest.fixture
def model_s():
    return DummyModelS()

@pytest.fixture
def model_z():
    return DummyModelZ()

@pytest.fixture
def model_s_2port():
    return DummyModelS2Port()

@pytest.fixture
def model_comp():
    return DummyCompositionalModel()

# ---------------------------------------------------------
# Primary Property Resolution
# ---------------------------------------------------------

def test_primary_property_resolution(model_s, model_z, model_comp):
    """Ensure the metaclass/property correctly identifies the overridden function."""
    assert model_s.primary_property == 's'
    assert model_z.primary_property == 'z'
    # Compositional models should inherit the primary property of what they build
    assert model_comp.primary_property == 's'

def test_primary_function_execution(model_s, basic_freq):
    """Test that calling .primary() routes to the correct evaluation method."""
    out = model_s.primary(basic_freq)
    assert out.shape == (10, 1, 1)

# ---------------------------------------------------------
# Network Conversion Hub
# ---------------------------------------------------------

def test_conversion_hub_z_to_s(model_z, basic_freq):
    """
    Test the JIT-compiled conversion graph. 
    DummyModelZ only defines .z(), so calling .s() should internally invoke z2s.
    A 50-ohm Z-parameter with z0=50 should yield an S-parameter of 0.
    """
    s_params = model_z.s(basic_freq)
    
    assert s_params.shape == (10, 1, 1)
    # The reflection coefficient of a 50 ohm load in a 50 ohm system is 0
    assert jnp.allclose(s_params, 0.0 + 0.0j)

def test_conversion_hub_s_to_a(model_s, basic_freq):
    """Calling .a() on an S-defined model should trigger s2a conversion."""
    try:
        a_params = model_s.a(basic_freq)
    except IndexError as e:
        # Expected: ABCD conversion natively rejects 1-port matrices
        pass
    except Exception as e:
        pytest.fail(f"Unexpected error during s -> a conversion routing: {e}")

# ---------------------------------------------------------
# Introspection Properties
# ---------------------------------------------------------

def test_nports_and_tuples(model_s):
    """Test that jax.eval_shape correctly infers the number of ports."""
    assert model_s.number_of_ports == 1
    assert model_s.nports == 1
    assert model_s.port_tuples == [(0, 0)]

# ---------------------------------------------------------
# Dynamic Method Generation (__init_subclass__)
# ---------------------------------------------------------

def test_dynamic_methods_created(model_s, basic_freq):
    """
    Check if methods like `s_mag` or `s_db` were successfully dynamically 
    attached by the `__init_subclass__` hook.
    """
    assert hasattr(model_s, 's_mag')
    
    s_mag = model_s.s_mag(basic_freq)
    assert s_mag.shape == (10, 1, 1)

# ---------------------------------------------------------
# Magic Methods & Composition
# ---------------------------------------------------------

def test_compositional_operators(model_s_2port):
    """
    Test the syntactic sugar for cascades (**) and terminations (@).
    """
    try:
        # Cascade requires 2N-port models: 2-port ** 2-port
        cascaded_model = model_s_2port ** model_s_2port
        assert isinstance(cascaded_model, Model)
        
        # Termination requires a 2-port network terminated by a 1-port load
        terminated_model = model_s_2port @ 'short'
        assert isinstance(terminated_model, Model)
    except ImportError:
        pytest.skip("pmrf.models not fully available in test environment.")

# ---------------------------------------------------------
# scikit-rf Interoperability
# ---------------------------------------------------------

def test_to_skrf_conversion(model_s, basic_freq):
    """Test generating an explicit skrf.Network object."""
    skrf = pytest.importorskip("skrf")
    import numpy as np
    
    ntwk = model_s.to_skrf(basic_freq)
    assert isinstance(ntwk, skrf.Network)
    assert ntwk.s.shape == (10, 1, 1)
    
    # scikit-rf z0 is an array of shape (nfreq, nports), so we must use np.allclose
    assert np.allclose(ntwk.z0.real, 50.0)

# ---------------------------------------------------------
# JAX & Equinox Compatibility Boundaries
# ---------------------------------------------------------

def test_jit_compatibility(model_z, basic_freq):
    """
    Ensure that the Model class and its filter_jit decorators compile successfully.
    """
    @jax.jit
    def compile_s_params(model, f):
        return model.s(f)

    s_out = compile_s_params(model_z, basic_freq)
    assert s_out.shape == (10, 1, 1)