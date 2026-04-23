# tests/test_optimize/test_optimize.py
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import parax as prx

from pmrf.core import Model, Frequency
from pmrf.optimize.minimize import minimize
from pmrf.optimize.scipy import ScipyMinimize

# ---------------------------------------------------------
# Dummy Concrete Models for Testing
# ---------------------------------------------------------

class DummyOptModel(Model):
    """A simple 1-port model with one free parameter for optimization."""
    val: prx.Parameter = 1.0

    def s(self, freq: Frequency) -> jnp.ndarray:
        # Returns an S-parameter matrix where the element is just `self.val`
        nf = freq.npoints
        return jnp.ones((nf, 1, 1), dtype=complex) * self.val

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

@pytest.fixture
def model():
    return DummyOptModel(val=1.0)

# ---------------------------------------------------------
# `minimize` Tests
# ---------------------------------------------------------

def test_minimize_scipy_unbounded(model, basic_freq):
    """Test standard unconstrained optimization using the default Scipy backend."""
    # Objective: minimize the distance from m.val to 5.0
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    result = minimize(obj_fn, model, basic_freq, solver=ScipyMinimize())
    
    assert isinstance(result.model, DummyOptModel)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-3)

def test_minimize_scipy_bounded(basic_freq):
    """Test that parameter boundaries are successfully intercepted and enforced."""
    # Initialize parameter at 1.0, trying to reach 5.0, but capped at 3.0
    bounded_param = prx.Parameter(1.0, bounds=jnp.array([0.0, 3.0]))
    bounded_model = DummyOptModel(val=bounded_param)
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    result = minimize(obj_fn, bounded_model, basic_freq, solver=ScipyMinimize())
    
    # The optimizer should hit the upper bound and stop
    assert jnp.allclose(result.model.val, 3.0, atol=1e-3)

def test_minimize_optimistix(model, basic_freq):
    """Test optimization using a JAX-native Optimistix solver."""
    optx = pytest.importorskip("optimistix")
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    # Use a gradient-free JAX solver
    solver = optx.NelderMead(rtol=1e-5, atol=1e-5)
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_steps=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_list_of_objectives(model, basic_freq):
    """Ensure that passing a list of callables automatically sums them via parax."""
    # The minimum of (x-2)^2 + (x-4)^2 is exactly x=3
    obj1 = lambda m, f: jnp.sum((m.val - 2.0)**2)
    obj2 = lambda m, f: jnp.sum((m.val - 4.0)**2)
    
    result = minimize([obj1, obj2], model, basic_freq)
    assert jnp.allclose(result.model.val, 3.0, atol=1e-3)

def test_minimize_no_free_params(basic_freq):
    """Ensure an exception is raised if there are no parameters to optimize."""
    # Freeze all parameters
    frozen_model = DummyOptModel(val=1.0).with_fixed_params('val')
    
    def obj_fn(m, f):
        return jnp.sum((m.val - 5.0)**2)
        
    with pytest.raises(Exception, match="Received no free parameters"):
        minimize(obj_fn, frozen_model, basic_freq)
