# tests/test_optimize/test_optimize.py
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import parax as prx

from pmrf.core import Model, Frequency
from pmrf.optimize.minimize import minimize
from pmrf.optimize.fit import fit
from pmrf.optimize.solvers import ScipyMinimizer

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
    
    result = minimize(obj_fn, model, basic_freq, solver=ScipyMinimizer())
    
    assert isinstance(result.model, DummyOptModel)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-3)

def test_minimize_scipy_bounded(basic_freq):
    """Test that parameter boundaries are successfully intercepted and enforced."""
    # Initialize parameter at 1.0, trying to reach 5.0, but capped at 3.0
    bounded_param = prx.Parameter(1.0, bounds=jnp.array([0.0, 3.0]))
    bounded_model = DummyOptModel(val=bounded_param)
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    result = minimize(obj_fn, bounded_model, basic_freq, solver=ScipyMinimizer())
    
    # The optimizer should hit the upper bound and stop
    assert jnp.allclose(result.model.val, 3.0, atol=1e-3)

def test_minimize_optimistix(model, basic_freq):
    """Test optimization using a JAX-native Optimistix solver."""
    optx = pytest.importorskip("optimistix")
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    # Use a gradient-free JAX solver
    solver = optx.NelderMead(rtol=1e-5, atol=1e-5)
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iters=500)
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

# ---------------------------------------------------------
# `fit` Tests
# ---------------------------------------------------------

def test_fit_ndarray_target(model, basic_freq):
    """Test fitting directly to a JAX/Numpy array."""
    # We want the 1-port S-parameters to perfectly match 5.0 + 0j
    target_data = jnp.ones((basic_freq.npoints, 1, 1), dtype=complex) * 5.0
    
    # fit will automatically wrap the 's' feature and target data in a TargetLoss
    result = fit(model, target_data, basic_freq, features='s')
    
    assert jnp.allclose(result.model.val, 5.0, atol=1e-3)

def test_fit_missing_freq_error(model):
    """Ensure raw arrays throw an error if no frequency axis is provided."""
    target_data = jnp.ones((5, 1, 1))
    
    with pytest.raises(Exception, match="Frequency must be passed"):
        fit(model, target_data, frequency=None)

def test_fit_skrf_network(model, basic_freq):
    """Test fitting to a scikit-rf Network, verifying auto-extraction of frequency."""
    skrf = pytest.importorskip("skrf")
    import numpy as np
    
    # Create a mock target scikit-rf Network with S-params = 4.0
    skrf_freq = basic_freq.to_skrf()
    target_s = np.ones((basic_freq.npoints, 1, 1), dtype=complex) * 4.0
    ntwk = skrf.Network(frequency=skrf_freq, s=target_s, z0=50)
    
    # We omit the `frequency` argument here intentionally. `fit` should extract it!
    # Because 's' is the default feature, we don't strictly need to pass it, but we can.
    result = fit(model, ntwk, features='s')
    
    assert jnp.allclose(result.model.val, 4.0, atol=1e-3)