# tests/test_optimize/test_optimize.py
import pytest
import jax.numpy as jnp

import pmrf as prf
from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.parameters import Bounded
from pmrf.optimize.minimize import minimize
from pmrf.optimize.backends.scipy import ScipyMinimize

# ---------------------------------------------------------
# Dummy Concrete Models for Testing
# ---------------------------------------------------------

class DummyOptModel(Model):
    """A simple 1-port model with one free parameter for optimization."""
    val: prf.Param = prf.param(1.0)

    def s(self, freq: Frequency) -> jnp.ndarray:
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
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    result = minimize(obj_fn, model, basic_freq, solver=ScipyMinimize())
    
    assert isinstance(result.model, DummyOptModel)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-3)

def test_minimize_scipy_bounded(basic_freq):
    """Test that parameter boundaries are successfully intercepted and enforced."""
    # Initialize parameter at 1.0, trying to reach 5.0, but capped at 3.0
    bounded_param = Bounded(0.0, 3.0, value=1.0)
    bounded_model = DummyOptModel(val=bounded_param)
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    result = minimize(obj_fn, bounded_model, basic_freq, solver=ScipyMinimize())
    
    # The optimizer should hit the upper bound and stop
    assert jnp.allclose(result.model.val, 3.0, atol=1e-3)

def test_minimize_nelder(model, basic_freq):
    """Test the Nelder-Mead solver"""
    optx = pytest.importorskip("optimistix")
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    # Use a gradient-free JAX solver
    solver = prf.optimize.NelderMead(xrtol=1e-5, xatol=1e-5)
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iter=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_bfgs(model, basic_freq):
    """Test the BFGS solver"""
    optx = pytest.importorskip("optimistix")
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    solver = prf.optimize.BFGS(step_rtol=1e-5, step_atol=1e-5)
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iter=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_lbfgsb(model, basic_freq):
    """Test the LBFGS-B solver"""
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    solver = prf.optimize.LBFGSB()
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iter=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_optimistix(model, basic_freq):
    """Test the optimistix wrapper"""
    optx = pytest.importorskip("optimistix")
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    solver = prf.optimize.OptimistixMinimise(solver=optx.BFGS(rtol=1e-5, atol=1e-5))
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iter=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_list_of_objectives(model, basic_freq):
    """Ensure that passing a list of callables automatically sums them."""
    # The minimum of (x-2)^2 + (x-4)^2 is exactly x=3
    obj1 = lambda m, f: jnp.sum((m.val - 2.0)**2)
    obj2 = lambda m, f: jnp.sum((m.val - 4.0)**2)
    
    result = minimize([obj1, obj2], model, basic_freq)
    assert jnp.allclose(result.model.val, 3.0, atol=1e-3)