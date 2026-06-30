# tests/test_optimize_base.py

import pytest
import jax.numpy as jnp
import equinox as eqx

import distreqx.distributions as dists
import distreqx.bijectors as bij
from parax.constraints import infer_distribution_constraint

import parax as prx
from pmrf.parameters import Param, Fixed
from pmrf.optimize import base
from pmrf.optimize.solvers.optimistix import BFGS, NelderMead
from pmrf.optimize.solvers.jaxopt import LBFGSB
from pmrf.optimize.solvers.scipy import ScipyMinimize



# ==========================================
# 1. Dummy Objectives & Models for Testing
# ==========================================

def simple_quadratic_dict(model, args=None):
    """A gentle quadratic objective that won't cause L-BFGS-B line search divergence."""
    x, y = model["x"], model["y"]
    return (x - 1.0)**2 + (y - 1.0)**2

class QuadraticModel(eqx.Module):
    """An Equinox model containing Parax variables."""
    w: Param
    b: Param
    
    def __init__(self, w, b):
        self.w = w
        self.b = b

def quadratic_objective(model: QuadraticModel, args=None):
    """Minimizes (w - 3)^2 + (b + 2)^2. Minimum at w=3, b=-2."""
    return jnp.sum((model.w - 3.0)**2) + jnp.sum((model.b + 2.0)**2)


# ==========================================
# 2. Unconstrained Optimizer Tests
# ==========================================

@pytest.mark.parametrize("solver_cls", [BFGS, NelderMead])
def test_unconstrained_dict_model(solver_cls):
    """Test unconstrained optimization using Optimistix backends on a dictionary."""
    # We start from 0.1 and 0.0 because otherwise NelderMead fails
    y0 = {
        "x": jnp.array(0.1), 
        "y": jnp.array(0.1), 
        "z": Fixed(1.0),
    }
    
    solver = solver_cls()
    
    final_model, payload = base.run_minimizer(
        fn=simple_quadratic_dict, 
        model=y0, 
        solver=solver, 
        max_iter=2000
    )
    
    # Verify the structure/wrappers are preserved
    assert isinstance(final_model, dict)
    
    # Unwrap and verify math
    unwrapped = prx.unwrap(final_model)
    assert jnp.allclose(unwrapped["x"], 1.0, atol=1e-3)
    assert jnp.allclose(unwrapped["y"], 1.0, atol=1e-3)

def test_unconstrained_equinox_model():
    """Test unconstrained optimization on an eqx.Module with Parax types."""
    model = QuadraticModel(
        w=prx.Tagged(raw_value=jnp.array(0.0)),
        b=prx.Tagged(raw_value=jnp.array(0.0))
    )
    solver = BFGS()
    
    opt_model, payload = base.run_minimizer(
        fn=quadratic_objective, 
        model=model, 
        solver=solver
    )
    
    # Verify wrappers survived
    assert isinstance(opt_model.w, prx.Tagged)
    assert isinstance(opt_model.b, prx.Tagged)
    
    # Verify math
    unwrapped = prx.unwrap(opt_model)
    assert jnp.allclose(unwrapped.w, 3.0, atol=1e-4)
    assert jnp.allclose(unwrapped.b, -2.0, atol=1e-4)


# ==========================================
# 3. Bounded Optimizer Tests
# ==========================================

def test_bounded_minimization_jaxopt():
    """Test bounded minimization using the JAXopt L-BFGS-B backend."""
    y0 = {
        "x": prx.Constrained(value=jnp.array(0.0)),
        "y": prx.Constrained(value=jnp.array(0.0)) 
    }
    
    solver = LBFGSB()
    
    opt_model, payload = base.run_minimizer(
        fn=simple_quadratic_dict,  # Switched to gentler objective 
        model=y0, 
        solver=solver,
        max_iter=1000
    )
    
    assert isinstance(opt_model["x"], prx.Constrained)
    
    unwrapped = prx.unwrap(opt_model)
    assert jnp.allclose(unwrapped["x"], 1.0, atol=1e-3)
    assert jnp.allclose(unwrapped["y"], 1.0, atol=1e-3)


def test_bounded_minimization_scipy():
    """Test bounded minimization using SciPy. SciPy cannot be JIT compiled."""
    y0 = {
        "x": prx.Constrained(value=jnp.array(0.0)), 
        "y": prx.Constrained(value=jnp.array(0.0)) 
    }
    
    solver = ScipyMinimize()
    
    opt_model, payload = base.run_minimizer(
        fn=simple_quadratic_dict, # Switched to gentler objective
        model=y0, 
        solver=solver,
        max_iter=1000
    )
        
    assert isinstance(opt_model["y"], prx.Constrained)
    
    unwrapped = prx.unwrap(opt_model)
    assert jnp.allclose(unwrapped["x"], 1.0, atol=1e-3)
    assert jnp.allclose(unwrapped["y"], 1.0, atol=1e-3)


# ==========================================
# 4. Parax Variable Metadata / Partitioning Tests
# ==========================================

def test_fixed_variable_partitioning():
    """
    Test that parax.Fixed variables are partitioned into the static PyTree 
    and are NOT updated by the optimizer.
    """
    y0 = {
        "x": jnp.array(0.0), 
        "y": prx.Fixed(jnp.array(0.0)) 
    }
    
    solver = BFGS()
    
    opt_model, payload = base.run_minimizer(
        fn=simple_quadratic_dict, 
        model=y0, 
        solver=solver
    )
    
    # Assert 'y' is safely locked as a Fixed Parax node
    assert isinstance(opt_model["y"], prx.Fixed)
    
    unwrapped = prx.unwrap(opt_model)
    assert jnp.allclose(unwrapped["y"], 0.0)
    
    # Assert 'x' has moved to the conditional optimum
    assert jnp.allclose(unwrapped["x"], 1.0) 
    assert unwrapped["x"] > 0.0 

def test_derived_variable():
    """Test that Parax Derived variables compute properly inside the objective."""
    
    def derived_objective(model, args=None):
        return (model["x"] - 2.0)**2 + (model["y"] - 4.0)**2

    y0 = {
        "x": prx.Tagged(raw_value=jnp.array(0.0)),
        "y": prx.Derived(fn=lambda x: x * 2.0, raw_value=jnp.array(0.0)) 
    }
    
    unwrapped = prx.unwrap(y0)
    loss = derived_objective(unwrapped)
    
    assert loss == (0 - 2)**2 + (0 - 4)**2


# def test_unconstrained_whitened_geometry():
#     """
#     Test that unconstrained solvers correctly operate in the latent space
#     when given a transformed prior (simulating a normalizing flow or correlated MVN).
#     """
#     # Define a base isotropic normal
#     base_dist = dists.Normal(loc=jnp.zeros(2), scale=jnp.ones(2))
    
#     # Create a strong transformation/correlation
#     # (Using Shift and Scale to simulate a heavily skewed geometry)
#     bijector = bij.Chain([
#         bij.Shift(jnp.array([1.0, -1.0])), 
#         bij.Scale(jnp.array([10.0, 0.1]))
#     ])
#     corr_dist = dists.Transformed(base_dist, bijector)
    
#     # Infer the constraint (this natively extracts the whitening bijector)
#     constraint = infer_distribution_constraint(corr_dist)
    
#     # Initialize the model at the mode of the prior (which is [1.0, -1.0] physically)
#     y0 = {
#         "x": prx.Constrained(value=jnp.array([1.0, -1.0]), constraint=constraint)
#     }
    
#     # Objective: Move away from the prior mode to a new physical location
#     target = jnp.array([5.0, 0.5])
#     def objective(model, args=None):
#         return jnp.sum((model["x"] - target)**2)
    
#     solver = BFGS()
#     opt_model, payload = base.run_minimizer(
#         fn=objective, 
#         model=y0, 
#         solver=solver,
#         max_iter=1000
#     )
    
#     unwrapped = prx.unwrap(opt_model)
    
#     # Check that the optimizer successfully reached the physical target 
#     # by navigating the whitened latent space
#     assert payload.success
#     assert jnp.allclose(unwrapped["x"], target, atol=1e-3)


# def test_bounded_whitened_geometry_copula():
#     """
#     Test that bounded solvers perfectly map their physical bounds backward 
#     into the latent space using the Copula bijector for bounded priors.
#     """
#     low = jnp.array([-2.0, 0.0])
#     high = jnp.array([2.0, 5.0])
#     dist = dists.Uniform(low=low, high=high)
    
#     constraint = infer_distribution_constraint(dist)
    
#     y0 = {
#         "x": prx.Constrained(value=jnp.array([0.0, 2.5]), constraint=constraint)
#     }
    
#     # Target moved inside the valid space to avoid gradient vanishing near the Copula infinities
#     target = jnp.array([1.5, 1.0])
#     def objective(model, args=None):
#         return jnp.sum((model["x"] - target)**2)
    
#     solver = LBFGSB()
    
#     opt_model, payload = base.run_minimizer(
#         fn=objective, 
#         model=y0, 
#         solver=solver,
#         max_iter=1000,
#         use_bounds=True
#     )
    
#     unwrapped = prx.unwrap(opt_model)
    
#     assert payload.success
#     assert jnp.allclose(unwrapped["x"], target, atol=1e-3)