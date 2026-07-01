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


def test_unconstrained_solver_with_univariate_constraint():
    """
    Test that an unconstrained solver (BFGS) respects univariate constraints
    by projecting through the Parax bijector API.
    """
    # Objective: Minimize (x - 2)^2. Global minimum is at x=2.
    def objective(model, args=None):
        return (model["x"] - 2.0)**2
    
    # Since the global minimum is 2, the constrained minimum should rest exactly at 5.0
    from parax.constraints import GreaterThan
    
    y0 = {
        "x": prx.Constrained(
            value=jnp.array(8.0), # Start in valid space
            constraint=GreaterThan(5.0) 
        )
    }
    
    solver = BFGS()
    
    opt_model, payload = base.run_minimizer(
        fn=objective, 
        model=y0, 
        solver=solver,
        max_iter=1000
    )
    assert payload.success
    assert isinstance(opt_model["x"], prx.Constrained)

    # The solver should hit the constraint boundary and stop at 5.0
    unwrapped = prx.unwrap(opt_model)
    assert jnp.allclose(unwrapped["x"], 5.0, atol=1e-3)

def test_unconstrained_whitened_geometry_correlated_starved():
    """
    Test that unconstrained solvers correctly operate in the latent space.
    
    Because the problem is perfectly whitened by the bijector, BFGS should
    solve this extremely ill-conditioned physical problem in just a few iterations.
    """
    # 1. Define a violently correlated Cholesky factor (Condition number ~ 10,000)
    L = jnp.array([
        [    1.0, 0.0],
        [10000.0, 1.0] 
    ])
    
    bijector = bij.Chain([
        bij.Shift(jnp.array([1.0, -1.0])), 
        bij.TriangularLinear(matrix=L) 
    ])
    
    # 2. Construct the unconstrained constraint 
    base_constraint = prx.constraints.RealLine(shape=(2,))
    constraint = prx.constraints.Transformed(constraint=base_constraint, bijector=bijector)
    
    # 3. Define Targets
    target_latent = jnp.array([5.0, 5.0])
    target_physical = bijector.forward(target_latent)
    
    y0_latent = jnp.array([-5.0, -5.0])
    y0_physical = bijector.forward(y0_latent)
    
    y0 = {
        "x": prx.Constrained(value=y0_physical, constraint=constraint)
    }
    
    # 4. Objective: Perfect sphere in LATENT space. 
    def objective(model, args=None):
        z = bijector.inverse(model["x"])
        return jnp.sum((z - target_latent)**2)
    
    solver = BFGS()
    opt_model, payload = base.run_minimizer(
        fn=objective, 
        model=y0, 
        solver=solver,
        # Starve the optimizer. If it operates in physical space, it fails.
        # If the latent projection works, it finishes in < 5 iterations.
        max_iter=15 
    )
    
    unwrapped = prx.unwrap(opt_model)
    
    # BFGS should easily pass this because of the latent projection
    assert payload.success
    assert jnp.allclose(unwrapped["x"], target_physical, atol=1e-2)

def test_bounded_whitened_geometry_correlated_starved():
    """
    Test that bounded solvers correctly operate in the latent space.
    
    CURRENTLY EXPECTED TO FAIL: The physical solver requires hundreds of 
    iterations to navigate the ill-conditioned ridge. A properly whitened 
    latent solver requires < 5 iterations.
    """
    # 1. Define a violently correlated Cholesky factor (Condition number ~ 10,000)
    L = jnp.array([
        [    1.0, 0.0],
        [10000.0, 1.0] 
    ])
    
    bijector = bij.Chain([
        bij.Shift(jnp.array([1.0, -1.0])), 
        bij.TriangularLinear(matrix=L) 
    ])
    
    # 2. Construct the bounded constraint 
    base_constraint = prx.constraints.Interval(
        lower=jnp.array([-100.0, -100.0]),
        upper=jnp.array([ 100.0,  100.0])
    )
    constraint = prx.constraints.Transformed(constraint=base_constraint, bijector=bijector)
    
    # 3. Define Targets
    target_latent = jnp.array([5.0, 5.0])
    target_physical = bijector.forward(target_latent)
    
    y0_latent = jnp.array([-5.0, -5.0])
    y0_physical = bijector.forward(y0_latent)
    
    y0 = {
        "x": prx.Constrained(value=y0_physical, constraint=constraint)
    }
    
    # 4. Objective: Perfect sphere in LATENT space. 
    def objective(model, args=None):
        z = bijector.inverse(model["x"])
        return jnp.sum((z - target_latent)**2)
    
    solver = LBFGSB()
    opt_model, payload = base.run_minimizer(
        fn=objective, 
        model=y0, 
        solver=solver,
        # Starve the optimizer. A whitened space solves this in ~3 iterations.
        # The physical space will require way more than 15.
        max_iter=15, 
        use_bounds=True
    )
    
    unwrapped = prx.unwrap(opt_model)
    
    # L-BFGS-B will hit the max_iter wall in physical space and fail
    assert payload.success
    assert jnp.allclose(unwrapped["x"], target_physical, atol=1e-2)