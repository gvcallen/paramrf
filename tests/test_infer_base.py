# tests/test_infer_base.py

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from pmrf.parameters import Random, Fixed
from pmrf.distributions import Normal, Uniform
from pmrf.infer import base


# ==========================================
# 1. Dummy Objectives & Models for Testing
# ==========================================

@pytest.fixture
def dummy_model():
    """A standard dictionary model with random Parax variables."""
    return {
        "x": Random(Normal(0.0, 5.0), value=jnp.array(0.0)), 
        "y": Random(Uniform(0.0, 5.0), value=jnp.array(2.5)) ,
        "z": Fixed(1.0),
    }

def dummy_ll(model, args=None):
    """
    A simple Gaussian log-likelihood. 
    Target optimums are x=2.0 and y=3.0.
    """
    x, y = model["x"], model["y"]
    ll_x = Normal(x, 0.5).log_prob(2.0)
    ll_y = Normal(y, 0.5).log_prob(3.0)
    return jnp.sum(ll_x) + jnp.sum(ll_y)

def check_samples(x_samples, y_samples):
    # Posterior checks with generous tolerances
    x_mean = jnp.mean(x_samples)
    y_mean = jnp.mean(y_samples)
    np.testing.assert_allclose(x_mean, 2.0, rtol=0.15)
    np.testing.assert_allclose(y_mean, 3.0, rtol=0.15)
    
    # Ensure the sampler narrowed the uncertainty compared to the Prior (std=5.0)
    assert jnp.std(x_samples) < 2.5
    assert jnp.std(y_samples) < 2.5


# ==========================================
# 2. Joint Sampler (MCMC) Tests
# ==========================================

@pytest.mark.parametrize("solver_name", ["NUTS", "HMC"])
def test_joint_samplers(solver_name, dummy_model):
    """Test unconstrained joint sampling using BlackJAX NUTS and HMC."""
    blackjax_backend = pytest.importorskip("pmrf.infer.solvers.blackjax")
    solver_cls = getattr(blackjax_backend, solver_name)

    key = jax.random.key(0)
    
    solver = solver_cls(num_warmup=50, show_progress=False)
    max_steps = 100
    
    batched_model, results = base.run_sampler(
        loglikelihood_fn=dummy_ll,
        model=dummy_model,
        solver=solver,
        key=key,
        max_steps=max_steps
    )
    x_samples = batched_model["x"]
    y_samples = batched_model["y"]

    # Structural tests (ensure we don't get the warmup samples)
    assert isinstance(batched_model, dict)
    assert x_samples.shape == (max_steps,)
    assert y_samples.shape == (max_steps,)
    assert jnp.isscalar(batched_model["z"])

    check_samples(x_samples, y_samples)


# ==========================================
# 3. Split Sampler Tests
# ==========================================

def test_split_sampler_nss(dummy_model):
    """Test constrained split sampling using BlackJAX NSS."""
    blackjax_backend = pytest.importorskip("pmrf.infer.solvers.blackjax")
    NSS = blackjax_backend.NSS
 
    key = jax.random.key(0)
    
    # NSS Requires a batch of initial points (live points)
    init_samples = {
        "x": Random(Normal(0.0, 5.0), value=jax.random.normal(jax.random.key(1), (10,))),
        "y": Random(Uniform(0.0, 5.0), value=jax.random.uniform(jax.random.key(2), (10,)) * 4.9 + 0.1),
        "z": Fixed(1.0),
    }
    
    solver = NSS(num_delete=5, num_inner_steps=2, evidence_convergence=0.5, show_progress=False)
    max_steps = 50
    
    batched_model, results = base.run_sampler(
        loglikelihood_fn=dummy_ll,
        model=dummy_model,
        solver=solver,
        key=key,
        init_samples=init_samples,
        max_steps=max_steps
    )
    
    assert results.weights is not None

    x_samples = batched_model["x"]
    y_samples = batched_model["y"]

    check_samples(x_samples, y_samples)

    # Evidence check
    # Z_x: Convolution of prior N(0, 5^2) and likelihood N(2, 0.5^2)
    # Z_y: Uniform prior 1/5 integrated over contained Gaussian
    log_z_x = jax.scipy.stats.norm.logpdf(2.0, loc=0.0, scale=np.sqrt(5.0**2 + 0.5**2))
    log_z_y = jnp.log(1.0 / 5.0)
    expected_log_z = log_z_x + log_z_y
    estimated_log_z = results.logevidence
    np.testing.assert_allclose(estimated_log_z, expected_log_z, atol=1.0)



# ==========================================
# 4. Hypercube Sampler Tests
# ==========================================

def test_hypercube_polychord(tmp_path, dummy_model):
    """Test unit hypercube sampling using PolyChord."""
    polychord_backend = pytest.importorskip("pmrf.infer.solvers.polychord")
    
    if not getattr(polychord_backend, "MPI_AVAILABLE", False):
        pytest.skip("PolyChord, anesthetic, or mpi4py not installed.")
        
    PolyChord = polychord_backend.PolyChord

    # Run a tiny nested sampling instance
    solver = PolyChord(nlive=50, num_repeats=2, do_clustering=False, base_dir=str(tmp_path), seed=0)
    
    batched_model, results = base.run_sampler(
        loglikelihood_fn=dummy_ll,
        model=dummy_model,
        solver=solver,
        key=jax.random.key(0),
    )
    
    assert results.samples["x"].ndim == 1
    assert results.weights is not None

    x_samples = batched_model["x"]
    y_samples = batched_model["y"]

    check_samples(x_samples, y_samples)    

    # Evidence check
    # Z_x: Convolution of prior N(0, 5^2) and likelihood N(2, 0.5^2)
    # Z_y: Uniform prior 1/5 integrated over contained Gaussian
    log_z_x = jax.scipy.stats.norm.logpdf(2.0, loc=0.0, scale=np.sqrt(5.0**2 + 0.5**2))
    log_z_y = jnp.log(1.0 / 5.0)
    expected_log_z = log_z_x + log_z_y
    estimated_log_z = results.logevidence
    np.testing.assert_allclose(estimated_log_z, expected_log_z, atol=1.0)