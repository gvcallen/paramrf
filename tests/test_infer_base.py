# tests/test_infer_base.py

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.parameters import Random, Fixed
from pmrf.distributions import Normal, Uniform
from pmrf.infer import base


# ==========================================
# 1. Dummy Objectives & Models for Testing
# ==========================================

def get_dummy_dict_model():
    """A standard dictionary model with random Parax variables."""
    return {
        "x": Random(Normal(0.0, 5.0), value=jnp.array(0.0)), 
        "y": Random(Uniform(0.0, 5.0), value=jnp.array(2.5)) ,
        "z": Fixed(1.0),
    }

def simple_loglikelihood(model, args=None):
    """
    A simple Gaussian log-likelihood. 
    Target optimums are x=2.0 and y=3.0.
    """
    x, y = model["x"], model["y"]
    ll_x = Normal(x, 0.5).log_prob(2.0)
    ll_y = Normal(y, 0.5).log_prob(3.0)
    return jnp.sum(ll_x) + jnp.sum(ll_y)


# ==========================================
# 2. Joint Sampler (MCMC) Tests
# ==========================================

@pytest.mark.parametrize("solver_name", ["NUTS", "HMC"])
def test_joint_samplers(solver_name):
    """Test unconstrained joint sampling using BlackJAX NUTS and HMC."""
    blackjax_backend = pytest.importorskip("pmrf.infer.backends.blackjax")
    solver_cls = getattr(blackjax_backend, solver_name)

    y0 = get_dummy_dict_model()
    key = jax.random.key(42)
    
    # Configure for a very fast execution
    solver = solver_cls(num_warmup=10)
    
    batched_model, payload, metrics = base.sample(
        loglikelihood_fn=simple_loglikelihood,
        model=y0,
        solver=solver,
        key=key,
        max_steps=20
    )
    
    # Verify the structure/wrappers are preserved across the batch
    assert isinstance(batched_model, dict)
    
    # Verify we got the requested number of samples
    assert payload.samples["x"].shape == (20,)
    assert payload.samples["y"].shape == (20,)


# ==========================================
# 3. Split Sampler Tests
# ==========================================

def test_split_sampler_nss():
    """Test constrained split sampling using BlackJAX NSS."""
    blackjax_backend = pytest.importorskip("pmrf.infer.backends.blackjax")
    NSS = blackjax_backend.NSS

    y0 = get_dummy_dict_model()
    key = jax.random.key(42)
    
    # NSS Requires a batch of initial points (live points)
    init_samples = {
        "x": Random(Normal(0.0, 5.0), value=jax.random.normal(jax.random.key(1), (10,))),
        "y": Random(Uniform(0.0, 5.0), value=jax.random.uniform(jax.random.key(2), (10,)) * 4.9 + 0.1),
        "z": Fixed(1.0),
    }
    
    solver = NSS(num_delete=5, num_inner_steps=2, evidence_convergence=0.5)
    max_steps = 10
    
    batched_model, payload, metrics = base.sample(
        loglikelihood_fn=simple_loglikelihood,
        model=y0,
        solver=solver,
        key=key,
        init_samples=init_samples,
        max_steps=max_steps
    )
    
    assert payload.weights is not None


# ==========================================
# 4. Hypercube Sampler Tests
# ==========================================

def test_hypercube_polychord(tmp_path):
    """Test unit hypercube sampling using PolyChord."""
    polychord_backend = pytest.importorskip("pmrf.infer.backends.polychord")
    
    if not getattr(polychord_backend, "MPI_AVAILABLE", False):
        pytest.skip("PolyChord, anesthetic, or mpi4py not installed.")
        
    PolyChord = polychord_backend.PolyChord

    y0 = get_dummy_dict_model()
    key = jax.random.key(42)
    
    # Run a tiny nested sampling instance
    solver = PolyChord(nlive=10, num_repeats=2, do_clustering=False, base_dir=str(tmp_path))
    
    batched_model, payload, metrics = base.sample(
        loglikelihood_fn=simple_loglikelihood,
        model=y0,
        solver=solver,
        key=key
    )
    
    assert payload.samples["x"].ndim == 1
    assert payload.weights is not None