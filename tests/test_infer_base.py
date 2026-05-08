# tests/test_infer_base.py

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

import parax as prx
from distreqx.distributions import Normal, Uniform

from pmrf.infer import base
from pmrf.infer.backends.blackjax import NUTS, HMC, NSS
from pmrf.infer.backends.polychord import PolyChord, MPI_AVAILABLE


# ==========================================
# 1. Dummy Objectives & Models for Testing
# ==========================================

def get_dummy_dict_model():
    """A standard dictionary model with random Parax variables."""
    return {
        "x": prx.Random(Normal(0.0, 5.0), raw_value=jnp.array(0.0)), 
        "y": prx.Random(Uniform(0.0, 5.0), raw_value=jnp.array(2.5)) 
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

@pytest.mark.parametrize("solver_cls", [NUTS, HMC])
def test_joint_samplers(solver_cls):
    """Test unconstrained joint sampling using BlackJAX NUTS and HMC."""
    y0 = get_dummy_dict_model()
    key = jax.random.key(42)
    
    # Configure for a very fast execution
    solver = solver_cls(num_warmup=10)
    
    final_model, payload, metrics = base.sample(
        loglikelihood_fn=simple_loglikelihood,
        y0=y0,
        solver=solver,
        key=key,
        max_steps=20
    )
    
    # Verify the structure/wrappers are preserved across the batch
    assert isinstance(final_model, dict)
    assert isinstance(final_model["x"], prx.Random)
    
    # Verify we got the requested number of samples
    assert payload.samples["x"].shape == (20,)
    assert payload.samples["y"].shape == (20,)


# ==========================================
# 3. Split Sampler Tests
# ==========================================

def test_split_sampler_nss():
    """Test constrained split sampling using BlackJAX NSS."""
    y0 = get_dummy_dict_model()
    key = jax.random.key(42)
    
    # NSS Requires a batch of initial points (live points)
    init_samples = {
        "x": prx.Random(Normal(0.0, 5.0), raw_value=jax.random.normal(jax.random.key(1), (10,))),
        "y": prx.Random(Uniform(0.0, 5.0), raw_value=jax.random.uniform(jax.random.key(2), (10,)) * 4.9 + 0.1)
    }
    
    solver = NSS(num_delete=1, num_inner_steps=2, logZ_convergence=0.5)
    
    final_model, payload, metrics = base.sample(
        loglikelihood_fn=simple_loglikelihood,
        y0=y0,
        solver=solver,
        key=key,
        init_samples=init_samples,
        max_steps=5
    )
    
    assert payload.samples["x"].shape[0] == 5
    assert payload.weights is not None
    assert "logZ" in metrics


# # ==========================================
# # 4. Hypercube Sampler Tests
# # ==========================================

# @pytest.mark.skipif(not MPI_AVAILABLE, reason="PolyChord, anesthetic, or mpi4py not installed.")
# def test_hypercube_polychord(tmp_path):
#     """Test unit hypercube sampling using PolyChord."""
#     y0 = get_dummy_dict_model()
#     key = jax.random.key(42)
    
#     # Run a tiny nested sampling instance
#     solver = PolyChord(nlive=10, num_repeats=2, do_clustering=False, base_dir=str(tmp_path))
    
#     final_model, payload, metrics = base.sample(
#         loglikelihood_fn=simple_loglikelihood,
#         y0=y0,
#         solver=solver,
#         key=key
#     )
    
#     assert payload.samples["x"].ndim == 1
#     assert payload.weights is not None
#     assert "logZ" in metrics


# # ==========================================
# # 5. Parax Variable Metadata / Bugfix Tests
# # ==========================================

# class MockHypercubeSampler(base.AbstractHypercubeSampler):
#     """A dummy sampler to test the hypercube wrapper logic without PolyChord."""
#     def sample(self, loglikelihood_fn, prior_transform_fn, y0, key, **kwargs):
#         # Generate a dummy hypercube coordinate (0.5 for all parameters)
#         u_cube = jax.tree.map(lambda x: jnp.array([0.5, 0.5]), y0)
        
#         # This will trigger the safe_icdf wrapper
#         c_params = prior_transform_fn(u_cube)
        
#         payload = base.SamplerPayload(
#             samples=c_params,
#             fn_values=jnp.array([-1.0, -1.0])
#         )
#         return payload, None


# def test_hypercube_frozen_variable_bypass():
#     """
#     Test that parax.Fixed variables safely bypass the .cdf and .icdf calls 
#     inside the AbstractHypercubeSampler wrapper.
#     """
#     y0 = {
#         "x": prx.Random(Normal(0.0, 5.0), raw_value=jnp.array(0.0)), 
#         "y": prx.Fixed(jnp.array(3.14159)) 
#     }
    
#     solver = MockHypercubeSampler()
    
#     final_model, payload, metrics = base.sample(
#         loglikelihood_fn=simple_loglikelihood,
#         y0=y0,
#         solver=solver,
#         key=jax.random.key(0)
#     )
    
#     # Assert 'y' survived as a Fixed Parax node
#     assert isinstance(final_model["y"], prx.Fixed)
    
#     # Extract the physical values via unwrapping
#     unwrapped = prx.unwrap(final_model)
    
#     # The Fixed value should have bypassed the transform completely and retained its physical value
#     assert jnp.allclose(unwrapped["y"], 3.14159)
    
#     # The Random value should have been transformed via icdf(0.5) = mean of Normal(0, 5) = 0.0
#     assert jnp.allclose(unwrapped["x"], 0.0)