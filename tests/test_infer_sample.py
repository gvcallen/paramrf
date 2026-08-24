# tests/test_infer_sample.py

import importlib

import pytest
import jax
import jax.numpy as jnp

from pmrf.parameters import Param, Random
from pmrf.distributions import Normal
from pmrf.infer.sample import sample
from pmrf.infer.solvers.blackjax import NUTS
from pmrf.infer.result import InferResult
from pmrf.infer.base import SampleResult
from pmrf.models import Model
from pmrf.frequency import Frequency

# ==========================================
# 1. Fixtures & Objectives
# ==========================================

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=2.0, npoints=2, unit='GHz')

class DummyInferModel(Model):
    val: Param = Random(Normal(0.0, 5.0), value=0.0)

    def s(self, freq: Frequency) -> jnp.ndarray:
        nf = freq.npoints
        return jnp.ones((nf, 1, 1), dtype=complex) * self.val

@pytest.fixture
def infer_model():
    return DummyInferModel()

def simple_ll(model, freq):
    """A basic log-likelihood targeting val=2.0."""
    return jnp.sum(Normal(model.val, 0.5).log_prob(2.0))

def penalty_ll(model, freq):
    """A secondary log-likelihood penalty targeting val=0.0 to test lists."""
    return jnp.sum(Normal(model.val, 1.0).log_prob(0.0))

# ==========================================
# 2. Higher-Level Wrapper Tests
# ==========================================

def test_sample_wrapper_propagates_evidence(monkeypatch):
    """Evidence returned by a solver is preserved by the high-level API."""
    sample_module = importlib.import_module("pmrf.infer.sample")
    expected_logevidence = jnp.array(-1.25)
    expected_error = jnp.array(0.1)

    def fake_run_sampler(*, model, **kwargs):
        return model, SampleResult(
            samples=model,
            fn_values=jnp.array([0.0]),
            logevidence=expected_logevidence,
            logevidence_error=expected_error,
        )

    monkeypatch.setattr(sample_module, "run_sampler", fake_run_sampler)

    result = sample_module.sample(
        loglikelihood=lambda model: jnp.array(0.0),
        model={"value": jnp.array(0.0)},
        solver=object(),
    )

    assert result.logevidence == expected_logevidence
    assert result.logevidence_error == expected_error

def test_sample_wrapper_basic(infer_model, basic_freq):
    """Test the higher-level sample API with a single loglikelihood using NUTS."""
    key = jax.random.key(0)
    
    # Configure a fast NUTS execution
    solver = NUTS(num_warmup=10, show_progress=False)
    
    result = sample(
        loglikelihood=simple_ll,
        model=infer_model,
        frequency=basic_freq,
        solver=solver,
        key=key,
        max_steps=20
    )
    
    # Verify result type packaging
    assert isinstance(result, InferResult)
    
    # Verify batched dimensions for sampled payloads
    assert result.sampled_model.val.shape == (20,)
    assert result.fn_values.shape == (20,)
    
    # Verify MAP/MLE extraction (unbatched best_model extraction)
    assert result.best_model.val.ndim == 0


def test_sample_wrapper_list_loglikelihood(infer_model, basic_freq):
    """Test the sample wrapper's ability to sum a list of loglikelihood functions."""
    key = jax.random.key(0)
    
    solver = NUTS(num_warmup=5, show_progress=False)
    
    result = sample(
        loglikelihood=[simple_ll, penalty_ll],
        model=infer_model,
        frequency=basic_freq,
        solver=solver,
        key=key,
        max_steps=10
    )
    
    # Verify the structure successfully evaluated through the ex.Sum wrapping
    assert isinstance(result, InferResult)
    assert result.sampled_model.val.shape == (10,)
    assert result.best_model.val.ndim == 0
    assert result.fn_values.shape == (10,)
