# tests/test_infer/test_infer.py
import pytest
import jax.numpy as jnp
import parax as prx

from pmrf.core import Frequency, Model
from pmrf.infer.sample import sample
from pmrf.infer.condition import condition

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def basic_freq():
    # Very small frequency grid for fast likelihood evaluations
    return Frequency(start=1.0, stop=2.0, npoints=2, unit='GHz')

class DummyInferModel(Model):
    """
    A simple 1-port model for testing Bayesian inference.
    Crucially, parameters MUST have assigned distributions for the 
    Nested Sampler's `prior_transform_fn` (ICDF) to work.
    """
    val: prx.Parameter = prx.Uniform(0.0, 10.0, value=5.0)

    def s(self, freq: Frequency) -> jnp.ndarray:
        nf = freq.npoints
        return jnp.ones((nf, 1, 1), dtype=complex) * self.val

@pytest.fixture
def infer_model():
    return DummyInferModel()

# ---------------------------------------------------------
# PolyChord / Inference Tests
# ---------------------------------------------------------

def test_sample_polychord(infer_model, basic_freq):
    """
    Test the lower-level sample() wrapper using PolyChord.
    Exercises the ICDF prior transformation, batched models, and posterior packing.
    """
    pytest.importorskip("mpi4py")
    pytest.importorskip("pypolychord")
    pytest.importorskip("anesthetic")
    distreqx = pytest.importorskip("distreqx")
    
    from inferix import PolyChord
    
    # Define a simple log-likelihood: Gaussian centered at 7.0
    def log_like(m, f):
        return -0.5 * jnp.sum((m.val - 7.0)**2)
        
    result = sample(
        log_likelihood_fn=log_like,
        model=infer_model,
        frequency=basic_freq,
        solver=PolyChord(),
        nlive_factor=5,
        num_repeats=1,
        precision_criterion=1.0,
        feedback=0,         # Mute terminal output
        write_resume=False, # Don't litter the test directory
        write_live=False,
        write_dead=False,
        write_stats=False
    )
    
    # 1. Verify the Maximum Likelihood Estimate (MLE) model was extracted
    assert isinstance(result.model, DummyInferModel)
    assert result.model.val.value > 0.0 # Should have moved toward 7.0
    
    # 2. Verify the posterior distribution mechanics
    assert len(result.model.param_groups()) > 0
    posterior_dist = result.model.param_groups()[0].distribution
    assert isinstance(posterior_dist, distreqx.distributions.WeightedEmpirical)
    
    # Ensure the empirical distribution can be sampled or averaged
    mean_val = posterior_dist.mean()
    assert mean_val.shape == (1,)
    assert not jnp.isnan(mean_val)
    
    # 3. Verify the batched models were returned correctly
    batched_model = result.sampled_models
    assert isinstance(batched_model, DummyInferModel)
    
    # The internal arrays of the batched model should have a leading batch dimension
    n_samples = result.log_likelihoods.shape[0]
    assert batched_model.val.value.shape == (n_samples,)

def test_condition_polychord(infer_model, basic_freq):
    """
    Test the high-level condition() wrapper using PolyChord.
    Ensures Feature extractors, Likelihoods, and Data coercion work.
    """
    pytest.importorskip("mpi4py")
    pytest.importorskip("pypolychord")
    pytest.importorskip("anesthetic")
    
    from inferix import PolyChord
    from pmrf.likelihoods import GaussianLikelihood
    
    # Our target data: S-parameter magnitude is 3.0 at all frequencies
    target_data = jnp.ones(basic_freq.npoints) * 3.0
    
    result = condition(
        model=infer_model,
        data=target_data,
        frequency=basic_freq,
        solver=PolyChord(),
        features='s_mag',
        likelihood_fn=GaussianLikelihood(noise=1.0), # FIXED: was sigma=1.0
        nlive_factor=5,
        num_repeats=1,
        precision_criterion=1.0,
        feedback=0, # Mute terminal output
        write_resume=False,
        write_live=False,
        write_dead=False,
        write_stats=False
    )
    
    assert isinstance(result.model, DummyInferModel)
    
    # Check that the batched models are properly synced with the likelihoods
    n_samples = result.log_likelihoods.shape[0]
    assert result.sampled_models.val.value.shape == (n_samples,)