# tests/test_infer/test_infer.py
import pytest
import jax.numpy as jnp
import parax as prx

from pmrf.core import Frequency, Model
from pmrf.infer.sample import sample
from pmrf.fitting import fit_sample

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

def test_sample_polychord(infer_model, basic_freq, tmp_path):
    """
    Test the lower-level sample() wrapper using PolyChord.
    Exercises the ICDF prior transformation, batched models, and posterior packing.
    """
    pytest.importorskip("mpi4py")
    pytest.importorskip("pypolychord")
    pytest.importorskip("anesthetic")
    distreqx = pytest.importorskip("distreqx")
    
    from inferix import PolyChord
    
    def log_like(m, f):
        return -0.5 * jnp.sum((m.val - 7.0)**2)
        
    result = sample(
        log_likelihood=log_like,
        model=infer_model,
        frequency=basic_freq,
        solver=PolyChord(do_clustering=False),
        nlive=5,
        num_repeats=1,
        precision_criterion=1.0,
        feedback=0,         
        write_resume=False, 
        write_live=False,
        write_dead=False,
        write_stats=False,
        base_dir=str(tmp_path)
    )
    
    assert isinstance(result.model, DummyInferModel)
    assert result.model.val.value > 0.0 
    
    groups = result.model.param_groups()
    assert len(groups) > 0
    posterior_dist = groups[0].distribution
    assert isinstance(posterior_dist, distreqx.distributions.WeightedEmpirical)
    
    mean_val = posterior_dist.mean()
    assert mean_val.shape == (1,)
    assert not jnp.isnan(mean_val)
    
    batched_model = result.sampled_models
    assert isinstance(batched_model, DummyInferModel)
    n_samples = result.log_likelihood_values.shape[0]
    assert batched_model.val.value.shape == (n_samples,)


def test_fit_polychord(infer_model, basic_freq, tmp_path): # <-- Add tmp_path here
    """
    Test the high-level fit_sample() wrapper using PolyChord.
    Ensures Feature extractors, Likelihoods, and Data coercion work.
    """
    pytest.importorskip("mpi4py")
    pytest.importorskip("pypolychord")
    pytest.importorskip("anesthetic")
    
    from inferix import PolyChord
    from pmrf.likelihoods import GaussianLikelihood
    
    target_data = 3.0 * jnp.ones(basic_freq.npoints).reshape(2, 1, 1)
    
    result = fit_sample(
        model=infer_model,
        data=target_data,
        frequency=basic_freq,
        solver=PolyChord(do_clustering=False),
        features='s_mag',
        likelihood=GaussianLikelihood(noise=1.0),
        nlive=5,
        num_repeats=1,
        precision_criterion=1.0,
        feedback=0, 
        write_resume=False,
        write_live=False,
        write_dead=False,
        write_stats=False,
        base_dir=str(tmp_path)  # <-- Force PolyChord to write to the temp directory!
    )
    
    assert isinstance(result.model, DummyInferModel)
    n_samples = result.solution.log_likelihood_values.shape[0]
    assert result.solution.sampled_models.val.value.shape == (n_samples,)