# tests/test_infer/test_infer.py
import pytest
import jax.numpy as jnp

from pmrf.core import Frequency, Model
from pmrf.infer.sample import sample
from pmrf.fitting import fit_sample
from pmrf.parameters import Param, Fixed, Free, free, bounded, Bounded, Uniform
from pmrf.fields import field, frozen

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
    val: Param = Uniform(0.0, 10.0, value=5.0)

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
    
    from pmrf.infer import PolyChord
    
    def log_like(m, f):
        return -0.5 * jnp.sum((m.val - 7.0)**2)
        
    result = sample(
        loglikelihood=log_like,
        model=infer_model,
        frequency=basic_freq,
        solver=PolyChord(
            nlive=5,
            do_clustering=False,
            num_repeats=1,
            precision_criterion=1.0,
            feedback=0,         
            write_resume=False, 
            write_live=False,
            write_dead=False,
            write_stats=False,
            base_dir=str(tmp_path)
        ),
    )
    
    assert isinstance(result.model, DummyInferModel)
    assert result.model.val.physical_value > 0.0 
    
    groups = result.model.param_groups()
    assert len(groups) > 0
    posterior_dist = groups[0].distribution
    assert isinstance(posterior_dist, distreqx.distributions.WeightedEmpirical)
    
    mean_val = posterior_dist.mean()
    assert mean_val.shape == (1,)
    assert not jnp.isnan(mean_val)
    
    batched_model = result.sampled_models
    assert isinstance(batched_model, DummyInferModel)
    n_samples = result.loglikelihood_values.shape[0]
    assert batched_model.val.physical_value.shape == (n_samples,)


def test_fit_polychord(infer_model, basic_freq, tmp_path): # <-- Add tmp_path here
    """
    Test the high-level fit_sample() wrapper using PolyChord.
    Ensures Feature extractors, Likelihoods, and Data coercion work.
    """
    pytest.importorskip("mpi4py")
    pytest.importorskip("pypolychord")
    pytest.importorskip("anesthetic")
    
    from pmrf.infer import PolyChord
    from pmrf.likelihoods import GaussianLikelihood
    
    target_data = 3.0 * jnp.ones(basic_freq.npoints).reshape(2, 1, 1)
    
    result = fit_sample(
        model=infer_model,
        data=target_data,
        frequency=basic_freq,
        solver=PolyChord(
            nlive=5,
            do_clustering=False,
            num_repeats=1,
            precision_criterion=1.0,
            feedback=0, 
            write_resume=False,
            write_live=False,
            write_dead=False,
            write_stats=False,
            base_dir=str(tmp_path)
        ),
        features='s_mag',
        likelihood=GaussianLikelihood(noise=1.0),
    )
    
    assert isinstance(result.model, DummyInferModel)
    n_samples = result.solution.loglikelihood_values.shape[0]
    assert result.solution.sampled_models.val.value.shape == (n_samples,)