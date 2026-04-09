# tests/test_evaluators/test_goals.py
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import parax as prx

from pmrf.core.frequency import Frequency
from pmrf.core.model import Model
from pmrf.evaluators import Feature, TargetLoss, MarginalLogLikelihood, Goal

# We use importorskip because distreqx and pmrf.losses are required for these tests
dist = pytest.importorskip("distreqx.distributions")
losses = pytest.importorskip("pmrf.losses")

# ---------------------------------------------------------
# Dummy Concrete Models for Testing
# ---------------------------------------------------------

class DummyEvalModel(Model):
    """A 2-port model returning a deterministic S-parameter matrix."""
    def s(self, freq: Frequency) -> jnp.ndarray:
        nf = freq.npoints
        # Matrix: [[1+0j, 2+0j], [3+0j, 4+0j]]
        mat = jnp.array([
            [1.0 + 0.0j, 2.0 + 0.0j],
            [3.0 + 0.0j, 4.0 + 0.0j]
        ])
        return jnp.tile(mat, (nf, 1, 1))
        
class ParentModel(Model):
    """A model containing a submodel to test nested attribute access."""
    amplifier: DummyEvalModel = DummyEvalModel()
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        return self.amplifier.s(freq)

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

@pytest.fixture
def model():
    return DummyEvalModel()

@pytest.fixture
def nested_model():
    return ParentModel()

# ---------------------------------------------------------
# Feature Extractor Tests
# ---------------------------------------------------------

def test_feature_regex_standard(model, basic_freq):
    """Test standard regex parsing for scattering parameters (e.g., s12_mag)."""
    # DummyEvalModel s_mag will just be the real parts since imag is 0
    # s12 is index [0, 1] which is 2.0
    feat = Feature('s12_mag')
    result = feat(model, basic_freq)
    
    assert result.shape == (5,)
    assert jnp.allclose(result, 2.0)

def test_feature_nested_path(nested_model, basic_freq):
    """Test that dot notation successfully drills into submodels."""
    feat = Feature('amplifier.s21_mag')
    result = feat(nested_model, basic_freq)
    
    # s21 is index [1, 0] which is 3.0
    assert result.shape == (5,)
    assert jnp.allclose(result, 3.0)

def test_feature_special_groups(model, basic_freq):
    """Test the gamma (diagonal) and tau (off-diagonal) special string routes."""
    # Gamma should extract the diagonals: [1.0, 4.0]
    gamma_feat = Feature('s_gamma')
    gamma_res = gamma_feat(model, basic_freq)
    assert gamma_res.shape == (5, 2)
    assert jnp.allclose(gamma_res[0], jnp.array([1.0+0j, 4.0+0j]))
    
    # Tau should extract off-diagonals: [2.0, 3.0]
    tau_feat = Feature('s_tau')
    tau_res = tau_feat(model, basic_freq)
    assert tau_res.shape == (5, 2)
    # The boolean mask flattens the off-diagonals
    assert jnp.allclose(tau_res[0], jnp.array([2.0+0j, 3.0+0j]))

def test_feature_sequence_stacking(model, basic_freq):
    """Test passing a list of strings creates a Stacked operator."""
    feat = Feature(['s11_mag', 's22_mag'])
    result = feat(model, basic_freq)
    
    # Should yield shape (5, 2) with 1.0 and 4.0 stacked
    assert result.shape == (5, 2)
    assert jnp.allclose(result[0, 0], 1.0)
    assert jnp.allclose(result[0, 1], 4.0)

def test_feature_invalid_alias():
    """Ensure malformed strings raise ValueError."""
    with pytest.raises(ValueError, match="Invalid feature alias format"):
        Feature('invalid_alias_format!')

# ---------------------------------------------------------
# TargetLoss Tests
# ---------------------------------------------------------

def test_target_loss(model, basic_freq):
    """Test the base capability of evaluating predictions against a target."""
    target_data = jnp.ones((5,)) * 5.0
    
    # Simple MSE loss lambda
    mse_loss = lambda t, p: jnp.mean((t - p) ** 2)
    
    # Predictor extracts 's22_mag' which equals 4.0
    evaluator = TargetLoss(
        predictor=Feature('s22_mag'), 
        target=target_data, 
        loss=mse_loss
    )
    
    # Loss should be mean((5.0 - 4.0)^2) = 1.0
    loss_val = evaluator(model, basic_freq)
    assert jnp.allclose(loss_val, 1.0)

# ---------------------------------------------------------
# Goal Tests
# ---------------------------------------------------------

def test_goal_hinge_loss(model, basic_freq):
    """Test Goal constructor wraps Feature and HingeLoss correctly."""
    # We want s11_mag (1.0) to be > 2.0.
    # The HingeLoss logic should penalize this gap.
    goal = Goal(
        feature='s11_mag',
        operator='>',
        target=2.0,
        weight=1.0,
        loss_fn=lambda t, p: (t - p) ** 2  # simple MSE base
    )
    
    # Because 1.0 is NOT > 2.0, there is a penalty of (2.0 - 1.0)^2 = 1.0
    # The sum across 5 frequencies should roughly reflect this based on multioutput setting
    loss_val = goal(model, basic_freq)
    
    # Assuming 'uniform_average' means mean() over the batch
    assert jnp.allclose(loss_val, 1.0)

def test_goal_met_zero_loss(model, basic_freq):
    """Ensure a satisfied goal returns exactly 0.0 penalty."""
    # s11_mag is 1.0. We want it to be < 2.0 (which is true).
    goal = Goal(
        feature='s11_mag',
        operator='<',
        target=2.0
    )
    loss_val = goal(model, basic_freq)
    assert jnp.allclose(loss_val, 0.0)

# ---------------------------------------------------------
# MarginalLogLikelihood Tests
# ---------------------------------------------------------

def test_marginal_log_likelihood(model, basic_freq):
    """Test probabilistic evaluation and the data/event mapping."""
    # We observe s11_mag data that is exactly 1.0 for all 5 points
    target_data = jnp.ones(5)
    
    # We define a standard normal distribution likelihood
    # Note: Using positional args explicitly for distreqx
    def likelihood_fn(pred):
        scale = jnp.ones_like(pred)
        return dist.Normal(pred, scale) 
        
    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        data=target_data,
        likelihood=likelihood_fn
    )
    
    # The prediction (s11_mag) is 1.0. 
    # Our data is 1.0. 
    # Normal(loc=1.0, scale=1.0).log_prob(1.0) = -0.91893853
    # Summed over 5 frequency points = -4.5946927
    
    log_prob = mll(model, basic_freq)
    expected = -0.91893853 * 5
    assert jnp.allclose(log_prob, expected)

def test_mll_complex_default_event_map(model, basic_freq):
    """Ensure the default event mapper handles complex matrices properly."""
    # Observe an S-matrix
    target_data = jnp.zeros((5, 2, 2), dtype=complex)
    
    def likelihood_fn(pred):
        return dist.Normal(pred, jnp.ones_like(pred))
        
    mll = MarginalLogLikelihood(
        predictor=Feature('s'),
        data=target_data,
        likelihood=likelihood_fn
    )
    
    # Predictor returns (5, 2, 2) complex. 
    # Mapper converts to (2, 2, 2, 5) -> 2 ports, 2 ports, 2 (real/imag), 5 freqs
    # The evaluation should run without shape errors.
    log_prob = mll(model, basic_freq)
    assert log_prob.ndim == 0  # It sums down to a scalar
    assert not jnp.isnan(log_prob)