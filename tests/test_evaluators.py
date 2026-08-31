# tests/test_evaluators/test_goals.py
import pytest
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import distreqx.bijectors as bij

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.covariance_kernels import RBFKernel
from pmrf.discrepancy_models import GaussianProcess
from pmrf.likelihoods import GaussianLikelihood
from pmrf.evaluators import (
    Feature, GibbsMarginalLogLikelihood, Goal, MarginalLogLikelihood, Negated,
    TargetLoss, _orthogonal_projection,
)
from tests._dependency_checks import requires_distreqx_transpose

dist = pytest.importorskip("distreqx.distributions")
losses = pytest.importorskip("pmrf.losses")

# ---------------------------------------------------------
# Dummy Concrete Models for Testing
# ---------------------------------------------------------

class DummyEvalModel(Model):
    """A 2-port model returning a deterministic S-parameter matrix."""
    def s(self, freq: Frequency) -> jnp.ndarray:
        nf = freq.npoints
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
    target_data = jnp.ones((5,)) * 6.0
    
    mse_loss = lambda t, p: jnp.mean((t - p) ** 2)
    
    # Predictor extracts 's22_mag' which equals 4.0
    evaluator = TargetLoss(
        predictor=Feature('s22_mag'), 
        target=target_data, 
        loss=mse_loss
    )
    
    # Loss should be mean((6.0 - 4.0)^2) = 4.0
    loss_val = evaluator(model, basic_freq)
    assert jnp.allclose(loss_val, 4.0)

# ---------------------------------------------------------
# Goal Tests
# ---------------------------------------------------------

def test_goal_hinge_loss(model, basic_freq):
    """Test Goal constructor wraps Feature and HingeLoss correctly."""
    # We want s11_mag (1.0) to be > 2.0.
    goal = Goal(
        feature='s11_mag',
        operator='>',
        target=3.0,
        weight=1.0,
        loss=lambda t, p: (t - p) ** 2,
        multioutput='uniform_average',
    )
    
    # Because 1.0 is NOT > 3.0, there is a penalty of (3.0 - 1.0)^2 = 4.0.
    # Since w are using uniform average the final loss value should be the same
    loss_val = goal(model, basic_freq)
    assert jnp.allclose(loss_val, 4.0)

def test_goal_met_zero_loss(model, basic_freq):
    """Ensure a satisfied goal returns exactly 0.0 penalty."""
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

@requires_distreqx_transpose
def test_marginal_loglikelihood(model, basic_freq):
    """Test probabilistic evaluation and the data/event mapping."""
    # We observe s11_mag data that is exactly 1.0 for all 5 points
    target_data = jnp.ones(5)
    
    # We define a standard normal distribution likelihood
    def likelihood_fn(pred):
        scale = jnp.ones_like(pred)
        return dist.Normal(pred, scale) 
        
    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=target_data,
        likelihood=likelihood_fn
    )
    
    # The prediction (s11_mag) is 1.0. 
    # Our data is 1.0. 
    # Normal(loc=1.0, scale=1.0).log_prob(1.0) = -0.91893853
    # Summed over 5 frequency points = -4.5946927
    log_prob = mll(model, basic_freq)
    expected = -0.91893853 * 5
    assert jnp.allclose(log_prob, expected)

@requires_distreqx_transpose
def test_mll_complex_default_event_map(model, basic_freq):
    """Ensure the default event mapper handles complex matrices properly."""
    # Observe an S-matrix
    target_data = jnp.zeros((5, 2, 2), dtype=complex)
    
    def likelihood_fn(pred):
        return dist.Normal(pred, jnp.ones_like(pred))
        
    mll = MarginalLogLikelihood(
        predictor=Feature('s'),
        observed=target_data,
        likelihood=likelihood_fn
    )
    
    # Predictor returns (5, 2, 2) complex. 
    # Mapper converts to (2, 2, 2, 5) -> 2 ports, 2 ports, 2 (real/imag), 5 freqs
    # The evaluation should run without shape errors.
    log_prob = mll(model, basic_freq)
    assert log_prob.ndim == 0  # It sums down to a scalar
    assert not jnp.isnan(log_prob)

# ---------------------------------------------------------
# Conditional (prediction-dependent) event transform tests
# ---------------------------------------------------------

class ScaledModel(Model):
    """A 1-port model whose S11 varies over frequency and with two parameters."""
    gain: jax.Array
    slopes: jax.Array

    def s(self, freq: Frequency) -> jnp.ndarray:
        f = jnp.asarray(freq.f_scaled)
        response = self.gain + self.slopes[0] * f + self.slopes[1] * f**2
        return response[:, None, None].astype(complex)


@pytest.fixture
def scaled_model():
    return ScaledModel(gain=jnp.array(2.0), slopes=jnp.array([0.5, 0.05]))


def _unit_normal_likelihood(pred):
    """A fixed unit-variance Normal likelihood over a deterministic prediction."""
    return dist.Normal(pred, jnp.ones_like(pred))


def test_conditional_transform_constant_matches_static(scaled_model, basic_freq):
    """
    (i) A conditional transform that ignores the prediction and returns a constant,
    volume-preserving bijector reproduces the static-transform result exactly.
    """
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)
    shift = 0.25

    static = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=_unit_normal_likelihood,
        event_transform=bij.Shift(shift),
    )
    conditional = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=_unit_normal_likelihood,
        event_transform=lambda y_pred: bij.Shift(shift),
    )

    assert not static.has_conditional_event_transform
    assert conditional.has_conditional_event_transform

    static_lp = static(scaled_model, basic_freq)
    conditional_lp = conditional(scaled_model, basic_freq)

    # Exact equality: |det J| = 1 for a shift, so no log-det term is contributed.
    assert conditional_lp == static_lp


def test_conditional_transform_adds_log_det(scaled_model, basic_freq):
    """
    A constant but *non* volume-preserving conditional transform reproduces the
    static result offset by exactly sum(log|det J|), evaluated at the observation.
    """
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)
    # distreqx reports the log-det with the shape of its own parameters, so an
    # array-shaped scale gives one contribution per element.
    n = basic_freq.npoints
    bijector = bij.ScalarAffine(shift=np.full((n,), 0.25), scale=np.full((n,), 2.0))

    static = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=_unit_normal_likelihood,
        event_transform=bijector,
    )
    conditional = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=_unit_normal_likelihood,
        event_transform=lambda y_pred: bijector,
    )

    expected_log_det = jnp.sum(bijector.forward_log_det_jacobian(observed))
    assert jnp.allclose(expected_log_det, n * jnp.log(2.0))

    static_lp = static(scaled_model, basic_freq)
    conditional_lp = conditional(scaled_model, basic_freq)
    assert jnp.allclose(conditional_lp, static_lp + expected_log_det)


def test_conditional_transform_is_applied_to_both(scaled_model, basic_freq):
    """
    The resolved transform is applied to prediction and observation alike, so a
    prediction-dependent shift produces the residual in the prediction's own frame.
    """
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)

    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=_unit_normal_likelihood,
        event_transform=lambda y_pred: bij.Shift(-y_pred),
    )

    # The prediction maps to exactly zero in event space.
    pred_dist = mll.predictive_distribution(scaled_model, basic_freq)
    assert jnp.allclose(pred_dist.mean(), 0.0)

    # And the log-prob is that of the residual under a unit normal centred at zero.
    y_pred = Feature('s11_mag')(scaled_model, basic_freq)
    residual = observed - y_pred
    expected = jnp.sum(-0.5 * residual**2 - 0.5 * jnp.log(2.0 * jnp.pi))
    assert jnp.allclose(mll(scaled_model, basic_freq), expected)


def test_conditional_transform_rejects_non_bijector(scaled_model, basic_freq):
    """A conditional transform must return an AbstractBijector."""
    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=jnp.ones(basic_freq.npoints),
        likelihood=_unit_normal_likelihood,
        event_transform=lambda y_pred: y_pred,
    )
    with pytest.raises(TypeError, match="AbstractBijector"):
        mll(scaled_model, basic_freq)


def test_conditional_transform_sample_observation(scaled_model, basic_freq):
    """
    `sample_observation` inverts the same resolved transform, so a sample from a
    residual-frame model lands back in observation space around the prediction.
    """
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)
    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=lambda pred: dist.Normal(pred, jnp.full_like(pred, 1e-8)),
        event_transform=lambda y_pred: bij.Shift(-y_pred),
    )

    sample = mll.sample_observation(jax.random.PRNGKey(0), scaled_model, basic_freq)
    y_pred = Feature('s11_mag')(scaled_model, basic_freq)

    # With a near-zero noise scale the inverted sample is the prediction itself.
    assert sample.shape == observed.shape
    assert jnp.allclose(sample, y_pred, atol=1e-6)


def test_conditional_transform_with_orthogonal_gp_discrepancy(scaled_model, basic_freq):
    """
    (B5) Discrepancy + orthogonal projection + conditional transform together.

    `use_orthogonal_discrepancy` needs no special handling: `event_fn` closes over
    the model, so the conditional transform's own dependence on the model is
    differentiated through when the projection is built.
    """
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)
    gp = GaussianProcess(kernel=RBFKernel(lengthscale=1.0), jitter=1e-8)

    # Non volume-preserving and prediction-dependent, so both the log-det term and
    # the projection's dependence on the transform are exercised.
    def conditional(y_pred):
        return bij.ScalarAffine(
            shift=jnp.zeros_like(y_pred),
            scale=jnp.full_like(y_pred, 1.0) / (1.0 + jnp.mean(y_pred**2)),
        )

    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=GaussianLikelihood(noise=jnp.array(0.1)),
        discrepancy=gp,
        use_orthogonal_discrepancy=True,
        event_transform=conditional,
    )

    log_prob = mll(scaled_model, basic_freq)
    assert log_prob.shape == ()
    assert jnp.isfinite(log_prob)

    # The projection must actually be applied: without it the covariance differs.
    mll_unprojected = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=GaussianLikelihood(noise=jnp.array(0.1)),
        discrepancy=gp,
        use_orthogonal_discrepancy=False,
        event_transform=conditional,
    )
    assert not jnp.allclose(log_prob, mll_unprojected(scaled_model, basic_freq))

    # It stays differentiable with respect to the model parameters.
    grad = jax.grad(lambda g: mll(eqx.tree_at(lambda m: m.gain, scaled_model, g), basic_freq))(
        jnp.asarray(2.0)
    )
    assert jnp.isfinite(grad)


def test_orthogonal_projection_densifies_array_parameter_leaf(scaled_model, basic_freq):
    """The length-two slopes leaf contributes two dense Jacobian columns."""
    event_fn = lambda model: Feature('s11_mag')(model, basic_freq)
    projection = _orthogonal_projection(event_fn, scaled_model)

    assert not hasattr(scaled_model, "func_jacobian")
    assert projection.shape == (basic_freq.npoints, basic_freq.npoints)
    assert jnp.linalg.matrix_rank(jnp.eye(basic_freq.npoints) - projection) == 3

    gp = GaussianProcess(kernel=RBFKernel(lengthscale=1.0), jitter=1e-8)
    mean = event_fn(scaled_model)
    projected_covariance = gp(
        mean, basic_freq.f_scaled, orthogonal_projection=projection
    ).covariance()
    unprojected_covariance = gp(mean, basic_freq.f_scaled).covariance()
    assert not jnp.allclose(projected_covariance, unprojected_covariance)


def test_orthogonal_projection_preserves_batches_and_rejects_static_model(
    scaled_model, model, basic_freq
):
    def batched_event_fn(candidate):
        event = Feature('s11_mag')(candidate, basic_freq)
        return jnp.stack((event, 2.0 * event))

    projection = _orthogonal_projection(batched_event_fn, scaled_model)
    assert projection.shape == (2, basic_freq.npoints, basic_freq.npoints)
    projection_T = jnp.swapaxes(projection, -1, -2)
    assert jnp.allclose(projection, projection_T, atol=1e-8)
    assert jnp.allclose(projection @ projection, projection, atol=1e-8)

    gp = GaussianProcess(kernel=RBFKernel(lengthscale=1.0), jitter=1e-8)
    batched_event = batched_event_fn(scaled_model)
    projected = gp(
        batched_event,
        basic_freq.f_scaled,
        orthogonal_projection=projection,
    )
    assert projected.covariance().shape == (
        2,
        basic_freq.npoints,
        basic_freq.npoints,
    )

    with pytest.raises(ValueError, match="at least one differentiable JAX-array leaf"):
        _orthogonal_projection(lambda candidate: candidate.s_mag(basic_freq), model)


def test_mll_batched_orthogonal_gp_discrepancy(scaled_model, basic_freq):
    def batched_predictor(candidate, frequency):
        prediction = Feature('s11_mag')(candidate, frequency)
        return jnp.stack((prediction, 2.0 * prediction))

    observed = jnp.ones((2, basic_freq.npoints))
    mll = MarginalLogLikelihood(
        predictor=batched_predictor,
        observed=observed,
        likelihood=GaussianLikelihood(noise=jnp.array(0.1)),
        discrepancy=GaussianProcess(
            kernel=RBFKernel(lengthscale=1.0), jitter=1e-8
        ),
        use_orthogonal_discrepancy=True,
        event_transform=bij.Shift(0.0),
    )

    log_prob = mll(scaled_model, basic_freq)
    assert log_prob.shape == ()
    assert jnp.isfinite(log_prob)


def test_gibbs_orthogonal_gp_discrepancy_uses_functional_derivative(
    scaled_model, basic_freq
):
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)
    gp = GaussianProcess(kernel=RBFKernel(lengthscale=1.0), jitter=1e-8)
    gibbs = GibbsMarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        loss=lambda target, prediction: jnp.mean((target - prediction) ** 2),
        discrepancy=gp,
        use_orthogonal_discrepancy=True,
        event_transform=bij.Shift(0.0),
    )

    value = gibbs(scaled_model, basic_freq)
    assert value.shape == ()
    assert jnp.isfinite(value)


# ---------------------------------------------------------
# Negated tests
# ---------------------------------------------------------

def test_negated_wraps_marginal_log_likelihood(scaled_model, basic_freq):
    """Negated returns exactly the negative of the wrapped evaluator."""
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)
    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=_unit_normal_likelihood,
        event_transform=bij.Shift(0.0),
    )
    assert Negated(mll)(scaled_model, basic_freq) == -mll(scaled_model, basic_freq)


def test_negated_accepts_any_evaluator(model, basic_freq):
    """Negated touches no likelihood-specific API, so it negates any evaluator."""
    target_loss = TargetLoss(
        predictor=Feature('s22_mag'),
        target=jnp.ones((5,)) * 6.0,
        loss=lambda t, p: jnp.mean((t - p) ** 2),
    )
    assert Negated(target_loss)(model, basic_freq) == -4.0
