# tests/test_evaluators/test_goals.py
import pytest
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import parax.bijectors as bij
import pmrf as prf

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.covariance_kernels import (
    AutoCrossKernel, Matern52Kernel, PeriodicKernel, RBFKernel, SharedIndependentKernel,
    gram,
)
from pmrf.distributions import RelativeTruncatedNormal
from pmrf.discrepancy_models import GaussianProcess
from pmrf.likelihoods import GaussianLikelihood
from pmrf.evaluators import (
    Feature, GibbsMarginalLogLikelihood, Goal, MarginalLogLikelihood, Negated,
    TargetLoss, _orthogonal_projection,
)

import parax.distributions as dist
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


class RankDeficientModel(Model):
    coefficients: jax.Array

    def event(self, x):
        # The final two columns are identical, so J has rank two rather than three.
        return self.coefficients[0] + (
            self.coefficients[1] + self.coefficients[2]
        ) * x


class FreeFixedModel(Model):
    free_coefficient: object
    fixed_coefficient: object

    def event(self, x):
        return self.free_coefficient * x + self.fixed_coefficient * x**2


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
        orthogonal_rcond=1e-8,
        event_transform=conditional,
    ).with_orthogonal_reference(scaled_model, basic_freq)

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
    basis = _orthogonal_projection(event_fn, scaled_model, rcond=1e-8)
    projection = (
        jnp.eye(basic_freq.npoints)
        - basis.vectors @ jnp.swapaxes(basis.vectors, -1, -2)
    )

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

    basis = _orthogonal_projection(batched_event_fn, scaled_model, rcond=1e-8)
    projection = (
        jnp.eye(basic_freq.npoints)
        - basis.vectors @ jnp.swapaxes(basis.vectors, -1, -2)
    )
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

    with pytest.raises(ValueError, match="at least one free parameter"):
        _orthogonal_projection(
            lambda candidate: candidate.s_mag(basic_freq), model, rcond=1e-8
        )


def test_orthogonal_basis_uses_only_free_parameters():
    x = jnp.linspace(-1.0, 1.0, 7)
    model = FreeFixedModel(
        free_coefficient=prf.Unconstrained(2.0),
        fixed_coefficient=prf.Fixed(3.0),
    )
    basis = _orthogonal_projection(lambda candidate: candidate.event(x), model, rcond=1e-10)
    assert tuple(prf.params(model, free_only=True)) == ("free_coefficient",)
    assert jnp.sum(basis.mask) == 1


def test_orthogonal_block_density_matches_dense_rank_deficient_reference():
    x = jnp.linspace(-1.0, 1.0, 7)
    model = RankDeficientModel(coefficients=jnp.array([0.4, 0.2, -0.1]))
    basis = _orthogonal_projection(lambda candidate: candidate.event(x), model, rcond=1e-10)
    assert jnp.sum(basis.mask) == 2

    gp = GaussianProcess(kernel=RBFKernel(lengthscale=0.4), jitter=1e-10)
    mean = model.event(x)
    observed = jnp.sin(x)
    variance = jnp.asarray(0.07)
    block = gp.orthogonal_log_prob(mean, observed, x, variance, basis)

    Q1 = basis.vectors
    P = jnp.eye(x.size) - Q1 @ Q1.T
    K = gram(gp.kernel, x, jitter=gp.jitter)
    covariance = P @ K @ P.T + variance * jnp.eye(x.size)
    dense = dist.MultivariateNormalFullCovariance(mean, covariance).log_prob(observed)
    assert jnp.allclose(block, dense, rtol=1e-9, atol=1e-9)


def test_fixed_orthogonal_density_eager_jit_and_finite_difference(scaled_model, basic_freq):
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)
    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=GaussianLikelihood(noise=jnp.array(0.1)),
        discrepancy=GaussianProcess(RBFKernel(lengthscale=0.7), jitter=1e-10),
        use_orthogonal_discrepancy=True,
        orthogonal_rcond=1e-8,
        event_transform=bij.Shift(0.0),
    ).with_orthogonal_reference(scaled_model, basic_freq)

    def objective(gain):
        candidate = eqx.tree_at(lambda item: item.gain, scaled_model, gain)
        return mll(candidate, basic_freq)

    gain = scaled_model.gain
    eager = objective(gain)
    compiled = jax.jit(objective)(gain)
    assert eager == compiled

    automatic = jax.grad(objective)(gain)
    step = jnp.asarray(1e-5, dtype=gain.dtype)
    finite_difference = (objective(gain + step) - objective(gain - step)) / (2 * step)
    assert jnp.allclose(automatic, finite_difference, rtol=2e-5, atol=2e-5)


def test_recomputed_orthogonal_density_gradient_includes_basis_derivative(
    scaled_model, basic_freq
):
    observed = jnp.linspace(1.0, 3.0, basic_freq.npoints)

    def conditional(y_pred):
        return bij.ScalarAffine(
            shift=jnp.zeros_like(y_pred),
            scale=1.0 / (1.0 + jnp.mean(y_pred**2)),
        )

    mll = MarginalLogLikelihood(
        predictor=Feature('s11_mag'),
        observed=observed,
        likelihood=GaussianLikelihood(noise=jnp.array(0.1)),
        discrepancy=GaussianProcess(RBFKernel(lengthscale=0.7), jitter=1e-10),
        use_orthogonal_discrepancy=True,
        orthogonal_rcond=1e-8,
        orthogonal_recompute=True,
        event_transform=conditional,
    )

    def objective(gain):
        candidate = eqx.tree_at(lambda item: item.gain, scaled_model, gain)
        return mll(candidate, basic_freq)

    gain = scaled_model.gain
    automatic = jax.grad(objective)(gain)
    step = jnp.asarray(1e-5, dtype=gain.dtype)
    finite_difference = (objective(gain + step) - objective(gain - step)) / (2 * step)
    assert jnp.allclose(automatic, finite_difference, rtol=2e-4, atol=2e-4)


def test_mll_batched_orthogonal_gp_discrepancy(scaled_model, basic_freq):
    def batched_predictor(candidate, frequency):
        prediction = Feature('s11_mag')(candidate, frequency)
        return jnp.stack(
            (
                jnp.stack((prediction, 2.0 * prediction)),
                jnp.stack((3.0 * prediction, 4.0 * prediction)),
            )
        )

    def batched_kernel(x1, x2):
        squared_distance = jnp.sum((x1 - x2) ** 2)
        return jnp.exp(-jnp.array([0.5, 1.5]) * squared_distance)

    observed = jnp.ones((2, 2, basic_freq.npoints))
    mll = MarginalLogLikelihood(
        predictor=batched_predictor,
        observed=observed,
        likelihood=GaussianLikelihood(noise=jnp.array(0.1)),
        discrepancy=GaussianProcess(
            kernel=batched_kernel, jitter=1e-8
        ),
        use_orthogonal_discrepancy=True,
        orthogonal_rcond=1e-8,
        event_transform=bij.Shift(0.0),
    ).with_orthogonal_reference(scaled_model, basic_freq)

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
        orthogonal_rcond=1e-8,
        event_transform=bij.Shift(0.0),
    ).with_orthogonal_reference(scaled_model, basic_freq)

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


# ---------------------------------------------------------
# Closed-form Gaussian-process log-likelihood tests
# ---------------------------------------------------------


class _SlopedTwoPort(Model):
    """A two-port whose S-parameters vary over frequency with two parameters."""
    gain: jax.Array
    delay: jax.Array

    def s(self, freq: Frequency) -> jnp.ndarray:
        f = jnp.asarray(freq.f_scaled)
        ports = jnp.array([[0.1, 0.9], [0.8, 0.2]])
        response = self.gain * jnp.exp(-1j * self.delay * f) + 0.01 * f**2
        return response[:, None, None] * ports


def _sloped_two_port():
    return _SlopedTwoPort(gain=jnp.array(1.0), delay=jnp.array(0.3))


def _observed_two_port(frequency):
    truth = _SlopedTwoPort(gain=jnp.array(1.05), delay=jnp.array(0.32))
    f = jnp.asarray(frequency.f_scaled)
    wiggle = 0.02 * jnp.sin(1.7 * f)[:, None, None] * jnp.array([[1.0, 0.5], [0.5, 1.0]])
    return truth.s(frequency) + wiggle * (1.0 + 0.5j)


def _reference_log_prob(mll, model, frequency):
    """The log-likelihood through the predictive distribution objects."""
    obs_dist = mll.predictive_distribution(model, frequency)
    obs_event = mll.event_transform.forward(mll.observed)
    log_prob = lambda d, x: d.log_prob(x)
    for _ in range(obs_event.ndim - 1):
        log_prob = eqx.filter_vmap(log_prob)
    return jnp.sum(log_prob(obs_dist, obs_event))


def _cholesky_matrix_count(fn, *args):
    """The number of matrices factorized by Cholesky in the traced program of ``fn``."""
    def count(jaxpr):
        total = 0
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == 'cholesky':
                total += int(np.prod(eqn.invars[0].aval.shape[:-2]))
            for value in eqn.params.values():
                for sub in value if isinstance(value, (tuple, list)) else (value,):
                    if isinstance(sub, jax.extend.core.ClosedJaxpr):
                        total += count(sub.jaxpr)
                    elif isinstance(sub, jax.extend.core.Jaxpr):
                        total += count(sub)
        return total
    return count(eqx.filter_make_jaxpr(fn)(*args)[0].jaxpr)


def _gp_kernel(case, random=False):
    h = lambda v: prf.Random(RelativeTruncatedNormal(v, 0.1)) if random else v
    auto = lambda: Matern52Kernel(h(2.0)) * h(1e-2)
    cross = lambda: Matern52Kernel(h(1.0)) * h(3e-3)
    if case == 'unbatched':
        return auto()
    if case == 'batched':
        # A period per real/imaginary part, the last event batch axis.
        return PeriodicKernel(np.array([3.0, 5.0]), h(1.5)) * h(1e-2)
    if case == 'shared':
        return SharedIndependentKernel(auto())
    if case == 'auto_cross':
        # Routes along the trailing (port, Re/Im) axes, which is shape-valid.
        return AutoCrossKernel(auto(), cross(), num_outputs=2)
    if case == 'shared_auto_cross':
        return SharedIndependentKernel(AutoCrossKernel(auto(), cross(), num_outputs=2))
    raise ValueError(case)


_KERNELS = ['unbatched', 'batched', 'shared', 'auto_cross', 'shared_auto_cross']

_NOISES = {
    'scalar': lambda: 1e-3,
    'per_batch': lambda: np.linspace(5e-4, 2e-3, 8).reshape(2, 2, 2),
    'per_frequency': lambda: np.linspace(5e-4, 2e-3, 7),
}

_RANDOM_NOISES = {
    'scalar': lambda: prf.Random(RelativeTruncatedNormal(1e-3, 0.1)),
    'per_batch': lambda: prf.Random(RelativeTruncatedNormal(_NOISES['per_batch'](), 0.1)),
}


def _gp_mll(frequency, kernel, noise):
    return MarginalLogLikelihood(
        predictor=Feature('s'),
        observed=_observed_two_port(frequency),
        likelihood=GaussianLikelihood(noise=noise),
        discrepancy=GaussianProcess(kernel=kernel, jitter=1e-8),
    )


@pytest.fixture
def gp_freq():
    return Frequency(start=1.0, stop=10.0, npoints=7, unit='GHz')


@pytest.mark.parametrize('noise', ['scalar', 'per_batch'])
@pytest.mark.parametrize('kernel', _KERNELS)
def test_gp_log_likelihood_matches_distribution_path(gp_freq, kernel, noise):
    """Noise constant over frequency takes the closed form, which matches the distributions."""
    mll = _gp_mll(gp_freq, _gp_kernel(kernel), _NOISES[noise]())
    model = _sloped_two_port()
    expected = _reference_log_prob(mll, model, gp_freq)
    # Measured at 1.2e-15 relative or better across these cases.
    assert jnp.allclose(mll(model, gp_freq), expected, rtol=1e-10, atol=0.0)
    compiled = eqx.filter_jit(lambda e, m: e(m, gp_freq))(mll, model)
    assert jnp.allclose(compiled, expected, rtol=1e-10, atol=0.0)


def test_gp_log_likelihood_with_noise_varying_over_frequency(gp_freq):
    """Noise that varies along frequency falls back to the distribution path."""
    mll = _gp_mll(gp_freq, _gp_kernel('shared_auto_cross'), _NOISES['per_frequency']())
    model = _sloped_two_port()
    assert mll(model, gp_freq) == _reference_log_prob(mll, model, gp_freq)


@pytest.mark.parametrize('noise, kernel, matrices', [
    # K is (2, 2, 1): auto and cross for each port pair, shared across Re/Im.
    ('scalar', 'shared_auto_cross', 4),
    ('scalar', 'unbatched', 1),
    ('scalar', 'batched', 2),
    ('scalar', 'shared', 1),
    ('scalar', 'auto_cross', 4),
    # Noise varying across entries that share K needs one matrix per entry.
    ('per_batch', 'shared_auto_cross', 8),
])
def test_gp_log_likelihood_factorizes_smallest_broadcast_shape(gp_freq, noise, kernel, matrices):
    """Each distinct (kernel block, noise value) pair is factorized once."""
    mll = _gp_mll(gp_freq, _gp_kernel(kernel), _NOISES[noise]())
    assert _cholesky_matrix_count(lambda m: mll(m, gp_freq), _sloped_two_port()) == matrices


@pytest.mark.parametrize('noise', ['scalar', 'per_batch'])
@pytest.mark.parametrize('kernel', _KERNELS)
def test_gp_log_likelihood_gradients_match_distribution_path(gp_freq, kernel, noise):
    """Gradients wrt model, kernel hyperparameters and noise match under jit and grad."""
    mll = _gp_mll(gp_freq, _gp_kernel(kernel, random=True), _RANDOM_NOISES[noise]())
    model = _sloped_two_port()

    def closed_form(pair):
        evaluator, candidate = pair
        return evaluator(candidate, gp_freq)

    def reference(pair):
        evaluator, candidate = pair
        return _reference_log_prob(evaluator, candidate, gp_freq)

    value, grad = eqx.filter_jit(eqx.filter_value_and_grad(closed_form))((mll, model))
    expected_value, expected = eqx.filter_jit(eqx.filter_value_and_grad(reference))((mll, model))
    assert jnp.allclose(value, expected_value, rtol=1e-10, atol=0.0)

    # Every hyperparameter and noise parameter, and both model parameters, has a gradient.
    evaluator_grad, model_grad = grad
    raw_grads = [p.raw_value for p in prf.params(evaluator_grad).values() if prf.is_param(p)]
    assert len(raw_grads) == len(prf.params(mll))
    assert all(jnp.all(g != 0.0) for g in raw_grads + [model_grad.gain, model_grad.delay])

    leaves, expected_leaves = jax.tree.leaves(grad), jax.tree.leaves(expected)
    assert len(leaves) == len(expected_leaves)
    for actual, reference_leaf in zip(leaves, expected_leaves):
        # Measured at 2e-14 relative or better: summation-order roundoff between the two
        # backward passes. It grows with N (4e-9 at N = 1000 in #249).
        assert jnp.allclose(actual, reference_leaf, rtol=1e-10, atol=1e-14)


def test_gp_log_likelihood_gradient_under_vmap(gp_freq):
    """Per-gain gradients under vmap match the distribution path."""
    mll = _gp_mll(gp_freq, _gp_kernel('shared_auto_cross'), _NOISES['per_batch']())
    model = _sloped_two_port()

    def objective(gain, fn):
        return fn(eqx.tree_at(lambda m: m.gain, model, gain))

    gains = jnp.array([0.9, 1.0, 1.1])
    grad = jax.vmap(jax.grad(lambda g: objective(g, lambda m: mll(m, gp_freq))))(gains)
    expected = jax.vmap(jax.grad(
        lambda g: objective(g, lambda m: _reference_log_prob(mll, m, gp_freq))
    ))(gains)
    # Measured equal to the last bit.
    assert jnp.allclose(grad, expected, rtol=1e-10, atol=0.0)


def test_gp_log_likelihood_hyperparameter_gradient_matches_finite_difference(gp_freq):
    """The custom gradient through the kernel matches a central difference."""
    model = _sloped_two_port()

    def objective(lengthscale):
        kernel = SharedIndependentKernel(AutoCrossKernel(
            Matern52Kernel(lengthscale) * 1e-2,
            Matern52Kernel(1.0) * 3e-3,
            num_outputs=2,
        ))
        return _gp_mll(gp_freq, kernel, 1e-3)(model, gp_freq)

    lengthscale = jnp.asarray(2.0)
    automatic = jax.grad(objective)(lengthscale)
    step = 1e-5
    finite_difference = (objective(lengthscale + step) - objective(lengthscale - step)) / (2 * step)
    # Measured at 3e-10 relative with this step.
    assert jnp.allclose(automatic, finite_difference, rtol=1e-8, atol=0.0)


def test_gp_log_likelihood_random_hyperparameters_match_floats(gp_freq):
    """`Random` and float hyperparameters with equal values give identical values."""
    model = _sloped_two_port()
    value = eqx.filter_jit(lambda e: e(model, gp_freq))
    random = _gp_mll(gp_freq, _gp_kernel('shared_auto_cross', random=True), 1e-3)
    floats = _gp_mll(gp_freq, _gp_kernel('shared_auto_cross'), 1e-3)
    assert value(random) == value(floats)


# ---------------------------------------------------------
# Discrepancy prediction from a fitted marginal likelihood
# ---------------------------------------------------------


@pytest.fixture
def new_freq():
    """Off the fit grid, extending past both of its ends."""
    return Frequency(start=0.5, stop=11.0, npoints=9, unit='GHz')


def _hand_prediction(gp, transform, observed, model, frequency, new_frequency, noise):
    """#255's prediction called by hand on the event-space residual."""
    y_pred = Feature('s')(model, frequency)
    residual = transform.forward(observed) - transform.forward(y_pred)
    return gp.predict(residual, frequency.f_scaled, new_frequency.f_scaled, jnp.asarray(noise))


def _assert_same_prediction(actual, expected):
    # Same operations in the same order, so measured equal to the last bit.
    np.testing.assert_allclose(actual.mean(), expected.mean(), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(actual.covariance(), expected.covariance(), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize('noise', ['scalar', 'per_batch'])
@pytest.mark.parametrize('kernel', ['unbatched', 'shared', 'auto_cross', 'shared_auto_cross'])
def test_discrepancy_prediction_matches_gp_predict(gp_freq, new_freq, kernel, noise):
    """A complex two-port with the default event transform gives #255's prediction by hand."""
    mll = _gp_mll(gp_freq, _gp_kernel(kernel), _NOISES[noise]())
    model = _sloped_two_port()
    prediction = mll.predict_discrepancy(model, gp_freq, new_freq)
    assert prediction.mean().shape == (2, 2, 2, len(new_freq))
    expected = _hand_prediction(
        mll.discrepancy, mll.event_transform, mll.observed, model, gp_freq, new_freq,
        _NOISES[noise](),
    )
    _assert_same_prediction(prediction, expected)


def test_discrepancy_prediction_with_conditional_event_transform(gp_freq, new_freq):
    """A conditional transform is resolved from the prediction and applied to both sides."""
    base = _gp_mll(gp_freq, _gp_kernel('shared_auto_cross'), 1e-3).event_transform

    def conditional(y_pred):
        # A prediction-dependent shift, so the residual frame depends on the model.
        return bij.Chain([bij.Shift(jnp.mean(jnp.abs(y_pred))), base])

    gp = GaussianProcess(kernel=_gp_kernel('shared_auto_cross'), jitter=1e-8)
    mll = MarginalLogLikelihood(
        predictor=Feature('s'),
        observed=_observed_two_port(gp_freq),
        likelihood=GaussianLikelihood(noise=1e-3),
        discrepancy=gp,
        event_transform=conditional,
    )
    model = _sloped_two_port()
    resolved = conditional(Feature('s')(model, gp_freq))
    expected = _hand_prediction(gp, resolved, mll.observed, model, gp_freq, new_freq, 1e-3)
    _assert_same_prediction(mll.predict_discrepancy(model, gp_freq, new_freq), expected)


def test_discrepancy_prediction_converts_new_frequency_unit(gp_freq, new_freq):
    """A new frequency in another unit gives the prediction at the same points in the fit's unit."""
    mll = _gp_mll(gp_freq, _gp_kernel('shared_auto_cross'), 1e-3)
    model = _sloped_two_port()
    in_mhz = Frequency.from_f(new_freq.f_scaled * 1e3, unit='MHz')
    expected = mll.predict_discrepancy(model, gp_freq, new_freq)
    # Only the unit conversion's rounding differs; measured at 3e-16 relative.
    actual = mll.predict_discrepancy(model, gp_freq, in_mhz)
    np.testing.assert_allclose(actual.mean(), expected.mean(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(actual.covariance(), expected.covariance(), rtol=1e-12, atol=1e-15)


def test_discrepancy_prediction_from_fit_result(gp_freq, new_freq):
    """The evaluator recovered from `prf.fit` predicts with its optimized hyperparameters."""
    kernel = SharedIndependentKernel(AutoCrossKernel(
        Matern52Kernel(prf.Bounded(0.5, 5.0, value=2.0)) * prf.Bounded(1e-4, 1.0, value=1e-2),
        Matern52Kernel(prf.Bounded(0.5, 5.0, value=1.0)) * prf.Bounded(1e-4, 1.0, value=3e-3),
        num_outputs=2,
    ))
    model = _SlopedTwoPort(gain=prf.Bounded(0.5, 1.5, value=1.0), delay=prf.Fixed(0.3))
    result = prf.fitting.fit(
        model,
        _observed_two_port(gp_freq),
        frequency=gp_freq,
        likelihood=GaussianLikelihood(prf.Bounded(1e-6, 1e-1, value=1e-3)),
        discrepancy=GaussianProcess(kernel=kernel, jitter=1e-8),
    )
    (term,) = result.solution.objective
    mll = term.evaluator.evaluator
    fitted = prf.unwrap(mll)
    # The fit moved the hyperparameters, so the prediction uses the optimized ones.
    assert fitted.likelihood.noise != 1e-3

    prediction = mll.predict_discrepancy(result.model, result.frequency, new_freq)
    expected = _hand_prediction(
        fitted.discrepancy, fitted.event_transform, fitted.observed, prf.unwrap(result.model),
        gp_freq, new_freq, fitted.likelihood.noise,
    )
    _assert_same_prediction(prediction, expected)


def test_discrepancy_prediction_jit_and_grad_with_respect_to_model(gp_freq, new_freq):
    """jit and grad with respect to the model work through the prediction."""
    mll = _gp_mll(gp_freq, _gp_kernel('shared_auto_cross'), _NOISES['per_batch']())
    model = _sloped_two_port()

    def mean_sum(m):
        return jnp.sum(mll.predict_discrepancy(m, gp_freq, new_freq).mean())

    assert jnp.allclose(eqx.filter_jit(mean_sum)(model), mean_sum(model), rtol=1e-12, atol=0.0)
    grads = eqx.filter_jit(eqx.filter_grad(mean_sum))(model)
    automatic = jnp.array([grads.gain, grads.delay])
    step = 1e-6
    finite_difference = jnp.array([
        (mean_sum(eqx.tree_at(lambda m: m.gain, model, model.gain + step))
         - mean_sum(eqx.tree_at(lambda m: m.gain, model, model.gain - step))) / (2 * step),
        (mean_sum(eqx.tree_at(lambda m: m.delay, model, model.delay + step))
         - mean_sum(eqx.tree_at(lambda m: m.delay, model, model.delay - step))) / (2 * step),
    ])
    assert jnp.all(automatic != 0.0)
    # Measured at 1e-9 relative with this step.
    assert jnp.allclose(automatic, finite_difference, rtol=1e-7, atol=0.0)


def test_discrepancy_prediction_rejects_non_gp_discrepancy(gp_freq, new_freq):
    mll = MarginalLogLikelihood(
        predictor=Feature('s'),
        observed=_observed_two_port(gp_freq),
        likelihood=GaussianLikelihood(noise=1e-3),
    )
    with pytest.raises(TypeError, match="GaussianProcess"):
        mll.predict_discrepancy(_sloped_two_port(), gp_freq, new_freq)


def test_discrepancy_prediction_rejects_non_gaussian_likelihood(gp_freq, new_freq):
    mll = MarginalLogLikelihood(
        predictor=Feature('s'),
        observed=_observed_two_port(gp_freq),
        likelihood=_unit_normal_likelihood,
        discrepancy=GaussianProcess(kernel=_gp_kernel('shared_auto_cross'), jitter=1e-8),
    )
    with pytest.raises(TypeError, match="GaussianLikelihood"):
        mll.predict_discrepancy(_sloped_two_port(), gp_freq, new_freq)


def test_discrepancy_prediction_rejects_noise_varying_over_frequency(gp_freq, new_freq):
    mll = _gp_mll(gp_freq, _gp_kernel('shared_auto_cross'), _NOISES['per_frequency']())
    with pytest.raises(ValueError, match="constant along frequency"):
        mll.predict_discrepancy(_sloped_two_port(), gp_freq, new_freq)


def test_discrepancy_prediction_rejects_orthogonal_discrepancy(gp_freq, new_freq):
    mll = MarginalLogLikelihood(
        predictor=Feature('s'),
        observed=_observed_two_port(gp_freq),
        likelihood=GaussianLikelihood(noise=1e-3),
        discrepancy=GaussianProcess(kernel=_gp_kernel('shared_auto_cross'), jitter=1e-8),
        use_orthogonal_discrepancy=True,
        orthogonal_rcond=1e-10,
        orthogonal_recompute=True,
    )
    with pytest.raises(ValueError, match="orthogonal"):
        mll.predict_discrepancy(_sloped_two_port(), gp_freq, new_freq)
