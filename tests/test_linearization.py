"""The linearisation of a marginal likelihood and its posterior covariance (#262)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import parax as prx
import parax.bijectors as bij
import parax.distributions as dd
import pytest

import pmrf as prf
from pmrf.stats.covariance_kernels import Matern52Kernel, cross_gram, gram
from pmrf.stats.discrepancy_models import GaussianProcess
from pmrf.objectives.evaluators import MarginalLogLikelihood
from pmrf.stats.likelihoods import GaussianLikelihood
from pmrf.stats.linearization import posterior_covariance
from pmrf.models import Wrapped
from pmrf.models.base import Model

N = 12
P = 3
NOISE = 1e-6  # σ_n = 1e-3
JITTER = 1e-10
FREQ = prf.Frequency(1.0, 5.0, N, 'GHz')
RNG = np.random.default_rng(262)
#: One known Jacobian per event block: h_b = A_b θ.
A = RNG.normal(size=(2, N, P))
B = RNG.normal(size=(2, N, P))
MU0 = np.array([0.5, -1.0, 2.0])
SIGMA0 = np.array([[0.30, 0.05, 0.00], [0.05, 0.20, -0.04], [0.00, -0.04, 0.50]])


class _Linear(Model):
    """A predictor with no circuit: its parameters are read directly."""
    theta: jnp.ndarray
    delta: jnp.ndarray = None
    beta: jnp.ndarray = None
    gamma: jnp.ndarray = None

    def s(self, freq):
        return jnp.zeros((freq.npoints, 1, 1))


def _inner(model):
    model = prx.unwrap(model)
    return model.build() if isinstance(model, Wrapped) else model


def _linear_predictor(matrix):
    """h = A θ (+ δ), in observation space with frequency first."""
    def predictor(model, frequency):
        m = _inner(model)
        h = jnp.einsum('bnp,p->bn', matrix, m.theta)
        if m.delta is not None:
            h = h + m.delta
        return h.T
    return predictor


def _kernel():
    return Matern52Kernel(1.5) * 1e-2


def _theta_prior(model):
    return prf.prior(model, 'theta', dd.MultivariateNormalFullCovariance(jnp.asarray(MU0), jnp.asarray(SIGMA0)))


def _observed(matrix, seed):
    theta = np.array([0.7, -0.8, 1.6])
    rng = np.random.default_rng(seed)
    smooth = 0.05 * np.sin(np.linspace(0, 3, N))[None, :] * np.array([[1.0], [-0.5]])
    return (np.einsum('bnp,p->bn', matrix, theta) + smooth + 1e-3 * rng.normal(size=(2, N))).T


def _mll(matrix, observed, noise=NOISE, discrepancy=True, jitter=JITTER, **kwargs):
    return MarginalLogLikelihood(
        predictor=_linear_predictor(matrix),
        observed=observed,
        likelihood=GaussianLikelihood(noise=noise),
        discrepancy=GaussianProcess(_kernel(), jitter=jitter) if discrepancy else None,
        **kwargs,
    )


def _sigma_d(noise=NOISE):
    """Σ_D = K + Σ_n per block, as float64, shape (2, N, N)."""
    K = np.asarray(gram(_kernel(), FREQ.f_scaled, jitter=JITTER))
    noise = np.broadcast_to(np.asarray(noise, dtype=float), (2, N))
    return np.stack([K + np.diag(noise[b]) for b in range(2)])


def _mp_posterior(jacobians, sigma_ds, prior_precision):
    """(Σ_b J_bᵀ Σ_D,b⁻¹ J_b + Σ₀⁻¹)⁻¹ at 50 digits."""
    with mpmath.workdps(50):
        total = mpmath.matrix(prior_precision.tolist())
        for J, M in zip(jacobians, sigma_ds):
            J = mpmath.matrix(J.tolist())
            total += J.T * mpmath.inverse(mpmath.matrix(M.tolist())) * J
        result = mpmath.inverse(total)
        return np.array([[float(result[i, j]) for j in range(result.cols)] for i in range(result.rows)])


def _mp_map(jacobians, sigma_ds, observed):
    """(Σ_b J_bᵀ Σ_D,b⁻¹ J_b + Σ₀⁻¹)⁻¹ (Σ_b J_bᵀ Σ_D,b⁻¹ h̃_b + Σ₀⁻¹ μ₀) at 50 digits."""
    with mpmath.workdps(50):
        prior_precision = mpmath.inverse(mpmath.matrix(SIGMA0.tolist()))
        total = prior_precision
        rhs = prior_precision * mpmath.matrix(MU0.tolist())
        for J, M, h in zip(jacobians, sigma_ds, observed):
            J = mpmath.matrix(J.tolist())
            JtMinv = J.T * mpmath.inverse(mpmath.matrix(M.tolist()))
            total += JtMinv * J
            rhs += JtMinv * mpmath.matrix(h.tolist())
        result = mpmath.lu_solve(total, rhs)
        return np.array([float(result[i]) for i in range(result.rows)])


def _assert_close(actual, expected, tolerance):
    scale = np.abs(expected).max()
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.0, atol=tolerance * scale)


def test_linear_gaussian_posterior_covariance_is_exact():
    model = _theta_prior(_Linear(theta=prf.Unconstrained(jnp.zeros(P))))
    mll = _mll(A, _observed(A, 0))
    lin = mll.linearize(model, FREQ)
    assert lin.J.shape == (2, N, P)
    assert lin.residual.shape == (2, N)
    assert lin.names == ('theta',) and lin.shapes == ((P,),)
    np.testing.assert_allclose(lin.J, A, rtol=0.0, atol=1e-14)

    expected = _mp_posterior(A, _sigma_d(), np.linalg.inv(SIGMA0))
    # Measured at 1.4e-13 relative to the largest entry.
    _assert_close(posterior_covariance([lin], model), expected, 1e-10)


def test_posterior_covariance_of_several_datasets_is_that_of_the_stacked_problem():
    model = _theta_prior(_Linear(theta=prf.Unconstrained(jnp.zeros(P))))
    noise_b = np.linspace(5e-7, 2e-6, N)
    lin_a = _mll(A, _observed(A, 0)).linearize(model, FREQ)
    lin_b = _mll(B, _observed(B, 1), noise=noise_b).linearize(model, FREQ)

    jacobians = list(A) + list(B)
    sigma_ds = list(_sigma_d()) + list(_sigma_d(noise_b))
    expected = _mp_posterior(jacobians, sigma_ds, np.linalg.inv(SIGMA0))
    # Measured at 1.3e-13 relative to the largest entry.
    _assert_close(posterior_covariance([lin_a, lin_b], model), expected, 1e-10)


def _map_by_newton(mll, model):
    """The MAP of a linear-Gaussian posterior: one Newton step from `model`."""
    names = tuple(prf.values(model, free_only=True))
    lin = mll.linearize(model, FREQ)

    def log_posterior(x):
        candidate = prf.update(model, lin.unflatten(x))
        return mll(candidate, FREQ) + prf.log_prior(candidate)

    x0 = jnp.concatenate([jnp.ravel(prf.values(model)[name]) for name in names])
    return x0 + posterior_covariance([lin], model) @ jax.grad(log_posterior)(x0)


def test_marginal_map_equals_explicit_discrepancy_map():
    observed = _observed(A, 0)
    marginal = _map_by_newton(
        _mll(A, observed), _theta_prior(_Linear(theta=prf.Unconstrained(jnp.zeros(P))))
    )

    # δ as explicit parameters with prior N(0, K) per block, and no discrepancy.
    K = np.asarray(gram(_kernel(), FREQ.f_scaled, jitter=JITTER))
    explicit_model = _Linear(theta=prf.Unconstrained(jnp.zeros(P)), delta=prf.Unconstrained(jnp.zeros((2, N))))
    explicit_model = prf.prior(
        _theta_prior(explicit_model), 'delta',
        dd.MultivariateNormalFullCovariance(jnp.zeros(2 * N), jnp.asarray(np.kron(np.eye(2), K))),
    )
    explicit = _map_by_newton(_mll(A, observed, discrepancy=False), explicit_model)
    assert tuple(prf.values(explicit_model, free_only=True)) == ('theta', 'delta')
    # Measured at 1.5e-14 relative to the largest entry.
    _assert_close(explicit[:P], np.asarray(marginal), 1e-8)

    # Both against the closed-form MAP, independently of the code under test.
    expected = _mp_map(A, _sigma_d(), np.asarray(observed).T)
    # Measured at 6e-16 (marginal) and 1.6e-14 (explicit) relative to the largest entry.
    _assert_close(marginal, expected, 1e-8)
    _assert_close(explicit[:P], expected, 1e-8)


def test_jacobian_through_conditional_event_transform_matches_finite_differences():
    def predictor(model, frequency):
        m = _inner(model)
        return jnp.sin(jnp.einsum('bnp,p->bn', A, m.theta)).T

    def conditional(y_pred):
        # A prediction-dependent shift and scale, so the observation's event moves too.
        return bij.Chain([
            bij.ScalarAffine(jnp.mean(y_pred), 1.0 + jnp.mean(y_pred**2)), bij.Transpose((1, 0)),
        ])

    observed = np.asarray(_observed(A, 0))
    mll = MarginalLogLikelihood(
        predictor=predictor,
        observed=observed,
        likelihood=GaussianLikelihood(noise=NOISE),
        discrepancy=GaussianProcess(_kernel(), jitter=JITTER),
        event_transform=conditional,
    )
    model = _Linear(theta=prf.Unconstrained(jnp.array([0.3, -0.2, 0.4])))
    lin = mll.linearize(model, FREQ)

    # The prediction's own derivative misses the observation's dependence on θ.
    def prediction_event(theta):
        y = predictor(prf.update(model, {'theta': theta}), FREQ)
        return conditional(y).forward(y)
    assert not np.allclose(lin.J, jax.jacfwd(prediction_event)(model.theta.value), atol=1e-3)

    step = 1e-6
    finite_difference = np.stack([
        -(mll.linearize(prf.update(model, {'theta': model.theta.value + step * e}), FREQ).residual
          - mll.linearize(prf.update(model, {'theta': model.theta.value - step * e}), FREQ).residual)
        / (2 * step)
        for e in np.eye(P)
    ], axis=-1)
    # Measured at 1.9e-10 relative to the largest entry with this step.
    _assert_close(lin.J, finite_difference, 1e-7)


def test_parameter_without_prior_adds_no_prior_precision():
    def predictor(model, frequency):
        m = _inner(model)
        h = jnp.einsum('bnp,p->bn', A, m.theta) + m.beta * jnp.linspace(0, 1, N) + m.gamma
        return h.T

    model = _theta_prior(_Linear(
        theta=prf.Unconstrained(jnp.zeros(P)),
        beta=prf.Unconstrained(0.1),
        gamma=prf.Bounded(-5.0, 5.0, value=0.2),
    ))
    mll = MarginalLogLikelihood(
        predictor=predictor,
        observed=_observed(A, 0),
        # A larger noise keeps F comparable to the prior precision recovered below.
        likelihood=GaussianLikelihood(noise=1e-2),
    )
    lin = mll.linearize(model, FREQ)
    assert lin.names == ('theta', 'beta', 'gamma')
    precision = np.linalg.inv(np.asarray(posterior_covariance([lin], model))) - np.asarray(lin.F)
    scale = np.abs(np.asarray(lin.F)).max()
    # Measured at 3e-16 relative to the largest Fisher entry.
    np.testing.assert_allclose(precision[P:, :], 0.0, atol=1e-10 * scale)
    np.testing.assert_allclose(precision[:, P:], 0.0, atol=1e-10 * scale)
    np.testing.assert_allclose(precision[:P, :P], np.linalg.inv(SIGMA0), atol=1e-10 * scale)


@pytest.mark.parametrize('noise', [
    np.linspace(5e-7, 2e-6, N),                    # shared by the blocks
    np.linspace(5e-7, 2e-6, 2 * N).reshape(2, N),  # one factor per block
], ids=['shared', 'per_block'])
def test_without_discrepancy_sigma_d_is_the_noise(noise):
    model = _Linear(theta=prf.Unconstrained(jnp.zeros(P)))
    lin = _mll(A, _observed(A, 0), noise=noise, discrepancy=False).linearize(model, FREQ)
    assert lin.chol.shape == noise.shape[:-1] + (N, N)
    np.testing.assert_allclose(lin.chol, np.sqrt(noise)[..., None] * np.eye(N), rtol=1e-14)
    noise = np.broadcast_to(noise, (2, N))
    expected = sum(A[b].T @ np.diag(1 / noise[b]) @ A[b] for b in range(2))
    np.testing.assert_allclose(lin.F, expected, rtol=1e-12)


def test_orthogonal_discrepancy_raises():
    mll = _mll(
        A, _observed(A, 0),
        use_orthogonal_discrepancy=True, orthogonal_rcond=1e-10, orthogonal_recompute=True,
    )
    with pytest.raises(ValueError, match="orthogonal"):
        mll.linearize(_Linear(theta=prf.Unconstrained(jnp.zeros(P))), FREQ)


def test_posterior_covariance_rejects_another_space():
    model = _theta_prior(_Linear(theta=prf.Unconstrained(jnp.zeros(P))))
    lin = _mll(A, _observed(A, 0)).linearize(model, FREQ)
    with pytest.raises(ValueError, match="space"):
        posterior_covariance([lin], model, space='raw')


def _mp_condition(mean, covariance, observed, size):
    """Condition a stacked Gaussian's first `size` entries on all remaining entries."""
    with mpmath.workdps(50):
        mu = mpmath.matrix(np.asarray(mean).tolist())
        sigma = mpmath.matrix(np.asarray(covariance).tolist())
        cross = sigma[:size, size:]
        solve = cross * mpmath.inverse(sigma[size:, size:])
        conditional_mean = mu[:size, :] + solve * (mpmath.matrix(np.asarray(observed).tolist()) - mu[size:, :])
        conditional_covariance = sigma[:size, :size] - solve * cross.T
        return (
            np.array(conditional_mean, dtype=float).reshape(-1),
            np.array(conditional_covariance.tolist(), dtype=float),
        )


def _joint_reference(new_frequency, observed):
    """The generative covariance of [θ; δ(x_*); h], independently of linearisation."""
    K = np.kron(np.eye(2), np.asarray(gram(_kernel(), FREQ.f_scaled, jitter=JITTER)))
    K_star = np.kron(np.eye(2), np.asarray(gram(_kernel(), new_frequency.f / FREQ.multiplier, jitter=JITTER)))
    K_cross = np.kron(np.eye(2), np.asarray(cross_gram(_kernel(), new_frequency.f / FREQ.multiplier, FREQ.f_scaled)))
    H = A.reshape(2 * N, P)
    size = P + K_star.shape[0]
    covariance = np.block([
        [SIGMA0, np.zeros((P, K_star.shape[0])), SIGMA0 @ H.T],
        [np.zeros((K_star.shape[0], P)), K_star, K_cross],
        [H @ SIGMA0, K_cross.T, H @ SIGMA0 @ H.T + K + NOISE * np.eye(2 * N)],
    ])
    mean = np.concatenate([MU0, np.zeros(K_star.shape[0]), H @ MU0])
    return _mp_condition(mean, covariance, np.asarray(observed).T.reshape(-1), size)


@pytest.mark.parametrize('new_frequency', [FREQ, prf.Frequency(900, 5300, 7, 'MHz')], ids=['fit_grid', 'new_grid_unit'])
def test_joint_prediction_equals_dense_gaussian_conditioning(new_frequency):
    observed = _observed(A, 0)
    theta_map = _mp_map(A, _sigma_d(), np.asarray(observed).T)
    model = _theta_prior(_Linear(theta=prf.Unconstrained(jnp.asarray(theta_map))))
    mll = _mll(A, observed)
    names, joint = mll.predict_joint(model, FREQ, new_frequency)
    assert names == ('theta',)
    expected_mean, expected_covariance = _joint_reference(new_frequency, observed)
    _assert_close(joint.mean(), expected_mean, 1e-10)
    _assert_close(joint.covariance(), expected_covariance, 1e-10)


@pytest.mark.parametrize('noise', [NOISE, np.linspace(5e-7, 2e-6, 2 * N).reshape(2, N)], ids=['shared', 'per_block'])
def test_joint_prediction_parameter_term_and_supplied_covariance(noise):
    model = _theta_prior(_Linear(theta=prf.Unconstrained(jnp.asarray(MU0))))
    mll = _mll(A, _observed(A, 0), noise=noise)
    new_frequency = prf.Frequency(0.9, 5.3, 7, 'GHz')
    lin = mll.linearize(model, FREQ)
    # A second independent fit informs θ, while δ remains conditioned on this fit.
    covariance = posterior_covariance([lin, _mll(B, _observed(B, 1)).linearize(model, FREQ)], model)
    _, joint = mll.predict_joint(model, FREQ, new_frequency, covariance=covariance)
    np.testing.assert_array_equal(joint.covariance()[:P, :P], covariance)
    discrepancy = mll.predict_discrepancy(model, FREQ, new_frequency)
    cross = np.asarray(joint.covariance()[P:, :P])
    parameter_term = cross @ np.linalg.solve(np.asarray(covariance), cross.T)
    remaining = np.asarray(joint.covariance()[P:, P:]) - parameter_term
    for block in range(2):
        sl = slice(block * 7, (block + 1) * 7)
        _assert_close(remaining[sl, sl], np.asarray(discrepancy.covariance())[block], 1e-12)
    _assert_close(joint.mean()[P:], np.asarray(discrepancy.mean()).reshape(-1), 1e-12)
    np.testing.assert_allclose(remaining[:7, 7:], 0.0, atol=1e-15)


def test_joint_prediction_as_transfer_prior_matches_joint_fit():
    observed_a = _observed(A, 0)
    # No nugget: A and B observe the same δ on the same grid. Prediction jitter
    # otherwise represents distinct independent nuggets at the two evaluations.
    mll_a = _mll(A, observed_a, jitter=0.0)
    model_a = _theta_prior(_Linear(theta=prf.Unconstrained(jnp.zeros(P))))
    theta_map = _map_by_newton(mll_a, model_a)
    model_a = prf.update(model_a, {'theta': theta_map})
    names, joint = mll_a.predict_joint(model_a, FREQ, FREQ)
    transfer = _Linear(
        theta=prf.Unconstrained(jnp.zeros(P)),
        delta=prf.Unconstrained(jnp.zeros((2, N))),
    )
    transfer = prf.prior(transfer, [*names, 'delta'], joint)
    # The identity hand-off uses the joint mean's values in the original shapes.
    transfer = prf.update(transfer, {'theta': joint.mean()[:P], 'delta': joint.mean()[P:].reshape(2, N)})
    np.testing.assert_array_equal(prf.values(transfer)['delta'].reshape(-1), joint.mean()[P:])

    fixed_map = np.eye(2 * N) + 0.1 * np.random.default_rng(263).normal(size=(2 * N, 2 * N))
    observed_b = fixed_map @ np.asarray(observed_a).T.reshape(-1) + np.linspace(-0.01, 0.01, 2 * N)
    noise_b = 3e-6

    def transfer_predictor(model, frequency):
        corrected = _linear_predictor(A)(model, frequency).T.reshape(-1)
        return (fixed_map @ corrected).reshape(2, N).T

    mll_b = MarginalLogLikelihood(
        predictor=transfer_predictor,
        observed=observed_b.reshape(2, N).T,
        likelihood=GaussianLikelihood(noise=noise_b),
    )
    transfer_map = _map_by_newton(mll_b, transfer)
    transfer_covariance = posterior_covariance([mll_b.linearize(transfer, FREQ)], transfer)

    # Condition the original generative prior on A ∪ B in one dense solve.
    K = np.kron(np.eye(2), np.asarray(gram(_kernel(), FREQ.f_scaled)))
    prior = np.block([[SIGMA0, np.zeros((P, 2 * N))], [np.zeros((2 * N, P)), K]])
    prior_mean = np.concatenate([MU0, np.zeros(2 * N)])
    direct = np.concatenate([A.reshape(2 * N, P), np.eye(2 * N)], axis=1)
    observation_map = np.concatenate([direct, fixed_map @ direct])
    noise = np.diag(np.concatenate([np.full(2 * N, NOISE), np.full(2 * N, noise_b)]))
    covariance = np.block([
        [prior, prior @ observation_map.T],
        [observation_map @ prior, observation_map @ prior @ observation_map.T + noise],
    ])
    mean = np.concatenate([prior_mean, observation_map @ prior_mean])
    expected_mean, expected_covariance = _mp_condition(
        mean, covariance, np.concatenate([np.asarray(observed_a).T.reshape(-1), observed_b]), P + 2 * N,
    )
    _assert_close(transfer_map, expected_mean, 1e-8)
    _assert_close(transfer_covariance, expected_covariance, 1e-8)


@pytest.mark.parametrize('unsupported', ['no_gp', 'non_gaussian', 'orthogonal'])
def test_joint_prediction_rejects_unsupported_combinations(unsupported):
    model = _theta_prior(_Linear(theta=prf.Unconstrained(jnp.zeros(P))))
    kwargs = {}
    if unsupported == 'no_gp':
        kwargs['discrepancy'] = False
    elif unsupported == 'orthogonal':
        kwargs['use_orthogonal_discrepancy'] = True
        kwargs['orthogonal_rcond'] = 1e-10
    mll = _mll(A, _observed(A, 0), **kwargs)
    if unsupported == 'non_gaussian':
        mll = eqx.tree_at(lambda evaluator: evaluator.likelihood, mll, lambda event: dd.Normal(event, 1.0))
    exception, match = (ValueError, 'orthogonal') if unsupported == 'orthogonal' else (TypeError, 'Gaussian')
    with pytest.raises(exception, match=match):
        mll.predict_joint(model, FREQ, FREQ)
