"""The linearisation of a marginal likelihood and its posterior covariance (#262)."""
import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import parax as prx
import parax.bijectors as bij
import parax.distributions as dd
import pytest

import pmrf as prf
from pmrf.covariance_kernels import Matern52Kernel, gram
from pmrf.discrepancy_models import GaussianProcess
from pmrf.evaluators import MarginalLogLikelihood
from pmrf.likelihoods import GaussianLikelihood
from pmrf.linearization import posterior_covariance
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


def _mll(matrix, observed, noise=NOISE, discrepancy=True, **kwargs):
    return MarginalLogLikelihood(
        predictor=_linear_predictor(matrix),
        observed=observed,
        likelihood=GaussianLikelihood(noise=noise),
        discrepancy=GaussianProcess(_kernel(), jitter=JITTER) if discrepancy else None,
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
