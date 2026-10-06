# tests/test_discrepancy_models.py
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf  # noqa: F401  (enables jax_enable_x64)
from pmrf.covariance_kernels import (
    AutoCrossKernel,
    Matern52Kernel,
    RBFKernel,
    SharedIndependentKernel,
    gram,
)
from pmrf.discrepancy_models import GaussianProcess

#: Event batch: (port i, port j, Re/Im).
BATCH = (2, 2, 2)

AUTO_LENGTHSCALE, CROSS_LENGTHSCALE, CROSS_VARIANCE = 0.7, 0.4, 0.5


def _kernel(auto_lengthscale=AUTO_LENGTHSCALE, cross_lengthscale=CROSS_LENGTHSCALE):
    """Matérn 5/2 on the reflection blocks, a scaled RBF on the transmission blocks."""
    return SharedIndependentKernel(AutoCrossKernel(
        auto=Matern52Kernel(auto_lengthscale),
        cross=RBFKernel(cross_lengthscale) * CROSS_VARIANCE,
        num_outputs=2,
    ))


def _np_matern52(x1, x2, lengthscale):
    sq_dist = ((x1[:, None] - x2[None, :]) / lengthscale) ** 2
    # pmrf's Matérn 5/2 adds 1e-12 under the square root to keep gradients finite at
    # zero distance; mirror it so the comparison tests the formulas, not that guard.
    root5_dist = np.sqrt(5.0) * np.sqrt(sq_dist + 1e-12)
    return (1.0 + root5_dist + 5.0 / 3.0 * sq_dist) * np.exp(-root5_dist)


def _np_rbf(x1, x2, lengthscale):
    return np.exp(-0.5 * ((x1[:, None] - x2[None, :]) / lengthscale) ** 2)


def _np_block_kernel(i, j):
    if i == j:
        return lambda x1, x2: _np_matern52(x1, x2, AUTO_LENGTHSCALE)
    return lambda x1, x2: CROSS_VARIANCE * _np_rbf(x1, x2, CROSS_LENGTHSCALE)


def _np_prediction(residual, x_a, x_b, noise, jitter):
    """The two discrepancy prediction formulas in NumPy, block by block."""
    means = np.zeros(BATCH + (len(x_b),))
    covariances = np.zeros(BATCH + (len(x_b), len(x_b)))
    for index in np.ndindex(BATCH):
        k = _np_block_kernel(index[0], index[1])
        K_AA = k(x_a, x_a) + (jitter + noise[index]) * np.eye(len(x_a))
        K_BA = k(x_b, x_a)
        K_BB = k(x_b, x_b) + jitter * np.eye(len(x_b))
        means[index] = K_BA @ np.linalg.solve(K_AA, residual[index])
        covariances[index] = K_BB - K_BA @ np.linalg.solve(K_AA, K_BA.T)
    return means, covariances


def _cholesky_count(fn, n, *args):
    """The number of (n, n) matrices factorized by Cholesky in the traced program of ``fn``."""
    def count(jaxpr):
        total = 0
        for eqn in jaxpr.eqns:
            shape = eqn.invars[0].aval.shape if eqn.invars else ()
            if eqn.primitive.name == 'cholesky' and shape[-1] == n:
                total += int(np.prod(shape[:-2]))
            for value in eqn.params.values():
                for sub in value if isinstance(value, (tuple, list)) else (value,):
                    if isinstance(sub, jax.extend.core.ClosedJaxpr):
                        total += count(sub.jaxpr)
                    elif isinstance(sub, jax.extend.core.Jaxpr):
                        total += count(sub)
        return total
    return count(eqx.filter_make_jaxpr(fn)(*args)[0].jaxpr)


@pytest.fixture
def x_a():
    return np.linspace(0.0, 3.0, 9)


@pytest.fixture
def x_b():
    return np.linspace(-0.5, 3.5, 6)


@pytest.fixture
def residual(x_a):
    return np.random.default_rng(0).normal(scale=0.1, size=BATCH + (len(x_a),))


def _per_block_noise():
    return np.linspace(1e-3, 8e-3, 8).reshape(BATCH)


def test_prediction_matches_numpy_formulas(x_a, x_b, residual):
    """Mean and covariance of every block match the NumPy formulas with that block's kernel and noise."""
    noise = _per_block_noise()
    jitter = 1e-10
    prediction = GaussianProcess(_kernel(), jitter=jitter).predict(residual, x_a, x_b, noise)
    expected_mean, expected_covariance = _np_prediction(residual, x_a, x_b, noise, jitter)

    assert prediction.mean().shape == BATCH + (6,)
    # Measured at 9e-16 (mean) and 8e-16 (covariance) absolute.
    np.testing.assert_allclose(prediction.mean(), expected_mean, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(prediction.covariance(), expected_covariance, rtol=0.0, atol=1e-10)


def test_prediction_with_unbatched_kernel_and_scalar_noise(x_a, x_b, residual):
    """One kernel and one noise value shared by every block."""
    kernel = Matern52Kernel(AUTO_LENGTHSCALE)
    prediction = GaussianProcess(kernel, jitter=1e-10).predict(residual, x_a, x_b, 2e-3)
    K_AA = _np_matern52(x_a, x_a, AUTO_LENGTHSCALE) + (1e-10 + 2e-3) * np.eye(9)
    K_BA = _np_matern52(x_b, x_a, AUTO_LENGTHSCALE)
    K_BB = _np_matern52(x_b, x_b, AUTO_LENGTHSCALE) + 1e-10 * np.eye(6)
    # Measured at 1.1e-15 (mean) and 4e-16 (covariance) absolute.
    np.testing.assert_allclose(
        prediction.mean(), residual @ np.linalg.solve(K_AA, K_BA.T), rtol=0.0, atol=1e-10,
    )
    expected_covariance = K_BB - K_BA @ np.linalg.solve(K_AA, K_BA.T)
    np.testing.assert_allclose(
        prediction.covariance(), np.broadcast_to(expected_covariance, BATCH + (6, 6)),
        rtol=0.0, atol=1e-10,
    )


def test_prediction_at_fit_frequencies_without_noise_interpolates(x_a):
    """With x_B = x_A and no noise, the mean reproduces r and the covariance vanishes, up to jitter.

    With r = K w, the mean is r - jitter (K + jitter I)^{-1} K w, so it is within
    jitter |w| of r; the covariance's eigenvalues lie in [jitter, 2 jitter].
    """
    jitter = 1e-6
    w = np.random.default_rng(1).normal(size=BATCH + (len(x_a),))
    residual = np.einsum('...ij,...j->...i', np.broadcast_to(gram(_kernel(), x_a), BATCH + (9, 9)), w)
    prediction = GaussianProcess(_kernel(), jitter=jitter).predict(residual, x_a, x_a, 0.0)

    mean_error = np.linalg.norm(prediction.mean() - residual, axis=-1)
    # Measured at 0.999996 of the bound: K is well conditioned, so the bound is nearly
    # tight. The 1e-3 relative slack is for round-off only.
    assert np.all(mean_error <= (1 + 1e-3) * jitter * np.linalg.norm(w, axis=-1))
    eigenvalues = np.linalg.eigvalsh(prediction.covariance())
    # Measured within [1.99994, 2.0] jitter, since every eigenvalue of K is far above it.
    assert eigenvalues.min() >= 0.9 * jitter
    assert eigenvalues.max() <= 2.1 * jitter


def test_prediction_far_from_fit_frequencies_reverts_to_prior(x_a, residual):
    """Far from x_A relative to the length scales, the mean is 0 and the covariance K_BB."""
    jitter = 1e-10
    x_far = np.linspace(100.0, 101.0, 5)
    gp = GaussianProcess(_kernel(), jitter=jitter)
    prediction = gp.predict(residual, x_a, x_far, _per_block_noise())
    prior = np.broadcast_to(gram(_kernel(), x_far, jitter=jitter), BATCH + (5, 5))
    # K_BA is below 1e-130 here; measured at 6e-131 (mean) and exactly 0 (covariance).
    np.testing.assert_allclose(prediction.mean(), 0.0, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(prediction.covariance(), prior, rtol=0.0, atol=1e-10)


def test_prediction_jit_and_gradients_are_finite(x_a, x_b, residual):
    """jit and grad work through the prediction, w.r.t. hyperparameters, noise and residual."""
    target = np.random.default_rng(2).normal(scale=0.1, size=BATCH + (6,))

    def score(gp, noise, r):
        prediction = gp.predict(r, x_a, x_b, noise)
        log_prob = lambda d, y: d.log_prob(y)
        for _ in BATCH:
            log_prob = eqx.filter_vmap(log_prob)
        return jnp.sum(log_prob(prediction, target))

    kernel = _kernel(jnp.asarray(AUTO_LENGTHSCALE), jnp.asarray(CROSS_LENGTHSCALE))
    gp = GaussianProcess(kernel, jitter=1e-10)
    noise = jnp.asarray(_per_block_noise())
    value = eqx.filter_jit(score)(gp, noise, residual)
    assert jnp.allclose(value, score(gp, noise, residual), rtol=1e-12)

    grads = eqx.filter_jit(eqx.filter_grad(score))(gp, noise, residual)
    d_noise = eqx.filter_jit(jax.grad(score, argnums=1))(gp, noise, residual)
    d_residual = eqx.filter_jit(jax.grad(score, argnums=2))(gp, noise, jnp.asarray(residual))
    kernel_grads = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert len(kernel_grads) == 2
    for g in kernel_grads + [d_noise, d_residual]:
        assert jnp.all(jnp.isfinite(g))
        assert jnp.any(g != 0.0)


@pytest.mark.parametrize('noise, matrices', [
    # K is (2, 2, 1): auto and cross for each port pair, shared across Re/Im.
    (1e-3, 4),
    # Noise varying across entries that share K needs one matrix per entry.
    (_per_block_noise(), 8),
])
def test_prediction_factorizes_smallest_broadcast_shape(x_a, x_b, residual, noise, matrices):
    """K_AA + Σ_n is factorized once per distinct (kernel block, noise value) pair."""
    gp = GaussianProcess(_kernel(), jitter=1e-10)
    fn = lambda r: gp.predict(r, x_a, x_b, noise).mean()
    assert _cholesky_count(fn, len(x_a), residual) == matrices


def test_prediction_rejects_noise_that_does_not_broadcast(x_a, x_b, residual):
    """Noise with axes beyond the event batch is rejected, as in `log_prob`."""
    gp = GaussianProcess(_kernel(), jitter=1e-10)
    with pytest.raises(ValueError, match="do not broadcast"):
        gp.predict(residual, x_a, x_b, np.full((3, 1, 1, 1), 1e-3))
