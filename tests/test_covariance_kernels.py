# tests/test_covariance_kernels.py
import pytest
import equinox as eqx
import jax
import jax.numpy as jnp

import pmrf as prf

from pmrf.stats.covariance_kernels import (
    cross_gram,
    gram,
    RBFKernel,
    PeriodicKernel,
    Matern32Kernel,
    Matern52Kernel,
    ConstantKernel,
    CosineKernel,
    AutoCrossKernel,
    SharedIndependentKernel,
)
from pmrf.stats.discrepancy_models import GaussianProcess
from pmrf.stats.distributions import RelativeTruncatedNormal
from pmrf.objectives.evaluators import Feature, MarginalLogLikelihood
from pmrf.frequency import Frequency
from pmrf.stats.likelihoods import GaussianLikelihood
from pmrf.models.base import Model
from pmrf.utils import unwrap


@pytest.fixture
def x():
    return jnp.linspace(0.0, 2.0, 6)


@pytest.fixture
def x_periods():
    """Points spanning several periods of the cosine kernels below."""
    return jnp.linspace(0.0, 50.0, 200)


def _damped_cosine():
    return CosineKernel(period=7.3) * Matern52Kernel(lengthscale=20.0) * 2.0


def _gram_reference(kernel, x, jitter):
    """The covariance construction as it was written inline in GaussianProcess."""
    x_feat = x[:, None]
    inner_vmap = jax.vmap(kernel, in_axes=(None, 0), out_axes=-1)
    outer_vmap = jax.vmap(inner_vmap, in_axes=(0, None), out_axes=-2)
    K = outer_vmap(x_feat, x_feat)
    return K + jnp.eye(x.shape[0]) * jitter


def test_gram_matches_inline_construction(x):
    """`gram` reproduces the double-vmap construction it replaced, bit for bit."""
    kernel = RBFKernel(lengthscale=0.5)
    assert jnp.array_equal(gram(kernel, x, jitter=1e-10), _gram_reference(kernel, x, 1e-10))


def test_gram_default_jitter_is_zero(x):
    """The default returns the raw Gram matrix, with a unit diagonal for an RBF."""
    K = gram(RBFKernel(lengthscale=0.5), x)
    assert K.shape == (6, 6)
    assert jnp.allclose(jnp.diag(K), 1.0)
    assert jnp.allclose(K, K.T)


def test_gram_jitter_adds_to_diagonal(x):
    """Jitter is added to the diagonal only."""
    kernel = RBFKernel(lengthscale=0.5)
    K = gram(kernel, x)
    K_jittered = gram(kernel, x, jitter=1e-3)
    assert jnp.allclose(K_jittered - K, jnp.eye(6) * 1e-3)


def test_gram_method_matches_function(x):
    """`AbstractCovarianceKernel.gram` delegates to the module-level helper."""
    kernel = Matern32Kernel(lengthscale=0.75)
    assert jnp.array_equal(kernel.gram(x, jitter=1e-8), gram(kernel, x, jitter=1e-8))


def test_gram_preserves_batching(x):
    """A kernel with parameters of shape (D,) produces a batched (D, N, N) Gram."""
    periods = jnp.array([0.5, 1.0, 2.0])
    kernel = PeriodicKernel(period=periods, lengthscale=1.0)
    K = gram(kernel, x)
    assert K.shape == (3, 6, 6)

    # Each batch element equals the Gram of the corresponding scalar kernel.
    for i, period in enumerate(periods):
        assert jnp.allclose(K[i], gram(PeriodicKernel(period=period, lengthscale=1.0), x))


def test_gram_batched_jitter_hits_every_batch_diagonal(x):
    """Jitter broadcasts onto the diagonal of every batched Gram matrix."""
    kernel = PeriodicKernel(period=jnp.array([0.5, 2.0]), lengthscale=1.0)
    delta = gram(kernel, x, jitter=1e-3) - gram(kernel, x)
    assert delta.shape == (2, 6, 6)
    assert jnp.allclose(delta, jnp.broadcast_to(jnp.eye(6) * 1e-3, (2, 6, 6)))


def test_gram_nested_batching(x):
    """Shared axes are size-1 leading axes of the Gram, broadcasting to any size."""
    kernel = SharedIndependentKernel(
        base_kernel=RBFKernel(lengthscale=0.5),
        num_shared_axes=2,
    )
    K = gram(kernel, x)
    assert K.shape == (1, 1, 6, 6)
    assert jnp.array_equal(K[0, 0], gram(RBFKernel(lengthscale=0.5), x))


def test_gram_accepts_multidimensional_features():
    """An (N, d) input is used as N d-dimensional feature vectors."""
    x2d = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    K = gram(RBFKernel(lengthscale=1.0), x2d)
    assert K.shape == (3, 3)
    # Squared distance between rows 1 and 2 is 2, so K[1, 2] = exp(-1).
    assert jnp.allclose(K[1, 2], jnp.exp(-1.0))


def test_gram_accepts_plain_function(x):
    """`gram` works with a plain callable, not only an AbstractCovarianceKernel."""
    def kernel(x1, x2):
        return jnp.exp(-jnp.sum((x1 - x2) ** 2))
    K = gram(kernel, x)
    assert K.shape == (6, 6)
    assert jnp.allclose(jnp.diag(K), 1.0)


def test_gram_constant_kernel(x):
    """A constant kernel gives a rank-one Gram of that variance."""
    assert jnp.allclose(gram(ConstantKernel(variance=3.0), x), jnp.full((6, 6), 3.0))


def test_gaussian_process_uses_gram(x):
    """GaussianProcess builds its covariance from `gram` with its own jitter."""
    kernel = RBFKernel(lengthscale=0.5)
    gp = GaussianProcess(kernel=kernel, jitter=1e-8)
    y_event = jnp.zeros((6,))

    cov = gp(y_event, x).covariance()
    assert jnp.allclose(cov, gram(kernel, x, jitter=1e-8))


def test_gaussian_process_batched_kernel_covariance(x):
    """Batched kernels still give batched GP covariances via `gram`."""
    kernel = PeriodicKernel(period=jnp.array([0.5, 1.0, 2.0]), lengthscale=1.0)
    gp = GaussianProcess(kernel=kernel, jitter=1e-8)
    y_event = jnp.zeros((3, 6))

    cov = gp(y_event, x).covariance()
    assert cov.shape == (3, 6, 6)
    assert jnp.allclose(cov, gram(kernel, x, jitter=1e-8))


def test_cosine_kernel_values():
    """The cosine kernel is cos(2π Δx / period), reaching -1 at half a period."""
    kernel = CosineKernel(period=4.0)
    values = jnp.array([kernel(jnp.array([0.0]), jnp.array([x])) for x in [0.0, 1.0, 2.0, 3.0]])
    assert jnp.allclose(values, jnp.array([1.0, 0.0, -1.0, 0.0]), atol=1e-12)


def test_cosine_kernel_sums_multidimensional_differences():
    """For d > 1 the differences are summed: Δx = (1, 1) is half of a period of 4."""
    kernel = CosineKernel(period=4.0)
    # A Euclidean distance would give cos(2π √2 / 4) ≈ -0.61 instead.
    assert jnp.allclose(kernel(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])), -1.0)


@pytest.mark.parametrize("kernel", [
    CosineKernel(period=7.3),
    _damped_cosine(),
], ids=["cosine", "damped"])
def test_cosine_kernel_gram_is_positive_semidefinite(kernel, x_periods):
    """The Gram matrix has no negative eigenvalues beyond round-off, alone or damped."""
    eigvals = jnp.linalg.eigvalsh(kernel.gram(x_periods))
    assert eigvals.min() >= -1e-8 * eigvals.max()


def test_damped_cosine_kernel_gives_negative_covariance():
    """A damped cosine is negatively correlated at half a period."""
    kernel = _damped_cosine()
    assert kernel(jnp.array([0.0]), jnp.array([7.3 / 2])) < 0.0


def test_cosine_kernel_batched_period(x_periods):
    """A period of shape (D,) gives a (D, N, N) Gram of the scalar-period kernels."""
    periods = jnp.array([2.0, 5.0])
    K = CosineKernel(period=periods).gram(x_periods)
    assert K.shape == (2, 200, 200)
    for i, period in enumerate(periods):
        assert jnp.allclose(K[i], CosineKernel(period=period).gram(x_periods))


def test_cosine_kernel_gradients_are_finite(x_periods):
    """Gradients are finite in the period, and in x1 where x1 == x2."""
    grad_period = jax.grad(lambda p: CosineKernel(period=p).gram(x_periods).sum())(3.0)
    grad_x1 = jax.grad(lambda x1: CosineKernel(period=3.0)(x1, jnp.array([1.0])))(jnp.array([1.0]))
    assert jnp.isfinite(grad_period)
    assert jnp.all(jnp.isfinite(grad_x1))


def test_cosine_kernel_under_jit_and_vmap(x_periods):
    """jit and vmap over the period reproduce the eager Gram matrices."""
    periods = jnp.array([2.0, 5.0, 7.3])
    jitted = jax.jit(lambda p: CosineKernel(period=p).gram(x_periods))(3.0)
    vmapped = jax.vmap(lambda p: CosineKernel(period=p).gram(x_periods))(periods)
    assert jnp.allclose(jitted, CosineKernel(period=3.0).gram(x_periods))
    assert jnp.allclose(vmapped, CosineKernel(period=periods).gram(x_periods))


_AUTO = RBFKernel(lengthscale=0.3)
_CROSS = Matern32Kernel(lengthscale=2.0) * 0.5


def _shared_auto_cross():
    return SharedIndependentKernel(
        base_kernel=AutoCrossKernel(auto=_AUTO, cross=_CROSS, num_outputs=2),
        num_shared_axes=1,
    )


def _auto_where_ports_match(auto, cross):
    """Stack per-event (i, j, ReIm) matrices: `auto` exactly when i == j, else `cross`."""
    return jnp.stack([
        jnp.stack([jnp.stack([auto if i == j else cross] * 2) for j in range(2)])
        for i in range(2)
    ])


def test_shared_independent_appends_shared_axes_to_batched_base(x):
    """A batched base keeps its axes leading; the shared axis is a trailing size-1 axis."""
    K = gram(_shared_auto_cross(), x)
    assert K.shape == (2, 2, 1, 6, 6)


def test_shared_independent_routes_auto_cross_per_port(x):
    """Every (i, j, ReIm) entry gets the auto kernel exactly when i == j."""
    K = jnp.broadcast_to(gram(_shared_auto_cross(), x), (2, 2, 2, 6, 6))
    assert jnp.array_equal(K, _auto_where_ports_match(gram(_AUTO, x), gram(_CROSS, x)))


class _TwoPortModel(Model):
    def s(self, freq: Frequency) -> jnp.ndarray:
        return jnp.zeros((freq.npoints, 2, 2), dtype=complex)


def test_marginal_log_likelihood_routes_shared_auto_cross():
    """The predictive covariance of each (i, j, ReIm) event uses auto exactly when i == j."""
    frequency = Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')
    noise = 1e-2
    gp = GaussianProcess(kernel=_shared_auto_cross(), jitter=1e-8)
    mll = MarginalLogLikelihood(
        predictor=Feature('s'),
        observed=jnp.zeros((5, 2, 2), dtype=complex),
        likelihood=GaussianLikelihood(noise=noise),
        discrepancy=gp,
    )

    cov = mll.predictive_distribution(_TwoPortModel(), frequency).covariance()
    f = frequency.f_scaled
    expected = _auto_where_ports_match(
        gram(_AUTO, f, jitter=1e-8) + noise * jnp.eye(5),
        gram(_CROSS, f, jitter=1e-8) + noise * jnp.eye(5),
    )
    assert cov.shape == (2, 2, 2, 5, 5)
    assert jnp.allclose(cov, expected)


def _hyperparameter_kernel(random: bool):
    """A product-and-sum kernel whose hyperparameters are floats or `prf.Random`."""
    h = lambda v: prf.Random(RelativeTruncatedNormal(v, 0.1)) if random else v
    return (
        PeriodicKernel(h(1.2), h(1.5)) * Matern52Kernel(h(3.0)) * h(2e-1)
        + Matern52Kernel(h(0.4)) * h(3e-1)
    )


def test_gram_random_hyperparameters_match_floats(x):
    """`Random` hyperparameters give the same Gram as floats with the same values."""
    K_float = jax.jit(gram)(_hyperparameter_kernel(random=False), x)
    K_random = eqx.filter_jit(gram)(_hyperparameter_kernel(random=True), x)
    assert jnp.array_equal(K_random, K_float)


def test_gram_gradient_wrt_random_hyperparameters_is_unchanged(x):
    """Under jit and grad, `gram` differentiates exactly as the plain construction."""
    kernel = _hyperparameter_kernel(random=True)
    grad = eqx.filter_jit(eqx.filter_grad(lambda k: gram(k, x).sum()))(kernel)
    expected = eqx.filter_jit(eqx.filter_grad(
        lambda k: _gram_reference(unwrap(k), x, 0.0).sum()
    ))(kernel)
    grad_leaves, expected_leaves = jax.tree.leaves(grad), jax.tree.leaves(expected)
    assert len(grad_leaves) == len(expected_leaves)
    for actual, reference in zip(grad_leaves, expected_leaves):
        assert jnp.allclose(actual, reference, rtol=1e-12, atol=0.0)
    # Each of the six hyperparameters' raw values receives a gradient.
    raw_grads = [p.raw_value for p in prf.params(grad).values()]
    assert len(raw_grads) == 6
    assert all(g != 0.0 for g in raw_grads)


def test_marginal_log_likelihood_random_hyperparameters_match_floats():
    """`Random` and float hyperparameters with the same values give the same log-likelihood."""
    frequency = Frequency(start=1.0, stop=10.0, npoints=20, unit='GHz')
    observed = jnp.full((20, 2, 2), 0.01 + 0.02j)

    def mll(random):
        kernel = SharedIndependentKernel(AutoCrossKernel(
            _hyperparameter_kernel(random), _hyperparameter_kernel(random) * 0.5, num_outputs=2,
        ))
        return MarginalLogLikelihood(
            predictor=Feature('s'),
            observed=observed,
            likelihood=GaussianLikelihood(noise=1e-4),
            discrepancy=GaussianProcess(kernel=kernel, jitter=1e-8),
        )

    value = eqx.filter_jit(lambda e: e(_TwoPortModel(), frequency))
    assert value(mll(random=True)) == value(mll(random=False))


def test_cross_gram_of_inputs_with_themselves_is_the_gram(x):
    """The cross-Gram of x with itself equals the square Gram without jitter, bit for bit."""
    for kernel in [RBFKernel(lengthscale=0.5), _shared_auto_cross(), _hyperparameter_kernel(random=False)]:
        assert jnp.array_equal(cross_gram(kernel, x, x), gram(kernel, x))


def test_cross_gram_keeps_block_layout(x):
    """A cross-Gram has the (*batch, N1, N2) layout, routing blocks as the square Gram does."""
    x_new = jnp.linspace(-1.0, 3.0, 4)
    K = jnp.broadcast_to(cross_gram(_shared_auto_cross(), x_new, x), (2, 2, 2, 4, 6))
    expected = _auto_where_ports_match(cross_gram(_AUTO, x_new, x), cross_gram(_CROSS, x_new, x))
    assert jnp.array_equal(K, expected)
    # RBF entries are exp(-0.5 (Δx / l)^2).
    assert jnp.allclose(
        cross_gram(_AUTO, x_new, x), jnp.exp(-0.5 * ((x_new[:, None] - x[None, :]) / 0.3) ** 2),
        rtol=1e-14, atol=0.0,
    )
