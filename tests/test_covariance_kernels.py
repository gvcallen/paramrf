# tests/test_covariance_kernels.py
import pytest
import jax
import jax.numpy as jnp

from pmrf.covariance_kernels import (
    gram,
    RBFKernel,
    PeriodicKernel,
    Matern32Kernel,
    ConstantKernel,
    SharedIndependentKernel,
)
from pmrf.discrepancy_models import GaussianProcess


@pytest.fixture
def x():
    return jnp.linspace(0.0, 2.0, 6)


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
    """Multi-axis kernel batching is preserved as leading axes of the Gram."""
    kernel = SharedIndependentKernel(
        base_kernel=RBFKernel(lengthscale=0.5),
        output_shape=(2, 4),
    )
    K = gram(kernel, x)
    assert K.shape == (2, 4, 6, 6)
    # Every batch element is the same shared kernel.
    assert jnp.allclose(K, jnp.broadcast_to(gram(RBFKernel(lengthscale=0.5), x), (2, 4, 6, 6)))


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
