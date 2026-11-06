"""
Scattering cascade reduction of 2-ports.

A 2-port junction is inverted in closed form. Its reference is the N >= 2 SVD
path, reached by cascading the block-diagonal 4-port of two independent 2-port
chains: the junction blocks are then diagonal, and the stacked SVD pins a
2-port's direction exactly when the closed form does.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np

import pmrf  # noqa: F401  (enables x64)
from pmrf.rf.cascade import cascade_scattering, cascade_two_s


def random_sections(key, shape):
    k1, k2 = jax.random.split(key)
    return 0.3 * (jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape))


def block_diagonal(s_a, s_b):
    """The 4-port [a_in, b_in, a_out, b_out] of 2-ports `s_a` and `s_b` side by side."""
    zero = jnp.zeros((), s_a.dtype)
    return jnp.array([
        [s_a[0, 0], zero, s_a[0, 1], zero],
        [zero, s_b[0, 0], zero, s_b[0, 1]],
        [s_a[1, 0], zero, s_a[1, 1], zero],
        [zero, s_b[1, 0], zero, s_b[1, 1]],
    ])


def svd_path(s_stacked, partner):
    """2-port cascade of `s_stacked` through the 4-port SVD path, beside `partner`."""
    s4 = jax.vmap(block_diagonal)(s_stacked, partner)
    z0 = jnp.full(s4.shape[:-1], 50.0 + 0j)
    s_cas, _ = cascade_scattering(s4, z0)
    return s_cas[np.ix_((0, 2), (0, 2))]


def closed_form(s_stacked):
    z0 = jnp.full(s_stacked.shape[:-1], 50.0 + 0j)
    return cascade_scattering(s_stacked, z0)[0]


def star(s_a, s_b):
    """Plain scalar star product of two 2-ports, with no pinning."""
    x = 1 / (1 - s_b[0, 0] * s_a[1, 1])
    return jnp.array([
        [s_a[0, 0] + s_a[0, 1] * x * s_b[0, 0] * s_a[1, 0], s_a[0, 1] * x * s_b[0, 1]],
        [s_b[1, 0] * x * s_a[1, 0], s_b[1, 1] + s_b[1, 0] * x * s_a[1, 1] * s_b[0, 1]],
    ])


def loss(s):
    weights = jnp.arange(1.0, 5.0).reshape(2, 2)
    return jnp.sum(weights * jnp.real(s)) + jnp.sum(weights.T * jnp.imag(s))


def pinned_sections(t):
    """
    Two reflect-only 2-ports meeting at a lossless, unobservable cavity: the
    junction has ``m = 1 - b11 a22 = 0`` and ``a12 = b21 = 0``. `t` moves both
    outer reflections, so the gradient is non-trivial.
    """
    return jnp.array([
        [[0.2 + 0.1j + t, 0.0], [0.4, 1.0]],
        [[1.0, 0.3], [0.0, -0.5j + 2 * t]],
    ], dtype=complex)


def test_two_port_cascade_has_no_lapack_call():
    s = random_sections(jax.random.PRNGKey(0), (151, 11, 2, 2))
    z0 = jnp.full((151, 11, 2), 50.0 + 0j)
    hlo = jax.jit(jax.vmap(cascade_scattering)).lower(s, z0).compile().as_text()
    targets = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    assert not [t for t in targets if "lapack" in t], targets


def test_random_two_port_cascade_matches_svd_path():
    s = random_sections(jax.random.PRNGKey(0), (151, 11, 2, 2))
    partner = random_sections(jax.random.PRNGKey(1), (151, 11, 2, 2))
    got = jax.vmap(closed_form)(s)
    want = jax.vmap(svd_path)(s, partner)
    # Only rounding separates the scalar division from the 4-port LU solve:
    # 1.3e-15 relative at worst over these 11-section chains.
    np.testing.assert_allclose(got, want, rtol=2e-15, atol=0)


def test_pinned_junction_is_finite_and_matches_svd_path():
    s = pinned_sections(0.0)
    partner = random_sections(jax.random.PRNGKey(2), (2, 2, 2))

    got = closed_form(s)
    np.testing.assert_allclose(got, jnp.diag(jnp.array([s[0, 0, 0], s[1, 1, 1]])), atol=0)
    np.testing.assert_allclose(got, svd_path(s, partner), rtol=1e-15, atol=1e-16)

    grad = jax.grad(lambda t: loss(closed_form(pinned_sections(t))))(0.0)
    grad_svd = jax.grad(lambda t: loss(svd_path(pinned_sections(t), partner)))(0.0)
    assert np.isfinite(grad)
    np.testing.assert_allclose(grad, grad_svd, rtol=1e-15)

    # The gradient with respect to every entry, including those through the
    # pinned inverse, is finite and matches too.
    s_grad = jax.grad(lambda s: loss(closed_form(s)))(s)
    s_grad_svd = jax.grad(lambda s: loss(svd_path(s, partner)))(s)
    assert np.all(np.isfinite(s_grad))
    np.testing.assert_allclose(s_grad, s_grad_svd, rtol=1e-15, atol=1e-16)


def test_near_zero_observable_junction_is_not_pinned():
    # m = 1 - b11 a22 = 1e-12, observable through a small a12: inverted exactly.
    s_a = jnp.array([[0.1, 1e-6], [0.5, 1.0]], dtype=complex)
    s_b = jnp.array([[1.0 - 1e-12, 0.2], [0.3, 0.4j]], dtype=complex)
    z0 = jnp.full(2, 50.0 + 0j)
    got, _ = cascade_two_s(s_a, z0, s_b, z0)
    np.testing.assert_allclose(got, star(s_a, s_b), rtol=1e-15)
    assert abs(got[0, 0] - s_a[0, 0]) > 1e3
