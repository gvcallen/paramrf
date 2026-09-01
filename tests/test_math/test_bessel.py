"""The complex Bessel ratio that the exact cylindrical conductor rests on.

scipy is the reference here rather than a recorded ParamRF output: the point
of these tests is that the hand-rolled JAX evaluator reproduces a known
special function, not that it keeps doing whatever it does now.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ive

from pmrf.math.bessel import I0_OVER_I1_SERIES_CUTOFF, i0_over_i1


def _reference(x):
    """$I_0/I_1$ from scipy, via the exponentially scaled forms whose scalings cancel."""
    return ive(0, x) / ive(1, x)


@pytest.mark.parametrize(
    # Both branches, and the seam between them, are separately in range.
    "lo, hi, rtol, regime",
    [
        (1e-6, 1e-2, 5e-15, "small argument, where the ratio is essentially 2/x"),
        (1e-2, 19.0, 1e-12, "power series"),
        (19.0, 21.0, 5e-8, "the switch between the two branches"),
        (21.0, 1e2, 5e-8, "asymptotic expansion, just past the switch"),
        (1e2, 1e4, 5e-12, "asymptotic expansion, deep in its own regime"),
    ],
)
@pytest.mark.parametrize("arg_deg", [0.0, 45.0, 60.0])
def test_matches_scipy_over_the_argument_range(lo, hi, rtol, regime, arg_deg):
    """45 degrees is the ray gamma = sqrt(j*w*mu*sigma) sits on; 0 and 60 bracket it."""
    magnitude = np.logspace(np.log10(lo), np.log10(hi), 400)
    x = magnitude * np.exp(1j * np.deg2rad(arg_deg))

    got = np.asarray(i0_over_i1(jnp.asarray(x)))

    assert np.abs(got / _reference(x) - 1).max() < rtol, regime


def test_small_argument_tends_to_two_over_x():
    """The dc limit of a cylindrical conductor is this limit, with no special case."""
    x = jnp.asarray([1e-8, 1e-6, 1e-4]) * jnp.exp(1j * jnp.pi / 4)

    assert jnp.allclose(i0_over_i1(x), 2 / x, rtol=1e-8)


def test_large_argument_tends_to_one_plus_half_inverse():
    """The strong-skin limit: a half-space plus the leading curvature term."""
    x = jnp.asarray([1e3, 1e4]) * jnp.exp(1j * jnp.pi / 4)

    assert jnp.allclose(i0_over_i1(x), 1 + 1 / (2 * x), rtol=1e-6)


def test_gradient_is_finite_across_the_branch_switch():
    """Both branches are evaluated on safe arguments, so neither poisons the other."""
    def real_part(m):
        return jnp.real(i0_over_i1(m * jnp.exp(1j * jnp.pi / 4)))

    grads = jax.vmap(jax.grad(real_part))(
        jnp.asarray([1e-6, 1.0, I0_OVER_I1_SERIES_CUTOFF, 1e3, 1e4])
    )

    assert jnp.all(jnp.isfinite(grads))


def test_the_derivative_jump_at_the_seam_is_small():
    """The seam is a switch, not a blend, so record how big its kink actually is."""
    def real_part(m):
        return jnp.real(i0_over_i1(m * jnp.exp(1j * jnp.pi / 4)))

    eps = 1e-6
    below = jax.grad(real_part)(I0_OVER_I1_SERIES_CUTOFF - eps)
    above = jax.grad(real_part)(I0_OVER_I1_SERIES_CUTOFF + eps)

    assert abs(above / below - 1) < 2e-4
