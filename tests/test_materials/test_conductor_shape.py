import jax.numpy as jnp
import pytest
from scipy.constants import mu_0

from pmrf.frequency import Frequency
from pmrf.materials.conductor_shape import (
    HalfSpaceShape,
    TescheRodShape,
    TescheTubeShape,
)


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=20.0, npoints=20, unit='GHz')


def _zeta_c(freq, sigma, mu_r=1.0):
    return jnp.sqrt(1j * freq.w * mu_0 * mu_r / sigma)


def test_half_space_is_the_trivial_shape_factor(freq):
    """Leontovich's half-space carries no shape factor: Z_s == zeta_c exactly."""
    sigma, mu_r = 5.8e7, 1.0
    zeta_c = _zeta_c(freq, sigma, mu_r)

    zs = HalfSpaceShape().impedance(freq.w, zeta_c, sigma, mu_r)

    assert jnp.allclose(zs, zeta_c)


def test_tesche_rod_matches_its_own_strong_skin_asymptote():
    """Tesche's rod circuit is not exact, so it does not tend to zeta_c.

    Its documented defect is that the strong-skin limit of the shape factor
    is $1 + R_{dc,sq}/\\zeta_c$, not $1$: this is an algebraic identity of
    the circuit (the high-frequency term of the blend saturates at
    $\\zeta_c$), which holds once $\\zeta_c$ is large enough compared to
    $R_{dc,sq}$ that the blend's own high-frequency term has converged --
    a synthetic 100 EHz makes that comparison unambiguous, well past the
    frequencies :class:`TescheCoaxialFormulation` is ever used at.
    """
    sigma, mu_r, radius = 5.8e7, 1.0, 0.455e-3
    freq = Frequency.from_f(jnp.array([1e20]))
    zeta_c = _zeta_c(freq, sigma, mu_r)

    zs = TescheRodShape().impedance(freq.w, zeta_c, sigma, mu_r, radius=radius)

    r_dc_sq = 2 / (radius * sigma)
    expected_factor = 1 + r_dc_sq / zeta_c
    assert jnp.allclose(zs / zeta_c, expected_factor, rtol=1e-6)


def test_tesche_tube_matches_its_own_strong_skin_asymptote():
    """Same defect as the rod, with the tube's own dc sheet resistance."""
    sigma, mu_r, radius, thickness = 5.8e7, 1.0, 0.455e-3, 0.2e-3
    freq = Frequency.from_f(jnp.array([1e20]))
    zeta_c = _zeta_c(freq, sigma, mu_r)

    zs = TescheTubeShape().impedance(
        freq.w, zeta_c, sigma, mu_r, radius=radius, thickness=thickness
    )

    r_dc_sq = 1 / (thickness * sigma)
    expected_factor = 1 + r_dc_sq / zeta_c
    assert jnp.allclose(zs / zeta_c, expected_factor, rtol=1e-6)


def test_tesche_shape_factors_tend_to_one_as_skin_effect_strengthens():
    """Tesche's known, documented departure from the exact 1 shrinks with frequency.

    At 20 GHz for a 0.455 mm inner radius the departure is on the order of
    0.4%, matching the size quoted for this circuit approximation; it keeps
    shrinking (roughly as $1/\\sqrt{f}$, since $\\zeta_c\\propto\\sqrt{f}$)
    rather than sitting at a fixed offset, which is what distinguishes a
    genuine asymptote from a bug.
    """
    sigma, mu_r, radius = 5.8e7, 1.0, 0.455e-3
    low = Frequency.from_f(jnp.array([1e9]))
    high = Frequency.from_f(jnp.array([20e9]))

    def departure(freq):
        zeta_c = _zeta_c(freq, sigma, mu_r)
        zs = TescheRodShape().impedance(freq.w, zeta_c, sigma, mu_r, radius=radius)
        return jnp.abs(zs / zeta_c - 1)

    departure_low, departure_high = departure(low)[0], departure(high)[0]

    assert departure_high < 0.01
    assert departure_high < departure_low


def test_tesche_tube_approaches_half_space_as_the_wall_thickens():
    """A very thick tube is a half-space: TescheTubeShape's own limit.

    $R_{dc,sq}=1/(t\\sigma)$ vanishes as the wall thickens, and $L_{int,sq}$
    diverges (only logarithmically in $t$, so the wall must be made very
    thick relative to the radius for the comparison to be tight).
    """
    sigma, mu_r, radius = 5.8e7, 1.0, 1e-3
    freq = Frequency.from_f(jnp.array([1e9]))
    zeta_c = _zeta_c(freq, sigma, mu_r)

    thick_tube = TescheTubeShape().impedance(
        freq.w, zeta_c, sigma, mu_r, radius=radius, thickness=1e6
    )
    half_space = HalfSpaceShape().impedance(freq.w, zeta_c, sigma, mu_r)

    assert jnp.allclose(thick_tube, half_space, rtol=1e-3)
