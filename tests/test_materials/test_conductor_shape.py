import jax.numpy as jnp
import pytest
from scipy.constants import mu_0

from pmrf.frequency import Frequency
from pmrf.materials import ConductorProperties
from pmrf.materials.conductor_shape import (
    HalfSpaceShape,
    TescheRodShape,
    TescheTubeShape,
)


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=20.0, npoints=20, unit='GHz')


def _conductor(freq, sigma, mu_r=1.0):
    zeta_c = jnp.sqrt(1j * freq.w * mu_0 * mu_r / sigma)
    ones = jnp.ones(freq.npoints)
    return ConductorProperties(zeta_c, sigma * ones, mu_r * ones)


def test_half_space_is_the_trivial_shape_factor(freq):
    """Leontovich's half-space carries no shape factor: Z_s == zeta_c exactly."""
    conductor = _conductor(freq, sigma=5.8e7)

    zs = HalfSpaceShape().impedance(freq.w, conductor)

    assert jnp.allclose(zs, conductor.zs)


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
    sigma, a = 5.8e7, 0.455e-3
    freq = Frequency.from_f(jnp.array([1e20]))
    conductor = _conductor(freq, sigma)

    zs = TescheRodShape().impedance(freq.w, conductor, a=a)

    r_dc_sq = 2 / (a * sigma)
    expected_factor = 1 + r_dc_sq / conductor.zs
    assert jnp.allclose(zs / conductor.zs, expected_factor, rtol=1e-6)


def test_tesche_tube_matches_its_own_strong_skin_asymptote():
    """Same defect as the rod, with the tube's own dc sheet resistance."""
    sigma, a, t = 5.8e7, 0.455e-3, 0.2e-3
    freq = Frequency.from_f(jnp.array([1e20]))
    conductor = _conductor(freq, sigma)

    zs = TescheTubeShape().impedance(freq.w, conductor, a=a, t=t)

    r_dc_sq = 1 / (t * sigma)
    expected_factor = 1 + r_dc_sq / conductor.zs
    assert jnp.allclose(zs / conductor.zs, expected_factor, rtol=1e-6)


def test_tesche_shape_factors_tend_to_one_as_skin_effect_strengthens():
    """Tesche's known, documented departure from the exact 1 shrinks with frequency.

    At 20 GHz for a 0.455 mm inner radius the departure is on the order of
    0.4%, matching the size quoted for this circuit approximation; it keeps
    shrinking (roughly as $1/\\sqrt{f}$, since $\\zeta_c\\propto\\sqrt{f}$)
    rather than sitting at a fixed offset, which is what distinguishes a
    genuine asymptote from a bug.
    """
    sigma, a = 5.8e7, 0.455e-3
    low = Frequency.from_f(jnp.array([1e9]))
    high = Frequency.from_f(jnp.array([20e9]))

    def departure(freq):
        conductor = _conductor(freq, sigma)
        zs = TescheRodShape().impedance(freq.w, conductor, a=a)
        return jnp.abs(zs / conductor.zs - 1)

    departure_low, departure_high = departure(low)[0], departure(high)[0]

    assert departure_high < 0.01
    assert departure_high < departure_low


def test_tesche_tube_approaches_half_space_as_the_wall_thickens():
    """A very thick tube is a half-space: TescheTubeShape's own limit.

    $R_{dc,sq}=1/(t\\sigma)$ vanishes as the wall thickens, and $L_{int,sq}$
    diverges (only logarithmically in $t$, so the wall must be made very
    thick relative to the radius for the comparison to be tight).
    """
    sigma, a = 5.8e7, 1e-3
    freq = Frequency.from_f(jnp.array([1e9]))
    conductor = _conductor(freq, sigma)

    thick_tube = TescheTubeShape().impedance(freq.w, conductor, a=a, t=1e6)
    half_space = HalfSpaceShape().impedance(freq.w, conductor)

    assert jnp.allclose(thick_tube, half_space, rtol=1e-3)
