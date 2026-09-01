import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.special import ive

from pmrf.frequency import Frequency
from pmrf.materials import BulkConductor, ConductorProperties, RoughConductor
from pmrf.materials.conductor_shape import (
    HalfSpaceShape,
    HollowayKuesterSlabShape,
    RootSumSquareSlabShape,
    SchelkunoffRodShape,
    SchelkunoffTubeShape,
    SchelkunoffCothTubeShape,
    TescheRodShape,
    TescheTubeShape,
)
from pmrf.models.components.lines.current_distribution import WheelerCurrentDistribution


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


def test_holloway_kuester_slab_matches_half_thickness_coth():
    """The total strip current sees Holloway--Kuester eq. (100)."""
    sigma, t = 5.8e7, 18e-6
    freq = Frequency.from_f(jnp.array([10e6, 30e6, 100e6, 500e6]))
    conductor = _conductor(freq, sigma)

    actual = HollowayKuesterSlabShape(t=t).impedance(freq.w, conductor)

    gamma = np.sqrt(1j * np.asarray(freq.w) * mu_0 * sigma)
    expected = np.asarray(conductor.zs) / np.tanh(gamma * t / 2)
    assert np.allclose(actual, expected, rtol=2e-7)


def test_holloway_kuester_slab_has_dc_and_strong_skin_limits():
    """The slab tends to 2/(sigma*t) at dc and zeta_c in strong skin effect."""
    sigma, t = 5.8e7, 35e-6
    freq = Frequency.from_f(jnp.array([0.0, 1e12]))
    conductor = _conductor(freq, sigma)

    zs = HollowayKuesterSlabShape(t=t).impedance(freq.w, conductor)

    assert jnp.allclose(zs[0], 2 / (sigma * t))
    assert jnp.allclose(zs[1], conductor.zs[1], rtol=1e-6)


def test_root_sum_square_slab_preserves_the_paramrf_convention():
    """The named compatibility entry blends resistance and retains half-space X."""
    sigma, t = 5.8e7, 35e-6
    freq = Frequency.from_f(jnp.array([0.0, 10e6, 500e6]))
    conductor = _conductor(freq, sigma)

    r_dc_sq = 1 / (sigma * t)
    zs = RootSumSquareSlabShape(dc_shape_factor=1 / t).impedance(freq.w, conductor)

    expected_r = jnp.sqrt(r_dc_sq**2 + jnp.real(conductor.zs) ** 2)
    expected = expected_r + 1j * jnp.imag(conductor.zs)
    assert jnp.allclose(zs, expected)


@pytest.mark.parametrize(
    "w,h,zc,expected,rtol",
    [
        (
            1.55e-3, 0.8e-3, 50.0,
            [[0.9668851882, 0.7848033691, 0.8436921430],
             [1.6462990845, 1.7548736823, 1.7819881438]],
            3e-7,
        ),
        (
            0.45e-3, 0.25e-3, 50.0,
            [[3.3303823150, 2.7032116046, 2.9060507149],
             [5.6705857357, 6.0445649056, 6.1379591621]],
            3e-7,
        ),
        (
            0.35e-3, 0.8e-3, 96.0,
            [[3.6163499495, 2.9353264057, 3.2399008371],
             [6.1574979986, 6.5635893792, 6.7053353283]],
            4e-7,
        ),
    ],
    ids=["w1p55_h0p8", "w0p45_h0p25", "w0p35_h0p8_96ohm"],
)
def test_tabulated_planar_entries_record_transition_error(w, h, zc, expected, rtol):
    r"""Slab, half-space, and compatibility blend at issue #100's geometries.

    Each row is `[exact slab, semi-infinite, root-sum-square blend]` in
    ohm/m at 10 and 50 MHz.  The literals were independently evaluated from
    Holloway--Kuester eq. (100) and Wheeler's published current weight using
    35 um copper with rho=1.68e-8 ohm m.  `h` records the complete tabulated
    cross-section even though these planar entries depend on it only through
    the supplied characteristic impedance.
    """
    del h
    t, sigma = 35e-6, 1 / 1.68e-8
    freq = Frequency.from_f(jnp.array([10e6, 50e6]))
    conductor = _conductor(freq, sigma)
    blend, weight = WheelerCurrentDistribution().distribute(
        freq, zc=jnp.full(freq.f.shape, zc), w=w, t=t,
    )[0]

    actual = jnp.stack(
        [
            jnp.real(HollowayKuesterSlabShape(t=t).impedance(freq.w, conductor) * weight),
            jnp.real(HalfSpaceShape().impedance(freq.w, conductor) * weight),
            jnp.real(blend.impedance(freq.w, conductor) * weight),
        ],
        axis=1,
    )

    assert jnp.allclose(actual, jnp.asarray(expected), rtol=rtol)


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


def test_schelkunoff_rod_matches_scipy_bessel_ratio():
    """The shape factor is Schelkunoff eq. (65), checked against scipy directly."""
    sigma, a = 5.8e7, 0.455e-3
    freq = Frequency(start=1.0, stop=20.0, npoints=20, unit='GHz')
    conductor = _conductor(freq, sigma)

    zs = SchelkunoffRodShape().impedance(freq.w, conductor, a=a)

    gamma_a = np.sqrt(1j * np.asarray(freq.w) * mu_0 * sigma) * a
    expected = np.asarray(conductor.zs) * ive(0, gamma_a) / ive(1, gamma_a)
    assert np.allclose(np.asarray(zs), expected, rtol=1e-10)


def test_schelkunoff_rod_reproduces_the_dc_resistance_without_a_special_case():
    """At dc the Bessel ratio's pole cancels zeta_c's zero, giving 1/(pi a^2 sigma).

    The dc point itself is the where-branch, but the limit is approached
    continuously from above: 1 Hz already lands on the same number, which is
    what shows the branch is a limit rather than a patch.
    """
    sigma, a = 5.8e7, 0.455e-3
    freq = Frequency.from_f(jnp.array([0.0, 1.0]))
    conductor = _conductor(freq, sigma)

    z = SchelkunoffRodShape().impedance(freq.w, conductor, a=a) / (2 * jnp.pi * a)

    r_dc = 1 / (jnp.pi * a**2 * sigma)
    assert jnp.allclose(jnp.real(z), r_dc, rtol=1e-9)


def test_schelkunoff_rod_tends_to_the_half_space_plus_curvature():
    """The strong-skin limit is $\\zeta_c(1 + 1/2\\gamma a)$, not $\\zeta_c$.

    This curvature term is exactly what :class:`TescheRodShape` cannot
    produce, and it is a real effect: at 20 GHz it is still 0.05% of the
    shape factor for a 0.455 mm radius.
    """
    sigma, a = 5.8e7, 0.455e-3
    freq = Frequency.from_f(jnp.array([20e9]))
    conductor = _conductor(freq, sigma)

    zs = SchelkunoffRodShape().impedance(freq.w, conductor, a=a)

    gamma_a = conductor.sigma * conductor.zs * a
    assert jnp.allclose(zs / conductor.zs, 1 + 1 / (2 * gamma_a), rtol=1e-4)
    assert jnp.abs(1 / (2 * gamma_a))[0] > 1e-4


def test_tesche_rod_departs_from_schelkunoff_by_a_shape_error():
    """The Tesche error is frequency-shaped, so refitting sigma cannot absorb it.

    Removing the best-fit global scale from the ratio of the two shape
    factors over 10-500 MHz leaves a residual of several percent for a thin
    inner conductor -- the number that motivates making Schelkunoff the
    default.
    """
    sigma, a = 5.8e7, 0.255e-3  # RG405-like inner conductor
    freq = Frequency(start=10.0, stop=500.0, npoints=51, unit='MHz')
    conductor = _conductor(freq, sigma)

    tesche = TescheRodShape().impedance(freq.w, conductor, a=a)
    exact = SchelkunoffRodShape().impedance(freq.w, conductor, a=a)

    ratio = jnp.real(tesche) / jnp.real(exact)
    residual = jnp.max(jnp.abs(ratio / jnp.mean(ratio) - 1))
    assert 0.03 < residual < 0.12


def test_schelkunoff_rod_gradients_are_finite_including_at_dc():
    """Fitting needs finite d/dsigma and d/df everywhere, dc included.

    The gradient is taken through :class:`BulkConductor` rather than through
    a hand-built ``zeta_c``: $\\zeta_c=\\sqrt{j\\omega\\mu/\\sigma}$ has an
    infinite $\\omega$-derivative at dc of its own, which every shape in
    this module would inherit, and the material is where that is handled.
    """
    a = 0.455e-3

    def resistance(sigma, f):
        freq = Frequency.from_f(jnp.atleast_1d(f))
        conductor = BulkConductor(sigma=sigma).properties(freq)
        return jnp.real(SchelkunoffRodShape().impedance(freq.w, conductor, a=a))[0]

    for f in [0.0, 1e3, 1e9]:
        d_sigma, d_f = jax.grad(resistance, argnums=(0, 1))(5.8e7, f)
        assert jnp.isfinite(d_sigma) and jnp.isfinite(d_f)


def test_schelkunoff_rod_is_lossless_for_a_perfect_conductor():
    """An infinite sigma sends gamma to infinity and zeta_c to zero: Z_s is 0.

    The Bessel argument is unevaluable there, so the shape has to recognise
    the limit rather than propagate a nan through it.
    """
    freq = Frequency.from_f(jnp.array([0.0, 1e9]))
    conductor = _conductor(freq, sigma=jnp.inf)

    zs = SchelkunoffRodShape().impedance(freq.w, conductor, a=0.455e-3)

    assert jnp.all(zs == 0)


def test_schelkunoff_tube_and_coth_approximation_agree_in_coax_band():
    """The cheap shield entry tracks the cylindrical tube in the RF band."""
    freq = Frequency(start=10.0, stop=500.0, npoints=20, unit='MHz')
    conductor = _conductor(freq, 5.8e7)
    exact = SchelkunoffTubeShape().impedance(
        freq.w, conductor, a=1.475e-3, t=25e-6
    )
    cheap = SchelkunoffCothTubeShape().impedance(
        freq.w, conductor, a=1.475e-3, t=25e-6
    )
    assert jnp.allclose(cheap, exact, rtol=2e-3)


@pytest.mark.parametrize(
    'shape, geometry',
    [
        (HollowayKuesterSlabShape(t=18e-6), {}),
        (SchelkunoffRodShape(), dict(a=20e-6)),
        (SchelkunoffTubeShape(), dict(a=1.475e-3, t=25e-6)),
        (SchelkunoffCothTubeShape(), dict(a=1.475e-3, t=25e-6)),
    ],
)
def test_roughness_scales_the_prefactor_and_not_the_shape_factor(shape, geometry):
    """Surface texture roughens Z_s; it does not change bulk field diffusion.

    A rough and a smooth conductor of the same bulk material must therefore
    produce the same dimensionless shape factor Z_s/zeta_c, and differ only
    by the roughness factor K on the prefactor. Deriving gamma as
    ``sigma * zs`` inflates the shape's argument by K and breaks this.
    """
    freq = Frequency(start=1.0, stop=20.0, npoints=20, unit='GHz')
    sigma = 5.8e7
    smooth = BulkConductor(sigma=sigma).properties(freq)
    rough = RoughConductor(sigma=sigma, roughness=2e-6).properties(freq)

    k = rough.zs / smooth.zs
    assert jnp.max(jnp.real(k)) > 1.5  # roughness is significant in this band
    assert jnp.allclose(jnp.imag(k), 0.0, atol=1e-12)

    smooth_factor = shape.impedance(freq.w, smooth, **geometry) / smooth.zs
    rough_factor = shape.impedance(freq.w, rough, **geometry) / rough.zs

    assert jnp.allclose(rough_factor, smooth_factor, rtol=1e-12)


def test_metal_propagation_constant_is_the_inverse_complex_skin_depth():
    """gamma = sqrt(j w mu sigma) = (1+j)/delta, with finite dc gradient."""
    freq = Frequency.from_f(jnp.array([0.0, 1e6, 1e9]))
    conductor = BulkConductor(sigma=5.8e7, mu_r=2.0).properties(freq)

    gamma = conductor.gamma(freq.w)

    w = np.asarray(freq.w)
    expected = np.sqrt(1j * w * mu_0 * 2.0 * 5.8e7)
    assert np.allclose(gamma, expected, rtol=1e-12)

    grad = jax.grad(
        lambda f: jnp.real(
            BulkConductor(sigma=5.8e7)
            .properties(Frequency.from_f(jnp.atleast_1d(f)))
            .gamma(jnp.atleast_1d(2 * jnp.pi * f))
        )[0]
    )(0.0)
    assert jnp.isfinite(grad)
