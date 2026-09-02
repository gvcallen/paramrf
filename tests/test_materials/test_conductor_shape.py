import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.special import ive, kve

from pmrf.frequency import Frequency
from pmrf.materials import BulkConductor, ConductorProperties, RoughConductor
from pmrf.materials.conductor_shape import (
    HalfSpaceShape,
    HollowayKuesterSlabShape,
    RootSumSquareSlabShape,
    SchelkunoffRodShape,
    SchelkunoffTubeShape,
    TescheRodShape,
    TescheTubeShape,
)
from pmrf.models.components.lines.microstrip import (
    MicrostripCrossSection,
    WheelerCurrentDistribution,
)
from pmrf.models.components.lines.planar import PlanarQuasiStaticResult


def _solved(zc):
    """A minimal solved state carrying only the impedance a strategy reads."""
    ones = jnp.ones_like(zc)
    return PlanarQuasiStaticResult(
        ep_eff=ones, zc=zc, w_eff=ones, shunt_conductance_factor=ones
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


def test_holloway_kuester_slab_matches_half_thickness_coth():
    """The total strip current sees Holloway--Kuester eq. (100)."""
    sigma, t = 5.8e7, 18e-6
    freq = Frequency.from_f(jnp.array([10e6, 30e6, 100e6, 500e6]))
    conductor = _conductor(freq, sigma)

    actual = HollowayKuesterSlabShape().impedance(freq.w, conductor, t=t)

    gamma = np.sqrt(1j * np.asarray(freq.w) * mu_0 * sigma)
    expected = np.asarray(conductor.zs) / np.tanh(gamma * t / 2)
    assert np.allclose(actual, expected, rtol=2e-7)


def test_holloway_kuester_slab_has_dc_and_strong_skin_limits():
    """The slab tends to 2/(sigma*t) at dc and zeta_c in strong skin effect."""
    sigma, t = 5.8e7, 35e-6
    freq = Frequency.from_f(jnp.array([0.0, 1e12]))
    conductor = _conductor(freq, sigma)

    zs = HollowayKuesterSlabShape().impedance(freq.w, conductor, t=t)

    assert jnp.allclose(zs[0], 2 / (sigma * t))
    assert jnp.allclose(zs[1], conductor.zs[1], rtol=1e-6)


def test_root_sum_square_slab_blends_resistance_and_keeps_half_space_reactance():
    """The blend takes its dimensions at call time and keeps the half-space X."""
    sigma, w, t, weight = 5.8e7, 1.55e-3, 35e-6, 963.7
    freq = Frequency.from_f(jnp.array([0.0, 10e6, 500e6]))
    conductor = _conductor(freq, sigma)

    zs = RootSumSquareSlabShape().impedance(freq.w, conductor, w=w, t=t, weight=weight)

    r_dc_sq = 1 / (sigma * w * t * weight)
    expected_r = jnp.sqrt(r_dc_sq**2 + jnp.real(conductor.zs) ** 2)
    expected = expected_r + 1j * jnp.imag(conductor.zs)
    assert jnp.allclose(zs, expected)


def test_root_sum_square_slab_reproduces_the_true_dc_resistance_under_its_weight():
    """Charged with the caller's weight, the dc floor is 1/(sigma*W*t) exactly."""
    sigma, w, t = 1 / 1.68e-8, 1.55e-3, 35e-6
    freq = Frequency.from_f(jnp.array([0.0]))
    conductor = _conductor(freq, sigma)
    (shape, weight), = WheelerCurrentDistribution().distribute(
        freq,
        MicrostripCrossSection(w=w, h=1.6e-3, t=t),
        _solved(jnp.full(freq.f.shape, 50.0)),
    )

    r_dc = shape.impedance(freq.w, conductor, w=w, t=t, weight=weight) * weight

    assert jnp.allclose(jnp.real(r_dc), 1 / (sigma * w * t), rtol=1e-12)


def test_exact_slab_wants_a_different_weight_from_wheelers():
    """The 2.99 normalisation finding: the two decompositions do not share a weight.

    The exact slab is referred to the total strip current, so its own weight
    is 1/(2W): its dc value 2/(sigma*t) times 1/(2W) is the true
    1/(sigma*W*t).  Wheeler's weight is larger because it covers the ground
    plane and the edge crowding as well as the strip, so charging the slab
    with it overstates the dc resistance by that same factor.  The literals
    are 35 um copper, rho = 1.68e-8 ohm m, on a 1.55 mm 50 ohm trace.
    """
    sigma, w, t = 1 / 1.68e-8, 1.55e-3, 35e-6
    freq = Frequency.from_f(jnp.array([0.0]))
    conductor = _conductor(freq, sigma)
    (_, wheeler_weight), = WheelerCurrentDistribution().distribute(
        freq,
        MicrostripCrossSection(w=w, h=1.6e-3, t=t),
        _solved(jnp.full(freq.f.shape, 50.0)),
    )
    slab_weight = 1 / (2 * w)

    assert jnp.allclose(wheeler_weight, 963.7, rtol=1e-4)
    assert jnp.allclose(slab_weight, 322.6, rtol=1e-4)
    assert jnp.allclose(wheeler_weight / slab_weight, 2.99, rtol=2e-3)

    zs = HollowayKuesterSlabShape().impedance(freq.w, conductor, t=t)
    assert jnp.allclose(zs * slab_weight, 1 / (sigma * w * t), rtol=1e-12)
    assert jnp.allclose(zs * wheeler_weight, 0.925, rtol=2e-3)
    assert jnp.allclose(1 / (sigma * w * t), 0.3097, rtol=2e-3)


def test_root_sum_square_slab_overstates_internal_inductance_below_one_skin_depth():
    """The blend keeps the half-space reactance, which the finite slab does not.

    Below t ~ delta the true internal reactance saturates at omega*mu*t/6
    while the blend's grows as sqrt(omega), so the entry is qualitatively
    wrong on internal inductance there.  The ratios are weight-independent
    -- both entries are charged with the same weight -- and are recorded
    here as the measured size of that limitation for 35 um copper.
    """
    sigma, t = 1 / 1.68e-8, 35e-6
    f = jnp.array([100e3, 1e6, 5e6, 10e6, 50e6])
    freq = Frequency.from_f(f)
    conductor = _conductor(freq, sigma)

    delta = jnp.sqrt(2 / (freq.w * mu_0 * sigma))
    # The tabulated t/delta and ratios are the measured values rounded to
    # two decimals, so they are compared to within half of that last place.
    assert jnp.allclose(t / delta, jnp.array([0.17, 0.54, 1.20, 1.70, 3.79]), atol=5e-3)

    blend_x = jnp.imag(conductor.zs)
    exact_x = jnp.imag(HollowayKuesterSlabShape().impedance(freq.w, conductor, t=t))

    ratio = blend_x / exact_x
    assert jnp.allclose(ratio, jnp.array([17.7, 5.6, 2.5, 1.8, 1.01]), atol=0.05)


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
    cross-section: it reaches the distribution through the typed record even
    though these planar entries depend on it only through the supplied
    characteristic impedance.
    """
    t, sigma = 35e-6, 1 / 1.68e-8
    freq = Frequency.from_f(jnp.array([10e6, 50e6]))
    conductor = _conductor(freq, sigma)
    blend, weight = WheelerCurrentDistribution().distribute(
        freq,
        MicrostripCrossSection(w=w, h=h, t=t),
        _solved(jnp.full(freq.f.shape, zc)),
    )[0]
    geometry = dict(w=w, t=t, weight=weight)

    actual = jnp.stack(
        [
            jnp.real(
                HollowayKuesterSlabShape().impedance(freq.w, conductor, **geometry)
                * weight
            ),
            jnp.real(HalfSpaceShape().impedance(freq.w, conductor, **geometry) * weight),
            jnp.real(blend.impedance(freq.w, conductor, **geometry) * weight),
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


def _schelkunoff_eq_74(w, sigma, a, t):
    """Schelkunoff eq. (74) referred to the inner surface, from scipy.

    Written in the paper's own grouping, with the exponential scaling of
    ``ive``/``kve`` carried explicitly so that nothing overflows in strong
    skin effect. ``ive(v, x) = I_v(x) e^{-|Re x|}`` and
    ``kve(v, x) = K_v(x) e^{x}``, and the common factor
    ``e^{|Re x_b| - x_a}`` cancels between numerator and denominator,
    leaving the bounded ``s`` below.
    """
    gamma = np.sqrt(1j * w * mu_0 * sigma)
    xa, xb = gamma * a, gamma * (a + t)
    s = np.exp(np.abs(xa.real) + xa - np.abs(xb.real) - xb)
    num = ive(0, xa) * kve(1, xb) * s + kve(0, xa) * ive(1, xb)
    den = ive(1, xb) * kve(1, xa) - ive(1, xa) * kve(1, xb) * s
    return (gamma / sigma) * num / den


@pytest.mark.parametrize(
    'f, t, rtol',
    [
        # Weak skin effect: the wall is a small fraction of a skin depth, so
        # the two surfaces are strongly coupled and the dc sheet resistance
        # dominates. This is the regime the previous strong-skin expression
        # was 33-43% wrong in.
        (1e3, 10e-6, 1e-12), (1e3, 50e-6, 1e-12), (1e3, 500e-6, 1e-12),
        (1e5, 500e-6, 1e-7),
        # Transition: wall and skin depth comparable.
        (1e5, 10e-6, 1e-7), (1e6, 50e-6, 1e-9),
        # Strong skin effect: the coupling term is exponentially small and
        # the tube is a curved half-space.
        (1e8, 10e-6, 1e-12), (1e10, 500e-6, 1e-12),
    ],
)
def test_schelkunoff_tube_matches_scipy_evaluation_of_eq_74(f, t, rtol):
    """The finite tube is eq. (74) itself, checked against scipy.

    The tolerances are the Bessel numerics of :mod:`pmrf.math.bessel`, not a
    physics allowance: 1e-7 is the seam of ``i0_over_i1`` at |x| = 20, which
    the 100 kHz cases sit near, and the rest are at machine level.
    """
    sigma, a = 5.8e7, 1.6e-3
    freq = Frequency.from_f(jnp.array([f]))
    conductor = _conductor(freq, sigma)

    zs = SchelkunoffTubeShape().impedance(freq.w, conductor, a=a, t=t)

    expected = _schelkunoff_eq_74(np.asarray(freq.w), sigma, a, t)
    assert np.allclose(zs, expected, rtol=rtol)


def test_schelkunoff_tube_dc_limit_is_the_tube_resistance_and_is_continuous():
    """The dc branch is the exact tube resistance the series limit reaches.

    The shape factor is 2*pi*a times the tube's dc resistance per unit
    length, 1/(sigma*pi*(b^2-a^2)) -- so 2a/(sigma*(b^2-a^2)), which is
    1/(sigma*t) only in the thin-wall limit. The earlier expression returned
    half of this from its series and the full 1/(sigma*t) from its dc
    branch, a factor-two discontinuity at the first nonzero frequency; the
    two must now agree.
    """
    sigma, a, t = 5.8e7, 1.6e-3, 50e-6
    freq = Frequency.from_f(jnp.array([0.0, 1e-3]))
    conductor = _conductor(freq, sigma)

    zs = SchelkunoffTubeShape().impedance(freq.w, conductor, a=a, t=t)

    exact_dc = 2 * a / (sigma * ((a + t) ** 2 - a**2))
    assert np.allclose(np.real(zs), exact_dc, rtol=1e-9)
    assert np.isclose(np.real(zs[1]), np.real(zs[0]), rtol=1e-9)
    # Continuous *from above*: dissipation never falls below the dc value.
    # The 1e-9 slack is the rounding of the two branches against each other,
    # not a physical allowance -- the difference at 1 mHz is 4e-10 relative.
    assert np.real(zs[1]) >= np.real(zs[0]) * (1 - 1e-9)


def test_schelkunoff_tube_infinite_wall_is_the_infinite_wall_limit():
    """An infinite wall decouples the surfaces: rho -> 0 leaves K0/K1."""
    freq = Frequency(start=10.0, stop=500.0, npoints=20, unit='MHz')
    conductor = _conductor(freq, 5.8e7)
    a = 1.475e-3

    infinite = SchelkunoffTubeShape().impedance(freq.w, conductor, a=a, t=jnp.inf)

    # 30 mm of copper is more than 10^4 skin depths at 10 MHz, so a finite
    # wall that thick must reproduce the analytic infinite-wall branch.
    thick = SchelkunoffTubeShape().impedance(freq.w, conductor, a=a, t=30e-3)
    assert jnp.all(jnp.isfinite(infinite))
    assert jnp.allclose(infinite, thick, rtol=1e-12)


def test_schelkunoff_tube_gradients_are_finite_at_dc_and_for_a_perfect_conductor():
    """Both unevaluable arguments must leave a usable gradient behind.

    dc sends gamma to zero and a perfect conductor sends it to infinity;
    either one propagates a nan into the gradient unless both branches are
    evaluated on safe arguments. As for the rod, the gradient is taken
    through :class:`BulkConductor`, which owns the dc derivative of
    $\zeta_c$ itself. A perfect conductor is checked on the shape alone:
    $\zeta_c=\sqrt{j\omega\mu/\sigma}$ has a nan $\omega$-derivative at
    $\sigma=\infty$ that every shape in this module inherits, so the
    conductor record is built directly to isolate the shape's own arithmetic.
    """
    a, t = 1.6e-3, 25e-6

    def resistance(sigma, f):
        freq = Frequency.from_f(jnp.atleast_1d(f))
        conductor = BulkConductor(sigma=sigma).properties(freq)
        return jnp.real(SchelkunoffTubeShape().impedance(freq.w, conductor, a=a, t=t))[0]

    for f in (0.0, 1e3, 1e9):
        d_sigma, d_f = jax.grad(resistance, argnums=(0, 1))(5.8e7, f)
        assert jnp.isfinite(d_sigma) and jnp.isfinite(d_f)
        # A perfect conductor is lossless at every frequency, dc included,
        # and the wall gradient there survives the infinite Bessel argument.
        freq = Frequency.from_f(jnp.array([f]))
        conductor = ConductorProperties(
            jnp.zeros_like(freq.w, dtype=complex), jnp.inf, 1.0
        )
        assert jnp.all(
            SchelkunoffTubeShape().impedance(freq.w, conductor, a=a, t=t) == 0
        )
        wall_gradient = jax.grad(
            lambda wall: jnp.real(
                SchelkunoffTubeShape().impedance(freq.w, conductor, a=a, t=wall)
            )[0]
        )(t)
        assert jnp.isfinite(wall_gradient)


@pytest.mark.parametrize(
    'shape, geometry',
    [
        (HollowayKuesterSlabShape(), dict(t=18e-6)),
        (SchelkunoffRodShape(), dict(a=20e-6)),
        (SchelkunoffTubeShape(), dict(a=1.475e-3, t=25e-6)),
        (SchelkunoffTubeShape(), dict(a=1.475e-3, t=jnp.inf)),
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
    """gamma = sqrt(j omega mu sigma) = (1+j)/delta, with finite dc gradient."""
    freq = Frequency.from_f(jnp.array([0.0, 1e6, 1e9]))
    conductor = BulkConductor(sigma=5.8e7, mu_r=2.0).properties(freq)

    gamma = conductor.gamma(omega=freq.w)

    w = np.asarray(freq.w)
    expected = np.sqrt(1j * w * mu_0 * 2.0 * 5.8e7)
    assert np.allclose(gamma, expected, rtol=1e-12)

    grad = jax.grad(
        lambda f: jnp.real(
            BulkConductor(sigma=5.8e7)
            .properties(Frequency.from_f(jnp.atleast_1d(f)))
            .gamma(omega=jnp.atleast_1d(2 * jnp.pi * f))
        )[0]
    )(0.0)
    assert jnp.isfinite(grad)
