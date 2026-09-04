import jax
import jax.numpy as jnp
import pytest
from scipy.constants import epsilon_0, mu_0

from pmrf.frequency import Frequency
from pmrf.materials import (
    BulkConductor,
    ConstantDielectric,
    HalfSpaceSurfaceImpedance,
    HollowayKuesterSlabSurfaceImpedance,
    RootSumSquareSlabSurfaceImpedance,
)
from pmrf.models import (
    CohnCurrentDistribution,
    HammerstadJensenMicrostripFormulation,
    IncrementalInductanceCurrentDistribution,
    MicrostripLine,
    WheelerCurrentDistribution,
    WheelerMicrostripFormulation,
)
from pmrf.models.components.lines.microstrip import MicrostripCrossSection
from pmrf.models.components.lines.planar import PlanarQuasiStaticResult
from pmrf.models.components.lines.stripline import StriplineCrossSection


def _solved(zc):
    """A minimal solved state carrying only the impedance a strategy reads."""
    ones = jnp.ones_like(zc)
    return PlanarQuasiStaticResult(
        ep_eff=ones, zc=zc, w_eff=ones, shunt_conductance_factor=ones
    )


def test_wheeler_distribution_returns_a_shape_and_frequency_resolved_weight():
    freq = Frequency(start=1.0, stop=10.0, npoints=2, unit="GHz")
    zc = jnp.array([50.0, 60.0])

    pairs = WheelerCurrentDistribution().distribute(
        freq, MicrostripCrossSection(w=3e-3, h=1.6e-3), _solved(zc)
    )

    z0 = jnp.sqrt(mu_0 / epsilon_0)
    expected = 2 / 3e-3 * jnp.exp(-1.2 * (zc / z0) ** 0.7)
    shape, weight = pairs[0]
    assert isinstance(shape, HalfSpaceSurfaceImpedance)
    assert jnp.allclose(weight, expected)


def test_current_distribution_can_be_selected_independently_of_microstrip_formulation():
    line = MicrostripLine(
        formulation=HammerstadJensenMicrostripFormulation(),
        current_distribution=WheelerCurrentDistribution(),
        dielectric=ConstantDielectric(ep_r=4.3),
        conductor=BulkConductor(),
        length=0.1,
    )

    assert isinstance(line.current_distribution, WheelerCurrentDistribution)


def test_cohn_distribution_uses_a_surface_pair():
    freq = Frequency.from_f(jnp.array([10e9]))
    zc = jnp.array([50.0])

    pairs = CohnCurrentDistribution().distribute(
        freq,
        StriplineCrossSection(w=2.655e-3, b=3.2e-3, t=35e-6, ep_r=jnp.array([2.2])),
        _solved(zc),
    )

    assert isinstance(pairs[0][0], HalfSpaceSurfaceImpedance)
    assert pairs[0][1].shape == (1,)
    assert jnp.all(pairs[0][1] > 0)


def test_wheeler_finite_thickness_slab_entry_is_a_selectable_field():
    """Which slab entry charges a stated thickness is a field, not a literal.

    The default is the root-sum-square blend, the only entry that satisfies
    both asymptotes under this distribution's frequency-independent weight;
    the exact strip-diffusion slab is reachable for a caller that supplies
    its own normalisation.
    """
    freq = Frequency.from_f(jnp.array([10e6]))
    zc = jnp.array([50.0])

    assert isinstance(WheelerCurrentDistribution().slab_impedance, RootSumSquareSlabSurfaceImpedance)

    cross_section = MicrostripCrossSection(w=1.55e-3, h=1.6e-3, t=35e-6)
    default_shape, _ = WheelerCurrentDistribution().distribute(
        freq, cross_section, _solved(zc)
    )[0]
    assert isinstance(default_shape, RootSumSquareSlabSurfaceImpedance)

    chosen = WheelerCurrentDistribution(slab_impedance=HollowayKuesterSlabSurfaceImpedance())
    chosen_shape, _ = chosen.distribute(freq, cross_section, _solved(zc))[0]
    assert isinstance(chosen_shape, HollowayKuesterSlabSurfaceImpedance)

    # An unspecified thickness is still a half-space: there is no slab.
    unspecified, _ = chosen.distribute(
        freq, MicrostripCrossSection(w=1.55e-3, h=1.6e-3), _solved(zc)
    )[0]
    assert isinstance(unspecified, HalfSpaceSurfaceImpedance)


def test_pairing_a_distribution_with_the_wrong_family_is_a_typed_failure():
    """A wrong-family pairing fails at the strategy boundary, not in binding.

    The families do not share a cross-section record, so the check is a real
    type check with a message naming both sides -- not a TypeError about a
    missing or unexpected keyword argument deep inside the call.
    """
    freq = Frequency.from_f(jnp.array([10e9]))
    solved = _solved(jnp.array([50.0]))
    stripline = StriplineCrossSection(w=2.655e-3, b=3.2e-3, t=35e-6, ep_r=jnp.array([2.2]))
    microstrip = MicrostripCrossSection(w=1.55e-3, h=1.6e-3, t=35e-6)

    with pytest.raises(TypeError, match="MicrostripCrossSection strategy"):
        WheelerCurrentDistribution().distribute(freq, stripline, solved)

    with pytest.raises(TypeError, match="StriplineCrossSection strategy"):
        CohnCurrentDistribution().distribute(freq, microstrip, solved)


def test_substrate_height_reaches_the_microstrip_distribution():
    """The record carries h, which edge-shape and ground-share models need."""
    freq = Frequency.from_f(jnp.array([10e9]))
    seen = {}

    class RecordingDistribution(WheelerCurrentDistribution):
        def _distribute(self, freq, cross_section, quasi_static):
            seen["h"] = cross_section.h
            return super()._distribute(freq, cross_section, quasi_static)

    line = MicrostripLine(
        w=1.55e-3, h=0.8e-3, length=0.1,
        current_distribution=RecordingDistribution(),
    )
    line.immittance(freq)

    assert seen["h"] == line.substrate.h


# The recession derivative against Wheeler's 1942 fit, at ep_r = 4.335,
# h = 1.6 mm, t = 35 um, Hammerstad-Jensen. Columns are
# (w/h, w, Re(Zc), Wheeler 1942 k_c, incremental-inductance k_c).
_INCREMENTAL_INDUCTANCE_CASES = [
    (2.5, 4.0e-3, 42.28, 385.69, 318.34),
    (5.0, 8.0e-3, 25.96, 207.89, 178.62),
    (10.0, 16.0e-3, 14.83, 110.35, 98.47),
    (20.0, 32.0e-3, 8.06, 57.62, 53.24),
    (50.0, 80.0e-3, 3.43, 23.91, 22.91),
]


def _incremental_inductance_weights(w, h=1.6e-3, t=35e-6, ep_r=4.335):
    """Return ``(Re(Zc), Wheeler 1942 weight, recession-derivative weight)``."""
    freq = Frequency.from_f(jnp.array([1e9]))
    formulation = HammerstadJensenMicrostripFormulation()
    quasi_static = formulation.quasi_static(
        w=w, h=h, t=t, ep_r=jnp.array([ep_r + 0j])
    )
    cross_section = MicrostripCrossSection(
        w=jnp.asarray(w), h=jnp.asarray(h), t=jnp.asarray(t)
    )
    (_, wheeler), = WheelerCurrentDistribution().distribute(
        freq, cross_section, quasi_static
    )
    (_, incremental), = IncrementalInductanceCurrentDistribution(
        formulation=formulation
    ).distribute(freq, cross_section, quasi_static)
    return jnp.real(quasi_static.zc)[0], wheeler[0], incremental[0]


@pytest.mark.parametrize(
    "w_over_h, w, zc_real, wheeler_kc, incremental_kc",
    _INCREMENTAL_INDUCTANCE_CASES,
)
def test_incremental_inductance_reproduces_the_recession_derivative_table(
    w_over_h, w, zc_real, wheeler_kc, incremental_kc
):
    """The derivative is evaluated, not fitted, so each case is exact.

    Both columns are closed-form functions of the geometry, so the tolerance
    is a rounding tolerance on the tabulated digits rather than a physics
    allowance: each expected value is quoted to two decimals, so half a unit
    in the last place is the whole budget. An independent 2D quasi-static
    field solve agrees with the right-hand column to within 0.76% across
    these five cases -- recorded here, not asserted, because it cannot be
    reproduced in-repo.
    """
    zc, wheeler, incremental = _incremental_inductance_weights(w)

    assert jnp.abs(zc - zc_real) <= 0.005
    assert jnp.abs(wheeler - wheeler_kc) <= 0.005
    assert jnp.abs(incremental - incremental_kc) <= 0.005

    # Wheeler's 1942 fit is high against the derivative, by 21% at w/h = 2.5
    # closing to 4% at w/h = 50. Recorded with its sign: a separate
    # trace/ground split reports the opposite sign, and that contradiction is
    # unresolved.
    assert incremental < wheeler


def test_incremental_inductance_wide_line_does_not_approach_two_over_w():
    """The wide-line limit is not Wheeler's 2/W prefactor, and should not be.

    2/W assumes a zero-thickness strip and neglects fringing. With fringing
    the air-filled capacitance is larger, so
    k_c = -eps_0 C^-2 dC/dn falls below 2/W. At w/h = 50 the ratio is 0.916
    and it is still falling, so a test that asserted convergence to 2/W would
    be asserting the wrong physics.
    """
    _, _, incremental = _incremental_inductance_weights(80.0e-3)

    ratio = incremental / (2 / 80.0e-3)
    assert 0.90 < ratio < 0.93
    assert jnp.abs(ratio - 0.9165) <= 0.001


def test_incremental_inductance_weight_is_differentiable_in_w_h_and_t():
    """The fitting path needs gradients through the geometry weight itself."""
    freq = Frequency.from_f(jnp.array([1e9]))
    distribution = IncrementalInductanceCurrentDistribution()

    def weight(w, h, t):
        quasi_static = HammerstadJensenMicrostripFormulation().quasi_static(
            w=w, h=h, t=t, ep_r=jnp.array([4.335 + 0j])
        )
        cross_section = MicrostripCrossSection(w=w, h=h, t=t)
        (_, value), = distribution.distribute(freq, cross_section, quasi_static)
        return value[0]

    args = (jnp.asarray(4.0e-3), jnp.asarray(1.6e-3), jnp.asarray(35e-6))
    grads = jax.grad(weight, argnums=(0, 1, 2))(*args)

    assert all(jnp.isfinite(g) for g in grads)
    # A wider strip spreads the current, so the weight falls with w.
    assert grads[0] < 0

    # And the derivative is a real one: a 1% step in w moves the weight by
    # what the gradient predicts, to better than 1%.
    step = 0.01 * args[0]
    predicted = weight(*args) + grads[0] * step
    actual = weight(args[0] + step, args[1], args[2])
    assert jnp.abs(predicted - actual) < 0.01 * jnp.abs(actual - weight(*args))


def test_incremental_inductance_refuses_a_thickness_blind_pairing():
    """A formulation that ignores t cannot see the strip thin.

    The top face and sidewalls would drop out of the weight entirely -- about
    30% low at w/h = 2.5, an error of similar size to the one this
    distribution exists to fix, and equally invisible behind a free sigma.
    """
    freq = Frequency.from_f(jnp.array([1e9]))
    quasi_static = HammerstadJensenMicrostripFormulation().quasi_static(
        w=4e-3, h=1.6e-3, t=35e-6, ep_r=jnp.array([4.335 + 0j])
    )
    cross_section = MicrostripCrossSection(w=4e-3, h=1.6e-3, t=35e-6)

    with pytest.raises(ValueError, match="thickness-aware formulation"):
        IncrementalInductanceCurrentDistribution(
            formulation=WheelerMicrostripFormulation()
        ).distribute(freq, cross_section, quasi_static)

    assert HammerstadJensenMicrostripFormulation.thickness_aware
    assert not WheelerMicrostripFormulation.thickness_aware


def test_incremental_inductance_refuses_an_unspecified_thickness():
    """Without t there is no top face or sidewall to recede."""
    freq = Frequency.from_f(jnp.array([1e9]))
    quasi_static = HammerstadJensenMicrostripFormulation().quasi_static(
        w=4e-3, h=1.6e-3, t=None, ep_r=jnp.array([4.335 + 0j])
    )

    with pytest.raises(ValueError, match="specified strip thickness"):
        IncrementalInductanceCurrentDistribution().distribute(
            freq, MicrostripCrossSection(w=4e-3, h=1.6e-3), quasi_static
        )


def test_incremental_inductance_is_selectable_on_a_line_without_changing_the_default():
    """It ships as a peer; WheelerCurrentDistribution stays the default."""
    freq = Frequency.from_f(jnp.array([1e9]))

    assert isinstance(
        MicrostripLine(length=0.1).current_distribution, WheelerCurrentDistribution
    )

    line = MicrostripLine(
        w=4e-3, h=1.6e-3, t=35e-6, length=0.1,
        current_distribution=IncrementalInductanceCurrentDistribution(),
    )
    default = MicrostripLine(w=4e-3, h=1.6e-3, t=35e-6, length=0.1)

    assert isinstance(
        line.current_distribution, IncrementalInductanceCurrentDistribution
    )
    assert isinstance(
        IncrementalInductanceCurrentDistribution().slab_impedance,
        RootSumSquareSlabSurfaceImpedance,
    )

    # Lower k_c means lower conductor loss than the 1942 fit gives.
    assert jnp.all(
        jnp.real(line.immittance(freq).Z) < jnp.real(default.immittance(freq).Z)
    )
