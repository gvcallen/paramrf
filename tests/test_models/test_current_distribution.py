import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.materials import (
    BulkConductor,
    ConstantDielectric,
    HalfSpaceShape,
    HollowayKuesterSlabShape,
)
from pmrf.models import (
    CohnCurrentDistribution,
    HammerstadJensenMicrostripFormulation,
    MicrostripLine,
    TraceGroundCurrentDistribution,
    WheelerCurrentDistribution,
)
from pmrf.models.components.lines.formulations import _wheeler_conductor_loss_factor


def test_wheeler_distribution_returns_a_shape_and_frequency_resolved_weight():
    freq = Frequency(start=1.0, stop=10.0, npoints=2, unit="GHz")
    zc = jnp.array([50.0, 60.0])

    pairs = WheelerCurrentDistribution().distribute(freq, w=3e-3, zc=zc)

    shape, weight = pairs[0]
    assert isinstance(shape, HalfSpaceShape)
    assert jnp.allclose(weight, _wheeler_conductor_loss_factor(3e-3, zc))


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
        freq, zc=zc, w=2.655e-3, b=3.2e-3, t=35e-6, ep_r=jnp.array([2.2])
    )

    assert isinstance(pairs[0][0], HalfSpaceShape)
    assert pairs[0][1].shape == (1,)
    assert jnp.all(pairs[0][1] > 0)


def test_trace_ground_distribution_reproduces_published_ground_effective_widths():
    """The ground weight integrates Holloway--Kuester's current density."""
    freq = Frequency.from_f(jnp.array([10e9]))
    conductor = BulkConductor(sigma=5.8e7).properties(freq)

    for width_over_height, expected in [(1.94, 3.67), (1.80, 3.90), (0.44, 14.5)]:
        w = width_over_height * 1e-3
        pairs = TraceGroundCurrentDistribution().distribute(
            freq, zc=jnp.array([50.0]), w=w, h=1e-3, t=35e-6,
            conductor=conductor,
        )
        _, ground_weight = pairs[1]
        ki = jnp.exp(-1.2 * (50.0 / 376.730313668) ** 0.7)
        ground_effective_width = ki / ground_weight[0]
        assert jnp.isclose(ground_effective_width / w, expected, rtol=8e-3)


def test_trace_ground_distribution_returns_independent_trace_and_ground_pairs():
    freq = Frequency.from_f(jnp.array([0.0, 100e9]))
    conductor = BulkConductor(sigma=5.8e7).properties(freq)

    pairs = TraceGroundCurrentDistribution().distribute(
        freq, zc=jnp.array([50.0, 50.0]), w=3e-3, h=1.6e-3,
        t=35e-6, conductor=conductor,
    )

    assert isinstance(pairs[0][0], HollowayKuesterSlabShape)
    assert isinstance(pairs[1][0], HalfSpaceShape)
    assert jnp.isclose(pairs[0][1][0], 1 / (2 * 3e-3))
    assert pairs[0][1][1] > pairs[0][1][0]


def test_microstrip_split_default_has_true_trace_dc_and_documented_skin_limit():
    sigma, w, h, t = 5.8e7, 1.94e-3, 1e-3, 35e-6
    freq = Frequency.from_f(jnp.array([0.0, 1e12]))
    line = MicrostripLine(
        w=w, h=h, t=t, dielectric=ConstantDielectric(ep_r=4.3, tand=0.0),
        conductor=BulkConductor(sigma=sigma), dispersion=None, length=0.1,
    )

    resistance = line.immittance(freq).R
    ki = jnp.exp(-1.2 * (jnp.real(line._resolved_quasi_static(freq).zc) / 376.730313668) ** 0.7)
    expected_skin_weight = 2 * ki[-1] / w + ki[-1] / (3.6671708626520503 * w)
    wheeler_skin_weight = 2 * ki[-1] / w
    surface_resistance = jnp.real(line.substrate.conductor.properties(freq).zs[-1])

    assert isinstance(line.current_distribution, TraceGroundCurrentDistribution)
    assert jnp.isclose(resistance[0], 1 / (sigma * w * t), rtol=1e-6)
    assert jnp.isclose(resistance[-1], surface_resistance * expected_skin_weight, rtol=2e-6)
    assert jnp.isclose(expected_skin_weight / wheeler_skin_weight, 1.13634, rtol=2e-5)
