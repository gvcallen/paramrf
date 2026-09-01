import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.materials import (
    BulkConductor,
    ConstantDielectric,
    HalfSpaceShape,
    HollowayKuesterSlabShape,
    RootSumSquareSlabShape,
)
from pmrf.models import (
    CohnCurrentDistribution,
    HammerstadJensenMicrostripFormulation,
    MicrostripLine,
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


def test_wheeler_finite_thickness_slab_entry_is_a_selectable_field():
    """Which slab entry charges a stated thickness is a field, not a literal.

    The default is the root-sum-square blend, the only entry that satisfies
    both asymptotes under this distribution's frequency-independent weight;
    the exact strip-diffusion slab is reachable for a caller that supplies
    its own normalisation.
    """
    freq = Frequency.from_f(jnp.array([10e6]))
    zc = jnp.array([50.0])

    assert isinstance(WheelerCurrentDistribution().slab_shape, RootSumSquareSlabShape)

    default_shape, _ = WheelerCurrentDistribution().distribute(
        freq, zc=zc, w=1.55e-3, t=35e-6
    )[0]
    assert isinstance(default_shape, RootSumSquareSlabShape)

    chosen = WheelerCurrentDistribution(slab_shape=HollowayKuesterSlabShape())
    chosen_shape, _ = chosen.distribute(freq, zc=zc, w=1.55e-3, t=35e-6)[0]
    assert isinstance(chosen_shape, HollowayKuesterSlabShape)

    # An unspecified thickness is still a half-space: there is no slab.
    unspecified, _ = chosen.distribute(freq, zc=zc, w=1.55e-3)[0]
    assert isinstance(unspecified, HalfSpaceShape)
