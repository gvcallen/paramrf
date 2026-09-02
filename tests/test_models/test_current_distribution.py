import jax.numpy as jnp
import pytest
from scipy.constants import epsilon_0, mu_0

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
    assert isinstance(shape, HalfSpaceShape)
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

    cross_section = MicrostripCrossSection(w=1.55e-3, h=1.6e-3, t=35e-6)
    default_shape, _ = WheelerCurrentDistribution().distribute(
        freq, cross_section, _solved(zc)
    )[0]
    assert isinstance(default_shape, RootSumSquareSlabShape)

    chosen = WheelerCurrentDistribution(slab_shape=HollowayKuesterSlabShape())
    chosen_shape, _ = chosen.distribute(freq, cross_section, _solved(zc))[0]
    assert isinstance(chosen_shape, HollowayKuesterSlabShape)

    # An unspecified thickness is still a half-space: there is no slab.
    unspecified, _ = chosen.distribute(
        freq, MicrostripCrossSection(w=1.55e-3, h=1.6e-3), _solved(zc)
    )[0]
    assert isinstance(unspecified, HalfSpaceShape)


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
