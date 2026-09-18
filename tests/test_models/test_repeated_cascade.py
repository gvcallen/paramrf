# tests/test_models/test_repeated_cascade.py
import pytest
import jax
import jax.numpy as jnp

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.materials import ConstantDielectric, BulkConductor
from pmrf.models import (
    Cascade, RepeatedCascade, MicrostripLine, PhaseLine, Resistor,
)

N_SECTIONS = 6
TOTAL_LENGTH = 50e-3


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=20.0, npoints=21, unit='GHz')


@pytest.fixture
def member():
    """A microstrip section, one N'th of a taper."""
    return MicrostripLine(
        w=4e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=TOTAL_LENGTH / N_SECTIONS,
    )


@pytest.fixture
def widths():
    return jnp.linspace(3e-3, 6e-3, N_SECTIONS)


@pytest.fixture
def hand_built(member, widths):
    """The same taper as a `Cascade` of independently parametrised members."""
    return Cascade([prf.update(member, {'w': float(w)}) for w in widths])


# ---------------------------------------------------------
# Equivalence to a hand-built Cascade
# ---------------------------------------------------------

@pytest.mark.parametrize('method', ['a', 's'])
def test_matches_hand_built_cascade(member, widths, hand_built, freq, method):
    """The repeated form is the cascade of the equivalent independent members."""
    repeated = RepeatedCascade(member, {'w': widths}, method=method)

    # Both reductions are exact matrix algebra over the same sections, so the
    # only difference is floating-point ordering.
    assert jnp.allclose(repeated.s(freq), hand_built.s(freq), atol=1e-13)
    assert jnp.allclose(repeated.a(freq), hand_built.a(freq), atol=1e-12)


def test_matches_hand_built_cascade_multiple_repeated_params(member, freq):
    """Several repeated parameters vary together along the cascade."""
    widths = jnp.linspace(3e-3, 6e-3, N_SECTIONS)
    lengths = jnp.linspace(6e-3, 10e-3, N_SECTIONS)

    repeated = RepeatedCascade(member, {'w': widths, 'length': lengths})
    hand = Cascade([
        prf.update(member, {'w': float(w), 'length': float(l)})
        for w, l in zip(widths, lengths)
    ])

    assert jnp.allclose(repeated.s(freq), hand.s(freq), atol=1e-13)


def test_uniform_repeats_equal_one_long_line(freq):
    """
    n identical sections of a uniform line are that line at full length.

    This is a physical check that does not go through `Cascade` at all: the
    ideal line is exact, so subdividing it must change nothing.
    """
    theta = 137.0
    section = PhaseLine(theta=theta / N_SECTIONS, z0=75.0, f0=1e9)
    whole = PhaseLine(theta=theta, z0=75.0, f0=1e9)

    repeated = RepeatedCascade(
        section, {'theta': jnp.full((N_SECTIONS,), theta / N_SECTIONS)}
    )

    assert jnp.allclose(repeated.s(freq), whole.s(freq), atol=1e-12)


def test_single_repeat(member, freq):
    """A cascade of one repeat is the member itself."""
    repeated = RepeatedCascade(member, {'w': jnp.array([3.5e-3])})
    expected = prf.update(member, {'w': 3.5e-3})

    assert repeated.repeats == 1
    assert jnp.allclose(repeated.s(freq), expected.s(freq), atol=1e-13)


def test_repeat_order_is_port_1_to_port_2(freq):
    """Repeat 0 sits at port 1: reversing the values flips the network."""
    section = PhaseLine(theta=30.0, z0=50.0, f0=1e9)
    z0s = jnp.array([25.0, 50.0, 100.0])

    forward = RepeatedCascade(section, {'z0': z0s})
    reversed_ = RepeatedCascade(section, {'z0': z0s[::-1]})

    s_fwd = forward.s(freq)
    s_rev = reversed_.s(freq)

    assert jnp.allclose(s_fwd[:, 0, 0], s_rev[:, 1, 1], atol=1e-12)
    assert jnp.allclose(s_fwd[:, 1, 1], s_rev[:, 0, 0], atol=1e-12)


# ---------------------------------------------------------
# Structure: one name set, static repeat count
# ---------------------------------------------------------

def test_one_name_set_not_one_per_section(member, widths, hand_built):
    """
    The point of the model: the member is named once, not once per section.

    The hand-built cascade gives every section its own `w`; the repeated form
    has a single one.
    """
    hand_names = set(prf.params(hand_built))
    assert 'cascade[0].w' in hand_names and 'cascade[1].w' in hand_names

    repeated = RepeatedCascade(member, {'w': widths})
    names = set(prf.params(repeated))
    assert not any(name.startswith('cascade[') for name in names)
    assert 'model.w' in names
    assert 'values.w' in names


def test_repeated_parameter_is_not_free(member, widths):
    """
    A repeated parameter's value on the member is discarded, so it is fixed.

    Leaving it free would hand an optimiser a parameter that moves nothing,
    which is exactly the failure this model exists to remove.
    """
    base = prf.update(member, 'w', fixed=False)
    repeated = RepeatedCascade(base, {'w': widths})

    free = set(prf.params(repeated, free_only=True))
    assert 'model.w' not in free
    assert 'values.w' in free


def test_unrepeated_parameters_are_shared_and_stay_free(widths, freq):
    """A parameter not named in the mapping is one parameter for all repeats."""
    base = MicrostripLine(
        w=4e-3,
        h=prf.Bounded(1e-3, 3e-3, value=1.6e-3),
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=TOTAL_LENGTH / N_SECTIONS,
    )
    repeated = RepeatedCascade(base, {'w': widths})

    assert 'model.substrate.h' in prf.params(repeated, free_only=True)

    # Moving the single shared name moves every section.
    moved = prf.update(repeated, {'model.substrate.h': 2.5e-3})
    assert not jnp.allclose(moved.s(freq), repeated.s(freq))


def test_repeat_count_is_static(member, widths):
    """The repeat count is known at trace time, from the leading axis."""
    repeated = RepeatedCascade(member, {'w': widths})
    assert repeated.repeats == N_SECTIONS
    assert isinstance(repeated.repeats, int)


def test_nports_is_the_members(member, widths):
    repeated = RepeatedCascade(member, {'w': widths})
    assert repeated.nports == member.nports == 2


# ---------------------------------------------------------
# Construction errors
# ---------------------------------------------------------

def test_mismatched_repeat_axis_raises(member):
    """Arrays that disagree on their leading axis have no repeat count."""
    with pytest.raises(ValueError, match="disagree on the length of their repeat axis"):
        RepeatedCascade(member, {
            'w': jnp.linspace(3e-3, 6e-3, 4),
            'length': jnp.full((5,), 1e-3),
        })


def test_unknown_parameter_name_raises(member, widths):
    with pytest.raises(ValueError, match="Unknown parameter name"):
        RepeatedCascade(member, {'not_a_parameter': widths})


def test_empty_mapping_raises(member):
    with pytest.raises(ValueError, match="at least one repeated parameter"):
        RepeatedCascade(member, {})


def test_scalar_value_raises(member):
    """A scalar has no repeat axis; a shared value belongs on the member."""
    with pytest.raises(ValueError, match="leading repeat axis"):
        RepeatedCascade(member, {'w': 4e-3})


def test_odd_port_member_raises(widths):
    with pytest.raises(ValueError, match="2N-port"):
        RepeatedCascade(Resistor(R=prf.Bounded(1.0, 100.0, value=50.0)).terminated(),
                        {'R': widths})


# ---------------------------------------------------------
# The two axes: repeat axis versus parameter batch axis
# ---------------------------------------------------------

def test_sweep_batch_dimension_over_a_shared_parameter(member, widths, freq):
    """
    The model still works with a parameter batch axis, which is not the repeat axis.

    The batch axis is added outside the model by `prf.sweep` and survives into
    the result; the repeat axis is inside and is reduced away.
    """
    repeated = RepeatedCascade(member, {'w': widths})
    ep_rs = jnp.array([4.0, 4.3, 4.6])

    batched = prf.update(repeated, {'model.substrate.dielectric.ep_r': ep_rs})
    swept = prf.sweep(lambda m: m.s(freq), batched, template=repeated)

    assert swept.shape == (len(ep_rs), freq.npoints, 2, 2)

    for i, ep_r in enumerate(ep_rs):
        single = prf.update(repeated, {'model.substrate.dielectric.ep_r': float(ep_r)})
        assert jnp.allclose(swept[i], single.s(freq), atol=1e-14)


def test_sweep_batch_dimension_over_the_repeated_values(member, widths, freq):
    """A sweep over whole tapers: the batch axis leads, the repeat axis follows."""
    repeated = RepeatedCascade(member, {'w': widths})
    batch = jnp.stack([widths, widths * 1.2, widths * 0.8])

    swept = prf.sweep(
        lambda values: RepeatedCascade(member, {'w': values}).s(freq), batch
    )

    assert swept.shape == (3, freq.npoints, 2, 2)
    for i in range(3):
        single = RepeatedCascade(member, {'w': batch[i]})
        assert jnp.allclose(swept[i], single.s(freq), atol=1e-14)


# ---------------------------------------------------------
# Differentiability
# ---------------------------------------------------------

def test_gradient_flows_to_every_repeat(member, widths, freq):
    """
    Each repeat's value gets its own gradient, through the one traced member.

    This is what lets a fitted shape function drive the values.
    """
    def loss(values):
        s = RepeatedCascade(member, {'w': values}).s(freq)
        return jnp.sum(jnp.abs(s[:, 0, 0]) ** 2)

    grad = jax.grad(loss)(widths)

    assert grad.shape == widths.shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.any(jnp.abs(grad) > 0)
