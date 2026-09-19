# tests/test_models/test_profiled_line.py
"""
Tests for `ProfiledLine`.

The physics here is validated against Pozar's closed form for the exponential taper
and against the integrator's own convergence order, rather than against scikit-rf as
`AGENTS.md` asks for: scikit-rf has no non-uniform line model to compare with, so
there is no external implementation to disagree with. The closed form is taken from
the cited textbook rather than from recorded ParamRF output, which is what that rule
exists to prevent.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.materials import ConstantDielectric, BulkConductor
from pmrf.models import (
    ExponentialProfile,
    LinearProfile,
    MicrostripLine,
    ProfiledLine,
    RepeatedCascade,
    RLGCLine,
    TransmissionLine,
    AbstractBuilder,
    AbstractUniformLine,
)

TOTAL_LENGTH = 50e-3


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=20.0, npoints=21, unit='GHz')


@pytest.fixture
def base():
    """A microstrip line, whose width a taper varies."""
    return MicrostripLine(
        w=4e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=TOTAL_LENGTH,
    )


@pytest.fixture
def taper():
    return ExponentialProfile(start=2e-3, end=8e-3)


@pytest.fixture
def line(base, taper):
    return ProfiledLine(base, {'w': taper}, n=16)


# ---------------------------------------------------------
# Construction: two ways in, one representation
# ---------------------------------------------------------

def test_accepts_a_constructed_line(base, taper):
    line = ProfiledLine(base, {'w': taper})
    assert isinstance(line.base, MicrostripLine)
    assert line.profiles['w'].evaluate(0.0) == pytest.approx(2e-3)


def test_accepts_a_class_plus_keywords(taper):
    """The class form builds the base from the plain keywords."""
    line = ProfiledLine(
        MicrostripLine,
        {'w': taper},
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        length=TOTAL_LENGTH,
    )
    assert isinstance(line.base, MicrostripLine)
    assert float(line.length) == pytest.approx(TOTAL_LENGTH)


def test_both_forms_store_the_same_representation(base, taper):
    """Whichever way in, one base model plus one target-to-profile mapping."""
    from_instance = ProfiledLine(base, {'w': taper}, n=8)
    from_class = ProfiledLine(
        MicrostripLine,
        {'w': taper},
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=TOTAL_LENGTH,
        n=8,
    )
    freq = Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

    assert sorted(prf.params(from_instance)) == sorted(prf.params(from_class))
    assert jnp.allclose(from_instance.s(freq), from_class.s(freq))


def test_class_form_leaves_the_base_unnamed(taper):
    """Parameter paths flatten to the container root, with no base name in them."""
    line = ProfiledLine(
        MicrostripLine, {'w': taper}, h=1.6e-3, length=TOTAL_LENGTH,
    )
    assert line.base.name is None
    assert 'length' in prf.params(line)


def test_keywords_are_rejected_alongside_a_constructed_base(base, taper):
    with pytest.raises(TypeError, match='constructed base'):
        ProfiledLine(base, {'w': taper}, h=1.6e-3)


def test_a_non_model_base_is_rejected(taper):
    with pytest.raises(TypeError, match='pmrf.Model'):
        ProfiledLine(object(), {'w': taper})


def test_no_profiles_is_rejected(base):
    with pytest.raises(ValueError, match='at least one profiled target'):
        ProfiledLine(base, {})


# ---------------------------------------------------------
# Reserved container keywords
# ---------------------------------------------------------

def test_reserved_keywords_belong_to_the_container(taper):
    """`n` configures the container, never the base."""
    line = ProfiledLine(
        MicrostripLine, {'w': taper}, h=1.6e-3, length=TOTAL_LENGTH, n=7,
    )
    assert line.n == 7
    assert line.build().repeats == 7


def test_a_reserved_keyword_is_not_forwarded_to_the_base(taper):
    """`extrapolate` is the container's from the start, before it does anything."""
    with pytest.raises(ValueError, match='belong to ProfiledLine'):
        ProfiledLine(
            MicrostripLine, {'w': taper}, h=1.6e-3, length=TOTAL_LENGTH,
            extrapolate=True,
        )


def test_a_base_class_colliding_with_a_reserved_keyword_raises(taper):
    """A base declaring `n` cannot be told apart from the container's own keyword."""
    class LineWithN(RLGCLine):
        n: prf.Param = prf.param(default=1.0)

    with pytest.raises(ValueError, match="reserves for itself"):
        ProfiledLine(LineWithN, {'L': taper}, length=TOTAL_LENGTH, n=4)


def test_reserved_keyword_set_is_documented():
    """The reserved set is public, so the error message and the docs cannot drift."""
    from pmrf.models.components.lines.nonuniform import RESERVED_KEYWORDS

    assert set(RESERVED_KEYWORDS) == {'n', 'extrapolate', 'name', 'metadata'}


# ---------------------------------------------------------
# Targets
# ---------------------------------------------------------

def test_a_nested_dotted_target_resolves(base):
    """A target is any exact dotted path naming a Param on the base."""
    line = ProfiledLine(
        base, {'substrate.dielectric.ep_r': LinearProfile(start=4.0, end=4.6)},
    )
    assert float(line.at(0.0).substrate.dielectric.ep_r) == pytest.approx(4.0)


@pytest.mark.parametrize('target', ['w*', '*', 'substrate.*', 'w?'])
def test_a_glob_target_is_rejected(base, taper, target):
    """A glob matching several parameters would silently share one shape object."""
    with pytest.raises(ValueError, match='glob'):
        ProfiledLine(base, {target: taper})


@pytest.mark.parametrize('target', [('w',), 42, None])
def test_a_non_string_target_is_rejected(base, taper, target):
    with pytest.raises(TypeError, match='exact dotted parameter paths'):
        ProfiledLine(base, {target: taper})


def test_an_unknown_target_is_rejected(base, taper):
    with pytest.raises(ValueError, match='do not name a parameter'):
        ProfiledLine(base, {'not_a_parameter': taper})


def test_a_target_that_is_not_a_parameter_is_rejected(base, taper):
    """A sub-model is a path, but not a `Param`, so it cannot be profiled."""
    with pytest.raises(ValueError, match='do not name a parameter'):
        ProfiledLine(base, {'substrate': taper})


@pytest.mark.parametrize('profile', [lambda t: t, [1.0, 2.0], 3.0])
def test_a_callable_or_sequence_profile_is_rejected(base, profile):
    with pytest.raises(TypeError, match='AbstractProfile'):
        ProfiledLine(base, {'w': profile})


# ---------------------------------------------------------
# Length
# ---------------------------------------------------------

def test_length_is_rejected_as_a_target(base, taper):
    with pytest.raises(ValueError, match='not profilable'):
        ProfiledLine(base, {'length': taper})


def test_length_forwards_to_the_base_and_is_the_total(line):
    assert float(line.length) == pytest.approx(TOTAL_LENGTH)
    assert 'length' not in type(line).__dataclass_fields__


def test_length_is_read_only(line):
    with pytest.raises(AttributeError):
        line.length = 1.0


def test_sections_each_carry_one_nth_of_the_total_length(line):
    assert float(line.build().model.length) == pytest.approx(TOTAL_LENGTH / line.n)


# ---------------------------------------------------------
# Parameters and names
# ---------------------------------------------------------

def test_coefficients_are_named_under_their_target(base, taper):
    line = ProfiledLine(base, {'w': taper})
    assert {'w.start', 'w.end'} <= set(prf.params(line))


def test_a_nested_target_names_its_coefficients_under_its_whole_path(base):
    line = ProfiledLine(
        base, {'substrate.dielectric.ep_r': LinearProfile(start=4.0, end=4.6)},
    )
    assert {'substrate.dielectric.ep_r.start', 'substrate.dielectric.ep_r.end'} <= set(
        prf.params(line)
    )


def test_container_field_layout_never_appears_in_a_name(line):
    """No `base.` or `profiles['w'].` prefix leaks out of the container."""
    names = set(prf.params(line))
    assert not any(name.startswith(('base', 'profiles', '[')) for name in names)
    assert 'w.start' in names


def test_a_profiled_target_is_not_a_parameter_any_more(base, taper):
    """
    A profiled target has no name at all: its profile's coefficients replace it.

    The `Param` stays where it is in the base tree -- `build` writes each profile's
    value back through the driven field's own converter and constraint, and ADR-0004
    keeps the base tree as built -- but the container declares it shadowed, so it is
    not a parameter of the profiled line.
    """
    line = ProfiledLine(base, {'w': taper})

    names = set(prf.params(line))
    assert 'w' not in names
    assert {'w.start', 'w.end'} <= names
    # Untouched parameters of the base are unaffected.
    assert {'length', 'substrate.h'} <= names


def test_profiled_coefficients_are_what_is_free(base):
    """What a fit moves is the shape, not the target the shape drives."""
    free_base = prf.update(base, '*', fixed=False)
    free_taper = ExponentialProfile(
        start=prf.Bounded(1e-3, 9e-3, value=2e-3),
        end=prf.Bounded(1e-3, 9e-3, value=8e-3),
    )
    line = ProfiledLine(free_base, {'w': free_taper})

    free = set(prf.params(line, free_only=True))
    assert 'w' not in free
    assert {'w.start', 'w.end'} <= free


def test_setting_a_shadowed_target_is_not_a_silent_no_op(freq):
    """
    Regression: a target's own value drives nothing, so naming it must fail loudly.

    Before the target was shadowed, `prf.update(line, {'w': ...})` resolved to a dead
    `Param` and changed the S-parameters by exactly zero, which is the failure mode a
    fit or a sweep aimed at `w` would have hit -- silently optimising nothing.
    """
    line = ProfiledLine(
        MicrostripLine,
        {'w': LinearProfile(start=2e-3, end=6e-3)},
        h=1.6e-3,
        length=TOTAL_LENGTH,
        n=8,
    )

    with pytest.raises(ValueError, match='[Uu]nknown parameter name'):
        prf.update(line, {'w': 99e-3})
    with pytest.raises(ValueError, match='[Uu]nknown parameter name'):
        prf.params(line, 'w')

    # The coefficient that replaced it does move the answer.
    moved = prf.update(line, {'w.start': 3e-3})
    assert float(jnp.max(jnp.abs(moved.s(freq) - line.s(freq)))) > 1e-3


def test_shadowing_does_not_change_any_other_model(base):
    """The naming layer is shared: a model that shadows nothing is untouched."""
    from pmrf.models import Cascade, Resistor

    assert 'w' in set(prf.params(base))
    assert set(prf.params(Cascade([base, base]))) == {
        f'cascade[{index}].{name}' for index in (0, 1) for name in prf.params(base)
    }
    assert set(prf.params(Resistor(50.0))) == {'R'}


def test_two_profiled_lines_shadow_their_own_targets(base, taper):
    """Shadowing is per position, so each container hides its own target."""
    from pmrf.models import Cascade

    cascade = Cascade([
        ProfiledLine(base, {'w': taper}, name='t1'),
        ProfiledLine(base, {'w': taper}, name='t2'),
    ])

    names = set(prf.params(cascade))
    assert {'t1_w.start', 't1_w.end', 't2_w.start', 't2_w.end'} <= names
    assert not any(name.endswith('.w') or name == 'w' for name in names)


def test_a_shadowed_target_is_still_written_through_its_own_field(base):
    """Shadowing hides a name; it does not bypass the driven field's constraint.

    `w` is constrained positive by `MicrostripLine`, and the profile's value goes in
    through that constraint, so an impossible profile is still rejected there.
    """
    line = ProfiledLine(base, {'w': LinearProfile(start=2e-3, end=-6e-3)})

    # The message names the field's own constraint, which is the thing doing the
    # rejecting: a shadowed target is still written in through its own field.
    with pytest.raises(Exception, match='falls outside the constraint'):
        line.at(1.0)


def test_a_value_for_a_profiled_target_raises(taper):
    """Silently discarded input is not worth the afternoon it costs."""
    with pytest.raises(ValueError, match='both a value and a profile'):
        ProfiledLine(
            MicrostripLine, {'w': taper}, w=4e-3, h=1.6e-3, length=TOTAL_LENGTH,
        )


def test_a_discarded_field_default_does_not_raise():
    """The user did not type it, so there is nothing to warn them about."""
    line = ProfiledLine(
        RLGCLine, {'L': ExponentialProfile(start=200e-9, end=400e-9)},
        length=TOTAL_LENGTH,
    )
    assert float(line.at(0.0).L) == pytest.approx(200e-9)


def test_a_scaled_target_is_rejected(base, taper):
    """A profile returns a physical value, so a target's own scale would reinterpret it."""
    scaled = prf.update(base, 'w', prf.Bounded(1.0, 9.0, value=4.0, scale=1e-3))
    with pytest.raises(ValueError, match='declare a scale'):
        ProfiledLine(scaled, {'w': taper})


def test_update_reaches_a_coefficient_by_name(line):
    updated = prf.update(line, 'w.start', value=3e-3)
    assert float(updated.at(0.0).w) == pytest.approx(3e-3)


def test_globs_over_coefficients(base):
    """`w.*` is one target's coefficients; `*.start` is every profile's start."""
    line = ProfiledLine(
        base,
        {
            'w': LinearProfile(start=2e-3, end=8e-3),
            'substrate.dielectric.ep_r': LinearProfile(start=4.0, end=4.6),
        },
    )

    per_target = prf.update(line, 'w.*', value=5e-3)
    assert float(per_target.at(0.0).w) == pytest.approx(5e-3)
    assert float(per_target.at(1.0).w) == pytest.approx(5e-3)
    # The other target is untouched.
    assert float(per_target.at(1.0).substrate.dielectric.ep_r) == pytest.approx(4.6)

    starts = prf.params(line, '*.start')
    assert set(starts) == {'w.start', 'substrate.dielectric.ep_r.start'}


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def test_is_a_transmission_line_and_a_builder(line):
    assert isinstance(line, TransmissionLine)
    assert isinstance(line, AbstractBuilder)


def test_is_deliberately_not_a_uniform_line(line):
    """A Zc that is silently the value at one position invites misuse."""
    assert not isinstance(line, AbstractUniformLine)
    assert not hasattr(line, 'zc_and_gammaL')


def test_build_returns_a_repeated_cascade(line):
    built = line.build()
    assert isinstance(built, RepeatedCascade)
    assert built.repeats == line.n


def test_representations_delegate_consistently(line, freq):
    """Every representation comes from the built cascade, so none can disagree."""
    built = line.build()

    assert jnp.allclose(line.s(freq), built.s(freq))
    assert jnp.allclose(line.a(freq), built.a(freq))
    assert jnp.allclose(line.y(freq), built.y(freq))
    assert jnp.allclose(line.z(freq), built.z(freq))
    assert jnp.allclose(line.mna(freq).Y, built.mna(freq).Y)
    assert line.nports == 2


def test_sections_are_sampled_at_the_midpoints(line):
    """The k'th section is the base at t = (k + 1/2)/n."""
    built = line.build()
    midpoints = (jnp.arange(line.n) + 0.5) / line.n

    assert jnp.allclose(built.values['w'], line.profiles['w'].evaluate(midpoints))


def test_jits(line, freq):
    jitted = jax.jit(lambda m: m.s(freq))
    assert jnp.allclose(jitted(line), line.s(freq))


# ---------------------------------------------------------
# Introspection: at(t)
# ---------------------------------------------------------

def test_at_returns_the_substituted_base(base, taper):
    line = ProfiledLine(base, {'w': taper})
    at_half = line.at(0.5)

    assert isinstance(at_half, MicrostripLine)
    assert float(at_half.w) == pytest.approx(taper.evaluate(0.5))
    # Everything else is the base, untouched, including the *total* length.
    assert float(at_half.length) == pytest.approx(TOTAL_LENGTH)


def test_at_vmaps_over_an_array_of_positions(line):
    positions = jnp.linspace(0.0, 1.0, 5)
    everywhere = line.at(positions)

    assert jnp.allclose(
        jnp.asarray(everywhere.w), line.profiles['w'].evaluate(positions)
    )


def test_at_composes_with_the_base_interface(base, taper, freq):
    """`line.at(t)` is a line, so its own methods answer for that position."""
    line = ProfiledLine(base, {'w': taper})
    zc = line.at(0.5).zc(freq)

    assert zc.shape == (freq.npoints,)


def test_at_matches_the_evaluated_sections(line, freq):
    """The substitution `at` exposes is the one the evaluator performs."""
    midpoints = (jnp.arange(line.n) + 0.5) / line.n
    assert jnp.allclose(jnp.asarray(line.at(midpoints).w), line.build().values['w'])


# ---------------------------------------------------------
# Physics
# ---------------------------------------------------------

def test_a_constant_profile_is_the_uniform_line(base, freq):
    """
    A profile that does not vary must reproduce the uniform line exactly.

    Subdividing a uniform line into n sections of length L/n and cascading them is
    exact matrix algebra, so this checks the container's bookkeeping -- the length
    division, the midpoint sampling and the substitution -- with no discretisation
    error in the way.
    """
    line = ProfiledLine(base, {'w': LinearProfile(start=4e-3, end=4e-3)}, n=9)

    # Tolerance is the accumulated round-off of a 9-term matrix product, not physics.
    assert jnp.allclose(line.s(freq), base.s(freq), atol=1e-12)


def test_an_exponential_taper_matches_its_closed_form():
    r"""
    An exponential taper's reflection matches Pozar's closed form.

    A lossless line with $L(t)$ and $C(t)$ exponential with reciprocal ratios has
    $Z_c(t) = \sqrt{L/C}$ exponential and $\beta = \omega\sqrt{LC}$ constant, which is
    exactly the taper of Pozar (4th ed.) eq. 5.71:

    $$\Gamma = \tfrac{1}{2}\ln\frac{Z_1}{Z_0}\, e^{-j\beta L}
               \frac{\sin \beta L}{\beta L}$$

    That form comes from the small-reflection theory of Section 5.8, which neglects
    multiple reflections, so it is an approximation of order $\Gamma^2$. The ratio
    here is deliberately small (1.2) to keep that model error below the tolerance:
    $\Gamma \lesssim 0.09$, so the neglected term is a few times $10^{-3}$ of the
    reflection, and the section count is large enough that the discretisation error
    is far below it.
    """
    z0, z1 = 50.0, 60.0
    velocity = 2e8
    length = 50e-3

    line = ProfiledLine(
        RLGCLine,
        {
            'L': ExponentialProfile(start=z0 / velocity, end=z1 / velocity),
            'C': ExponentialProfile(start=1 / (z0 * velocity), end=1 / (z1 * velocity)),
        },
        R=0.0,
        G=0.0,
        length=length,
        n=256,
    )

    freq = Frequency(start=0.5, stop=8.0, npoints=25, unit='GHz')
    # Port 2 is referenced to the taper's own terminating impedance: the closed form
    # is the taper's reflection into a matched load, not the step at a 50 ohm port.
    s11 = line.s(freq, z0=jnp.array([z0, z1]))[:, 0, 0]

    beta_l = 2 * np.pi * freq.f / velocity * length
    closed_form = 0.5 * np.log(z1 / z0) * np.exp(-1j * beta_l) * np.sinc(beta_l / np.pi)

    # Absolute tolerance on Gamma. The measured disagreement is 2.2e-4, which is the
    # small-reflection model error of the closed form (order Gamma^2, with a peak
    # Gamma of 0.09) and not a discretisation error: raising n does not move it.
    assert np.allclose(np.asarray(s11), closed_form, atol=5e-4)


def test_convergence_is_second_order():
    """
    Halving the section length quarters the error.

    The midpoint sampling with exact hyperbolic sections is the exponential midpoint
    rule, an order-2 Magnus integrator, so the error expands in even powers of the
    section length. This is the property Richardson extrapolation later relies on.
    """
    freq = Frequency(start=10.0, stop=10.0, npoints=1, unit='GHz')

    def taper(n):
        return ProfiledLine(
            RLGCLine,
            {
                'L': ExponentialProfile(start=250e-9, end=500e-9),
                'C': ExponentialProfile(start=100e-12, end=50e-12),
            },
            R=0.0, G=0.0, length=50e-3, n=n,
        )

    reference = taper(2048).a(freq)
    errors = [
        float(jnp.max(jnp.abs(taper(n).a(freq) - reference))) for n in (16, 32, 64)
    ]

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    # Second order is a ratio of 4; the window allows for the higher-order terms
    # that are still visible at these section counts.
    assert all(3.5 < ratio < 4.5 for ratio in ratios), (errors, ratios)
