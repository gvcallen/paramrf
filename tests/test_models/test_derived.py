"""Tests for derived models (`prf.derived`, #169)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.distributions import Uniform
from pmrf.models import Cascade, CoaxialLine, FloatingTwoPort, Resistor
from tests._jit import assert_same_jit_key


FREQ = prf.Frequency(10, 500, 21, 'MHz')


@prf.derived
def wet(cable, wet_length, wet_ep_r):
    wet_part = prf.replace(
        cable, length=wet_length, dielectric=prf.replace(cable.dielectric, ep_r=wet_ep_r),
    )
    dry_part = prf.replace(cable, length=cable.length - wet_length)
    return wet_part ** dry_part


@prf.derived
def wet_level(cable, wet_length):
    wet_part = prf.replace(cable, length=wet_length)
    dry_part = prf.replace(cable, length=cable.length - wet_length)
    return wet_part ** dry_part


@prf.derived
def wet_pair(system, wet_length):
    return prf.update(system, {
        'east': wet_level(system.cascade[0], wet_length=wet_length),
        'west': wet_level(system.cascade[1], wet_length=wet_length),
    })


def _cable(**kwargs):
    return CoaxialLine(
        length=prf.Random(Uniform(1.0, 3.0), value=2.0),
        d_in=prf.Random(Uniform(1.0, 1.3), value=1.12, scale=1e-3),
        **kwargs,
    )


def _wet(cable=None, w=0.4, ep_r=4.0):
    return wet(
        cable if cable is not None else _cable(),
        wet_length=prf.Random(Uniform(0.0, 1.0), value=w),
        wet_ep_r=prf.Random(Uniform(1.0, 80.0), value=ep_r),
    )


def _by_hand(cable, w, ep_r):
    wet_part = prf.update(cable, 'length', prf.Fixed(w))
    wet_part = prf.update(wet_part, 'dielectric.ep_r', prf.Fixed(ep_r))
    dry_part = prf.update(cable, 'length', prf.Fixed(cable.length.value - w))
    return wet_part ** dry_part


def test_exact_against_hand_built_cascade():
    cable = _cable()
    np.testing.assert_allclose(_wet(cable).s(FREQ), _by_hand(cable, 0.4, 4.0).s(FREQ), rtol=1e-12, atol=1e-14)


def test_zero_wet_length_is_the_plain_cable():
    cable = _cable()
    np.testing.assert_allclose(_wet(cable, w=0.0).s(FREQ), cable.s(FREQ), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('name', ['length', 'wet_length'])
def test_gradients_match_finite_differences(name):
    model = _wet()
    raw = prf.param_values(model, space='raw')

    def loss(values):
        return jnp.sum(jnp.abs(prf.update(model, values, space='raw').s(FREQ)[:, 0, 0]) ** 2)

    grad = jax.grad(loss)(raw)[name]
    h = 1e-6
    plus = dict(raw, **{name: raw[name] + h})
    minus = dict(raw, **{name: raw[name] - h})
    fd = (loss(plus) - loss(minus)) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-5)


def test_names_are_the_base_names_plus_keywords():
    cable = _cable(name='coax')
    model = _wet(cable)
    assert set(prf.params(model)) == set(prf.params(cable)) | {'wet_length', 'wet_ep_r'}
    assert len(prf.params(model)) == len(prf.params(cable)) + 2
    assert model.name == 'coax'


def test_names_are_prefixed_under_a_named_container():
    def names(part):
        west = Cascade([part, Resistor(50.0, name='r')], name='west')
        return set(prf.params(Cascade([west, Resistor(50.0, name='load')])))

    cable = _cable(name='coaxial')
    assert names(_wet(cable)) == names(cable) | {'west_coaxial.wet_length', 'west_coaxial.wet_ep_r'}
    assert 'west_coaxial.length' in names(_wet(cable))


def test_clashing_keyword_raises():
    with pytest.raises(ValueError, match="clash"):
        wet_level(_cable(), length=1.0)


@pytest.mark.parametrize('args', [(), (1, 2)])
def test_wrong_call_shape_raises(args):
    with pytest.raises(TypeError, match="exactly one positional base"):
        wet_level(*args, wet_length=1.0)


def test_non_model_return_raises():
    @prf.derived
    def bad(cable, x):
        return x

    with pytest.raises(TypeError, match="must return a pmrf.Model"):
        bad(_cable(), x=prf.Unconstrained(1.0)).s(FREQ)


def test_every_parameter_exists_once():
    cable = _cable()
    model = _wet(cable)
    assert len(prf.params(model, free_only=True)) == len(prf.params(cable, free_only=True)) + 2


@pytest.mark.parametrize('space', ['raw', 'declared', 'physical'])
def test_round_trip_keeps_the_jit_key(space):
    model = _wet()
    values = prf.param_values(model, space=space)
    assert_same_jit_key(prf.update(model, values, space=space), model)
    changed = prf.update(model, {'wet_length': 0.3})
    assert_same_jit_key(changed, model)


def test_previous_fit_applies_to_the_base():
    cable = _cable()
    fitted = prf.update(cable, {'length': 2.5, 'd_in': 1.2})
    model = prf.update(_wet(cable), prf.param_values(fitted))
    assert prf.param_values(model)['length'] == 2.5
    assert prf.param_values(model)['d_in'] == 1.2


def test_fixed_behaves_as_on_any_model():
    model = prf.update(_wet(), 'wet_*', fixed=True)
    free = prf.params(model, free_only=True)
    assert 'wet_length' not in free and 'wet_ep_r' not in free
    model = prf.update(model, 'dielectric.ep_r', fixed=False)
    assert 'dielectric.ep_r' in prf.params(model, free_only=True)


@pytest.mark.parametrize('space', ['raw', 'declared', 'physical'])
def test_log_prior_is_the_base_plus_the_new_parameters(space):
    cable = _cable()
    w = prf.Random(Uniform(0.0, 1.0), value=0.4)
    ep_r = prf.Random(Uniform(1.0, 80.0), value=4.0)
    model = wet(cable, wet_length=w, wet_ep_r=ep_r)
    expected = prf.log_prior(cable, space=space) + prf.log_prior((w, ep_r), space=space)
    np.testing.assert_allclose(prf.log_prior(model, space=space), expected)


def test_nesting_shares_one_parameter():
    east = CoaxialLine(length=2.0, name='east')
    west = CoaxialLine(length=3.0, name='west')
    system = wet_pair(east ** west, wet_length=prf.Unconstrained(0.5))
    assert list(prf.params(system)).count('wet_length') == 1
    assert 'east.length' in prf.params(system) and 'west.length' in prf.params(system)

    changed = prf.update(system, {'wet_length': 1.0})
    by_hand = (
        prf.update(east, {'length': 1.0}) ** prf.update(east, {'length': 1.0})
        ** prf.update(west, {'length': 1.0}) ** prf.update(west, {'length': 2.0})
    )
    np.testing.assert_allclose(changed.s(FREQ), by_hand.s(FREQ), rtol=1e-10, atol=1e-12)


def test_composition():
    model = _wet()
    cascaded = model ** Resistor(10.0)
    assert cascaded.s(FREQ).shape == (FREQ.npoints, 2, 2)
    assert FloatingTwoPort(floating=model).s(FREQ).shape == (FREQ.npoints, 4, 4)
    tied = prf.tie(model, 'wet_ep_r', 'dielectric.ep_r', fn=lambda x: 2 * x)
    assert 'wet_ep_r' not in prf.params(tied)
    np.testing.assert_allclose(tied.s(FREQ), _by_hand(_cable(), 0.4, 2 * 1.0).s(FREQ), rtol=1e-10, atol=1e-12)


def test_same_function_shares_the_jit_key():
    assert_same_jit_key(_wet(w=0.1), _wet(w=0.2))


def test_fit_smoke():
    from pmrf.fitting import fit_minimize

    truth = _wet(w=0.5)
    start = prf.update(_wet(w=0.3), '*', fixed=True)
    start = prf.update(start, 'wet_length', fixed=False)
    result = fit_minimize(start, np.asarray(truth.s(FREQ)), frequency=FREQ)
    assert isinstance(result.model, type(start))
    np.testing.assert_allclose(prf.param_values(result.model)['wet_length'], 0.5, atol=1e-3)


def test_tie_across_derived_and_plain_models():
    system = Cascade([_wet(_cable(name='coax')), Resistor(prf.Unconstrained(10.0), name='r')])
    tied = prf.tie(system, 'r.R', 'coax.wet_ep_r', fn=lambda x: 5 * x)
    assert 'r.R' not in prf.params(tied)
    expected = _by_hand(_cable(), 0.4, 4.0) ** Resistor(20.0)
    np.testing.assert_allclose(tied.s(FREQ), expected.s(FREQ), rtol=1e-10, atol=1e-12)


def test_structural_update_on_a_base_sub_model_and_free_values():
    from pmrf.materials import ConstantDielectric

    model = prf.update(_wet(), 'dielectric', ConstantDielectric(ep_r=prf.Fixed(1.0)))
    assert 'dielectric.ep_r' not in prf.param_values(model, free_only=True)
    assert {'wet_length', 'wet_ep_r'} <= set(prf.param_values(model, space='raw', free_only=True))
