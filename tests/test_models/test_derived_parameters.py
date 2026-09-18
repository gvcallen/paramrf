"""Tests for derived parameters (`prf.derived` on a value, #172)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.constraints import Positive
from pmrf.distributions import Normal, Uniform
from pmrf.materials import ConstantDielectric
from pmrf.models import Cascade, CoaxialLine, Resistor
from tests._jit import assert_same_jit_key


FREQ = prf.Frequency(10, 500, 21, 'MHz')
DELTA_T = 12.0
EP_R = 2.1
TC = 5e-4
VF = 0.83


@prf.derived
def drift(ep_r, tc):
    """Permittivity drifting from its nominal value with temperature."""
    return ep_r * (1 + tc * DELTA_T)


@prf.derived
def from_vf(vf):
    """Permittivity from the velocity factor, with no base."""
    return 1 / vf**2


@prf.derived
def stretched(cable, factor):
    """A derived model, for the composition tests."""
    return prf.replace(cable, length=cable.length * factor)


def _line(ep_r=EP_R, **kwargs):
    return CoaxialLine(
        length=prf.Random(Uniform(1.0, 3.0), value=2.0),
        d_in=prf.Random(Uniform(1.0, 1.3), value=1.12, scale=1e-3),
        dielectric=ConstantDielectric(ep_r=prf.Random(Uniform(1.0, 5.0), value=ep_r)),
        **kwargs,
    )


def _drifted(tc=TC, **kwargs):
    line = _line(**kwargs)
    return prf.update(
        line, 'dielectric.ep_r',
        drift(line.dielectric.ep_r, tc=prf.Random(Normal(0.0, 1e-3), value=tc)),
    )


def _from_vf(vf=VF):
    return CoaxialLine(
        length=prf.Random(Uniform(1.0, 3.0), value=2.0),
        d_in=prf.Random(Uniform(1.0, 1.3), value=1.12, scale=1e-3),
        dielectric=ConstantDielectric(
            ep_r=from_vf(vf=prf.Random(Normal(0.83, 0.025), value=vf)),
        ),
    )


def _by_hand(ep_r):
    return prf.update(_line(), 'dielectric.ep_r', prf.Fixed(ep_r))


# Constructor and error cases


def test_a_param_base_gives_a_value_not_a_model():
    node = drift(prf.Unconstrained(EP_R), tc=TC)
    assert isinstance(node, prf.DerivedValue) and prf.is_derived(node)
    assert not isinstance(node, prf.Model)


def test_a_model_base_still_gives_a_model():
    node = stretched(_line(), factor=prf.Unconstrained(1.5))
    assert isinstance(node, prf.Derived) and isinstance(node, prf.Model)


@pytest.mark.parametrize('base', [np.asarray(EP_R), (prf.Unconstrained(1.0), 2.0)])
def test_an_array_or_pytree_base_gives_a_value(base):
    assert isinstance(drift(base, tc=TC), prf.DerivedValue)


def test_a_collection_of_models_is_not_a_model():
    # Narrowed by #172: derive at the model that contains them instead.
    node = stretched((_line(), _line()), factor=1.5)
    assert isinstance(node, prf.DerivedValue) and not isinstance(node, prf.Model)


def test_no_base_gives_a_value():
    assert isinstance(from_vf(vf=VF), prf.DerivedValue)


def test_two_positional_bases_raise():
    with pytest.raises(TypeError, match="at most one positional base"):
        drift(1.0, 2.0, tc=TC)


def test_no_base_and_no_keywords_raises():
    with pytest.raises(TypeError, match="at least one new parameter"):
        from_vf()


def test_name_on_a_value_raises():
    with pytest.raises(TypeError, match="'name='"):
        from_vf(vf=VF, name='eps')


def test_a_model_base_returning_a_non_model_still_raises():
    @prf.derived
    def bad(cable, x):
        return x

    with pytest.raises(TypeError, match="must return a pmrf.Model"):
        bad(_line(), x=prf.Unconstrained(1.0)).s(FREQ)


# Fields


def test_a_field_stores_the_node_untouched():
    node = from_vf(vf=prf.Unconstrained(VF))
    line = CoaxialLine(length=1.0, dielectric=ConstantDielectric(ep_r=node))
    assert line.dielectric.ep_r is node

    # `update` rebuilds the tree, so the node is equal rather than identical.
    updated = prf.update(_line(), 'dielectric.ep_r', node)
    stored = updated.dielectric.ep_r
    assert isinstance(stored, prf.DerivedValue) and stored.fn is node.fn
    np.testing.assert_allclose(prf.unwrap(stored), prf.unwrap(node))


def test_the_field_constraint_and_scale_do_not_apply():
    class Scaled(prf.Module):
        x: prf.Param = prf.param(constraint=Positive(), scale=1e-3)

    # A negative derived value passes a positive-constrained field unchecked, and
    # is not rescaled: it is `fn`'s physical value.
    node = from_vf(vf=prf.Unconstrained(-0.5))
    scaled = Scaled(node)
    assert scaled.x is node
    np.testing.assert_allclose(prf.unwrap(Scaled(drift(-1.0, tc=0.0))).x, -1.0)


def test_a_derived_value_is_not_a_parameter():
    line = _drifted()
    assert 'dielectric.ep_r' in prf.params(line)  # the base, not the derived value
    assert not prf.is_param(line.dielectric.ep_r)
    values = prf.param_values(line)
    assert set(values) == set(prf.params(line))


# Exactness


def test_drift_matches_the_hand_built_line():
    np.testing.assert_allclose(
        _drifted().s(FREQ), _by_hand(EP_R * (1 + TC * DELTA_T)).s(FREQ),
        rtol=1e-12, atol=1e-14,
    )


def test_zero_drift_is_the_plain_line():
    np.testing.assert_allclose(
        _drifted(tc=0.0).s(FREQ), _line().s(FREQ), rtol=1e-12, atol=1e-14,
    )


def test_from_vf_matches_the_hand_built_line():
    np.testing.assert_allclose(
        _from_vf().s(FREQ), _by_hand(1 / VF**2).s(FREQ), rtol=1e-12, atol=1e-14,
    )


# Names


def test_names_with_a_base_are_the_line_plus_the_keyword():
    line, drifted = _line(), _drifted()
    assert set(prf.params(drifted)) == set(prf.params(line)) | {'dielectric.tc'}
    assert prf.params(drifted)['dielectric.ep_r'].distribution is not None


def test_names_with_no_base_replace_the_field():
    names = set(prf.params(_from_vf()))
    assert 'dielectric.ep_r' not in names
    assert 'dielectric.vf' in names


def test_names_are_prefixed_under_a_named_container():
    system = Cascade([_drifted(name='coax'), Resistor(50.0, name='load')])
    names = set(prf.params(system))
    assert {'coax.dielectric.tc', 'coax.dielectric.ep_r'} <= names

    west = Cascade([_from_vf(), Resistor(50.0, name='r')], name='west')
    system = Cascade([west, Resistor(50.0, name='load')])
    assert 'west.cascade[0].dielectric.vf' in prf.params(system)


def test_a_keyword_clashing_with_a_model_base_raises():
    with pytest.raises(ValueError, match="clash"):
        stretched(_line(), length=1.0)


def test_a_keyword_clashing_with_a_sibling_parameter_raises():
    line = _line()
    # 'tand' is a sibling field of 'ep_r' on the dielectric.
    line = prf.update(line, 'dielectric.ep_r', drift(line.dielectric.ep_r, tand=TC))
    with pytest.raises(ValueError, match="collision"):
        prf.params(line)


def test_two_derived_fields_sharing_a_keyword_raise():
    line = _line()
    line = prf.update(line, 'dielectric.ep_r', drift(line.dielectric.ep_r, k=TC))
    line = prf.update(line, 'dielectric.mu_r', drift(line.dielectric.mu_r, k=TC))
    with pytest.raises(ValueError, match="collision"):
        prf.params(line)


# Priors and dead parameters


def test_the_replaced_parameter_is_gone():
    line = _from_vf()
    assert 'dielectric.ep_r' not in prf.params(line)
    without = prf.update(_line(), 'dielectric.ep_r', prf.Fixed(1 / VF**2))
    np.testing.assert_allclose(
        prf.log_prior(line),
        prf.log_prior(without) + prf.log_prior(prf.Random(Normal(0.83, 0.025), value=VF)),
    )


@pytest.mark.parametrize('space', ['raw', 'declared', 'physical'])
def test_log_prior_is_the_line_plus_the_new_parameter(space):
    expected = prf.log_prior(_line(), space=space) + prf.log_prior(
        prf.Random(Normal(0.0, 1e-3), value=TC), space=space,
    )
    np.testing.assert_allclose(prf.log_prior(_drifted(), space=space), expected, rtol=1e-12)


def test_nested_derived_values_are_scored_once():
    inner = drift(prf.Random(Uniform(1.0, 5.0), value=EP_R), tc=prf.Unconstrained(TC))
    line = prf.update(_line(), 'dielectric.ep_r', from_vf(vf=inner))
    np.testing.assert_allclose(prf.log_prior(line), prf.log_prior(_line()))


# Gradients, jit and fixed


@pytest.mark.parametrize('name', ['dielectric.tc', 'dielectric.ep_r', 'length'])
def test_gradients_match_finite_differences(name):
    model = _drifted()
    raw = prf.param_values(model, space='raw')

    def loss(values):
        return jnp.sum(jnp.abs(prf.update(model, values, space='raw').s(FREQ)[:, 0, 0]) ** 2)

    grad = jax.grad(loss)(raw)[name]
    h = 1e-6
    plus = dict(raw, **{name: raw[name] + h})
    minus = dict(raw, **{name: raw[name] - h})
    np.testing.assert_allclose(grad, (loss(plus) - loss(minus)) / (2 * h), rtol=1e-4)


def test_gradients_flow_to_a_parameter_with_no_base():
    model = _from_vf()
    raw = prf.param_values(model, space='raw')

    def loss(values):
        return jnp.sum(jnp.abs(prf.update(model, values, space='raw').s(FREQ)[:, 0, 0]) ** 2)

    grad = jax.grad(loss)(raw)['dielectric.vf']
    h = 1e-6
    plus = dict(raw, **{'dielectric.vf': raw['dielectric.vf'] + h})
    minus = dict(raw, **{'dielectric.vf': raw['dielectric.vf'] - h})
    np.testing.assert_allclose(grad, (loss(plus) - loss(minus)) / (2 * h), rtol=1e-4)


@pytest.mark.parametrize('space', ['raw', 'declared', 'physical'])
def test_round_trip_keeps_the_jit_key(space):
    model = _drifted()
    values = prf.param_values(model, space=space)
    assert_same_jit_key(prf.update(model, values, space=space), model)


def test_changing_a_new_parameter_does_not_recompile():
    assert_same_jit_key(_drifted(tc=1e-3), _drifted(tc=TC))
    assert_same_jit_key(prf.update(_drifted(), {'dielectric.tc': 1e-3}), _drifted())


def test_fixed_behaves_as_on_any_parameter():
    model = prf.update(_drifted(), 'dielectric.tc', fixed=True)
    assert 'dielectric.tc' not in prf.params(model, free_only=True)
    assert 'dielectric.ep_r' in prf.params(model, free_only=True)
    model = prf.update(model, 'dielectric.tc', fixed=False)
    assert 'dielectric.tc' in prf.params(model, free_only=True)


def test_an_out_of_constraint_derived_value_is_computed():
    # CoaxialLine's length is positive-constrained; a derived negative one is not checked.
    line = prf.update(_line(), 'length', from_vf(vf=prf.Unconstrained(-1.0)))
    np.testing.assert_allclose(prf.unwrap(line).length, 1.0)


# Composition


def test_a_derived_field_inside_a_derived_model():
    model = stretched(_drifted(), factor=prf.Unconstrained(1.4))
    assert {'dielectric.tc', 'dielectric.ep_r', 'factor'} <= set(prf.params(model))
    expected = prf.update(_by_hand(EP_R * (1 + TC * DELTA_T)), {'length': 2.8})
    np.testing.assert_allclose(model.s(FREQ), expected.s(FREQ), rtol=1e-10, atol=1e-12)


def test_a_derived_model_producing_a_derived_field():
    @prf.derived
    def drifting(cable, tc):
        return prf.update(cable, 'dielectric.ep_r', drift(cable.dielectric.ep_r, tc=tc))

    model = drifting(_line(), tc=prf.Unconstrained(TC))
    assert 'tc' in prf.params(model) and 'dielectric.tc' not in prf.params(model)
    np.testing.assert_allclose(
        model.s(FREQ), _by_hand(EP_R * (1 + TC * DELTA_T)).s(FREQ), rtol=1e-10, atol=1e-12,
    )


def test_tie_works_on_a_derived_fields_parameter():
    system = Cascade([_drifted(name='coax'), Resistor(prf.Unconstrained(10.0), name='r')])
    tied = prf.tie(system, 'r.R', 'coax.dielectric.tc', fn=lambda x: 1e4 * x)
    assert 'r.R' not in prf.params(tied)
    expected = _by_hand(EP_R * (1 + TC * DELTA_T)) ** Resistor(1e4 * TC)
    np.testing.assert_allclose(tied.s(FREQ), expected.s(FREQ), rtol=1e-10, atol=1e-12)


def test_fit_smoke():
    from pmrf.fitting import fit_minimize

    truth = _drifted(tc=2e-3)
    start = prf.update(_drifted(tc=0.0), '*', fixed=True)
    start = prf.update(start, 'dielectric.tc', fixed=False)
    result = fit_minimize(start, np.asarray(truth.s(FREQ)), frequency=FREQ)
    np.testing.assert_allclose(
        prf.param_values(result.model)['dielectric.tc'], 2e-3, atol=1e-5,
    )
