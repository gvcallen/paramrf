import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.constraints import Positive
from pmrf.distributions import Uniform
from pmrf.models import Resistor

from tests._jit import assert_same_jit_key


class RC(prf.Model):
    R: prf.Param = prf.param()
    C: prf.Param = prf.param(scale=1e-12)

    def s(self, freq):
        _TRACES.append(None)
        w = freq.w
        z = self.R + 1 / (1j * w * self.C)
        g = (z - 50.0) / (z + 50.0)
        return g[:, None, None]


_TRACES = []


class BoundedPF(prf.Model):
    C: prf.Param = prf.param(constraint=Positive(), scale=1e-12)

    def s(self, freq):
        return jnp.zeros((freq.npoints, 1, 1), dtype=complex)


# Value spaces

def test_param_value_spaces():
    p = prf.Constrained(Positive(), 2.0, scale=1e-12)
    assert np.allclose(p.value, 2.0)
    assert np.allclose(p.physical_value, 2e-12)
    assert np.allclose(p.raw_to_declared_bijector.forward(p.raw_value), 2.0)
    assert np.allclose(p.declared_to_physical_bijector.forward(p.value), 2e-12)
    assert not hasattr(p, "unscaled_value")


def test_param_variable_field():
    p = prf.Unconstrained(2.0)
    assert hasattr(p, "variable") and not isinstance(p.variable, jax.Array)
    q = prf.Param(variable=p.variable, scale=1e-3)
    assert np.allclose(q.value, 2.0)


def test_unwrap_is_physical():
    rc = RC(R=1.0, C=2.0)
    unwrapped = prf.unwrap(rc)
    assert np.allclose(unwrapped.C, 2e-12)
    assert np.allclose(rc.C * 1.0, 2e-12)


# Scale is units

def test_field_scale_is_inherited():
    rc = RC(R=1.0, C=prf.Unconstrained(2.0))
    assert np.allclose(rc.C.value, 2.0)
    assert np.allclose(rc.C.physical_value, 2e-12)


def test_explicit_scale_overrides_field():
    rc = RC(R=1.0, C=prf.Unconstrained(2.0, scale=1e-9))
    assert np.allclose(rc.C.value, 2.0)
    assert np.allclose(rc.C.physical_value, 2e-9)


def test_bounds_are_declared_space():
    m = BoundedPF(C=prf.Bounded(1.0, 3.0, value=2.0))
    assert np.allclose(m.C.physical_value, 2e-12)
    assert np.allclose(m.C.bounds, (1.0, 3.0))
    assert np.allclose(prf.replace(m.C, value=2.5).physical_value, 2.5e-12)
    with pytest.raises(Exception, match="outside the constraint"):
        prf.replace(m.C, value=4.0)


def test_distribution_is_declared_space():
    from pmrf.parameters import node_distribution

    m = BoundedPF(C=prf.Random(Uniform(1.0, 3.0), value=2.0))
    assert np.allclose(prf.unwrap(m.C.distribution).mean(), 2.0)
    prior = node_distribution(m.C)
    assert np.isfinite(prior.log_prob(m.C.physical_value))
    assert np.isneginf(prior.log_prob(m.C.value))


# Existing value surfaces follow declared space

def test_param_values_and_update_are_declared():
    rc = RC(R=1.0, C=2.0)
    assert np.allclose(prf.param_values(rc)["C"], 2.0)
    updated = prf.update(rc, {"C": 3.0})
    assert np.allclose(updated.C.value, 3.0)
    assert np.allclose(updated.C.physical_value, 3e-12)


def test_replace_value_keeps_everything_else():
    p = prf.Random(Uniform(45.0, 55.0), value=50.0, name="R", scale=2.0, metadata={"a": 1})
    q = prf.replace(p, value=51.0)

    assert np.allclose(q.value, 51.0)
    assert np.allclose(q.physical_value, 102.0)
    assert q.name == "R" and q.scale == 2.0 and q.metadata == {"a": 1}
    assert q.distribution == p.distribution
    assert q.bounds is not None and np.allclose(q.bounds, p.bounds)
    assert not q.fixed


def test_replace_value_on_tree():
    load = Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load")
    updated = prf.update(load, "R", fn=lambda p: prf.replace(p, value=51.0))
    assert np.allclose(prf.param_values(updated)["R"], 51.0)


def test_replace_value_keeps_fixed_prior():
    p = prf.Random(Uniform(45.0, 55.0), value=50.0, fixed=True)
    q = prf.replace(p, value=51.0)
    assert q.fixed
    assert np.allclose(q.value, 51.0)
    assert np.allclose(prf.update(q, fixed=False).value, 51.0)
    assert prf.update(q, fixed=False).distribution is not None


def test_replace_value_out_of_bounds_raises():
    p = prf.Bounded(0.0, 1.0, value=0.5)
    with pytest.raises(Exception, match="outside the constraint"):
        prf.replace(p, value=2.0)


def test_replace_value_is_identity():
    p = prf.Random(Uniform(45.0, 55.0), value=50.0, scale=1e-3)
    q = prf.replace(p, value=p.value)
    assert np.allclose(q.value, 50.0)
    assert np.allclose(q.physical_value, 50e-3)


def test_update_out_of_bounds_raises_under_jit():
    import equinox as eqx

    load = Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load")

    @eqx.filter_jit
    def set_value(model, v):
        return prf.param_values(prf.update(model, {"R": v}))["R"]

    assert np.allclose(set_value(load, 51.0), 51.0)
    with pytest.raises(Exception, match="outside the constraint"):
        set_value(load, 60.0)


def test_param_constructor_out_of_bounds_raises_under_jit():
    import equinox as eqx

    @eqx.filter_jit
    def build(v):
        return prf.Bounded(0.0, 1.0, value=v).value

    with pytest.raises(Exception, match="outside the constraint"):
        build(2.0)


# Value changes keep the jit cache key

_PARAMS = [
    prf.Unconstrained(2.0),
    prf.Unconstrained(jnp.asarray([1.0, 2.0])),
    prf.Constrained(Positive(), 2.0, scale=1e-12),
    prf.Bounded(0.0, 5.0, value=2.0),
    prf.Random(Uniform(0.0, 5.0), value=2.0, fixed=True),
    prf.Fixed(2.0),
]
_NEW_VALUES = [3.0, np.float64(3.0), jnp.asarray(3.0), jnp.asarray(3, dtype=jnp.int32)]


@pytest.mark.parametrize("p", _PARAMS)
@pytest.mark.parametrize("v", _NEW_VALUES)
def test_replace_keeps_jit_key(p, v):
    v = jnp.broadcast_to(v, jnp.shape(p.value))
    assert_same_jit_key(p, prf.replace(p, value=v))


@pytest.mark.parametrize("v", _NEW_VALUES)
def test_update_keeps_jit_key(v):
    rc = RC(R=1.0, C=prf.Bounded(0.0, 5.0, value=2.0))
    assert_same_jit_key(rc, prf.update(rc, {"R": v, "C": v}))


def test_value_change_does_not_recompile():
    freq = prf.Frequency(1, 2, 3, "GHz")
    rc = RC(R=1.0, C=2.0)
    _TRACES.clear()
    prf.update(rc, {"C": 3.0}).s(freq)
    assert len(_TRACES) == 1
    prf.update(rc, {"C": jnp.asarray(4.0)}).s(freq)
    prf.update(rc, {"R": np.float64(5.0)}).s(freq)
    assert len(_TRACES) == 1
