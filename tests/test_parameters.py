import numpy as np
import pytest

import pmrf as prf
from pmrf.distributions import Uniform
from pmrf.models import Resistor


def test_replace_value_keeps_everything_else():
    """B3: `prf.replace(p, value=...)` changes only the value."""
    p = prf.Random(Uniform(45.0, 55.0), value=50.0, name="R", scale=2.0, metadata={"a": 1})
    q = prf.replace(p, value=102.0)

    assert np.allclose(q.value, 102.0)
    assert np.allclose(q.unscaled_value, 51.0)
    assert q.name == "R" and q.scale == 2.0 and q.metadata == {"a": 1}
    assert q.distribution == p.distribution
    assert q.bounds is not None and np.allclose(q.bounds, p.bounds)
    assert not q.fixed


def test_replace_value_on_tree():
    load = Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load")
    updated = load.at("R").apply(lambda p: prf.replace(p, value=51.0))
    assert updated.named_params() == {"R": 51.0}


def test_replace_value_keeps_fixed_prior():
    p = prf.Random(Uniform(45.0, 55.0), value=50.0, fixed=True)
    q = prf.replace(p, value=51.0)
    assert q.fixed
    assert np.allclose(q.unscaled_value, 51.0)
    assert np.allclose(q.as_free().unscaled_value, 51.0)
    assert q.as_free().distribution is not None


def test_replace_value_out_of_bounds_raises():
    p = prf.Bounded(0.0, 1.0, value=0.5)
    with pytest.raises(Exception, match="outside the constraint"):
        prf.replace(p, value=2.0)


def test_replace_value_is_physical_identity():
    """`replace(p, value=p.value)` is the identity, up to bijector round-trip."""
    p = prf.Random(Uniform(45.0, 55.0), value=50.0, scale=1e-3)
    q = prf.replace(p, value=p.value)
    assert np.allclose(q.value, p.value)
    assert np.allclose(q.unscaled_value, 50.0)


def test_with_values_out_of_bounds_raises_under_jit():
    import equinox as eqx
    import jax

    load = Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load")

    @eqx.filter_jit
    def set_value(model, v):
        return model.with_values({"R": v}).values()["R"]

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
