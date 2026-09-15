"""Reading and updating parameters by name: `prf.params`, `prf.param_values`,
`prf.log_prior` and `prf.update` (ADR-0002, decisions 3 to 5 and 7)."""
import distreqx.distributions as dist
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.distributions import Uniform
from pmrf.models import Capacitor, Cascade, Resistor

from tests._jit import assert_same_jit_key


class RC(prf.Model):
    R: prf.Param = prf.param()
    C: prf.Param = prf.param(scale=1e-12)

    def s(self, freq):
        w = freq.w
        z = self.R + 1 / (1j * w * self.C)
        g = (z - 50.0) / (z + 50.0)
        return g[:, None, None]


def _rc():
    return RC(R=prf.Bounded(0.0, 100.0, value=50.0), C=prf.Random(Uniform(1.0, 3.0), value=2.0), name="rc")


# ---- Reading ------------------------------------------------------------------------


def test_params_and_values_are_top_level():
    from pmrf import parameters
    assert prf.params is parameters.params
    assert prf.param_values is parameters.param_values
    assert prf.log_prior is parameters.log_prior
    assert prf.update is parameters.update


def test_params_returns_param_objects():
    rc = RC(R=1.0, C=2.0)
    params = prf.params(rc)
    assert set(params) == {"R", "C"}
    assert all(prf.is_param(p) for p in params.values())


@pytest.mark.parametrize("space, expected", [("declared", 2.0), ("physical", 2e-12)])
def test_param_values_spaces(space, expected):
    rc = RC(R=1.0, C=2.0)
    assert np.allclose(prf.param_values(rc, space=space)["C"], expected)


def test_param_values_raw():
    m = _rc()
    raw = prf.param_values(m, space="raw")
    assert np.allclose(raw["R"], m.R.raw_value)
    assert np.allclose(m.R.raw_to_declared_bijector.forward(raw["R"]), 50.0)


def test_param_values_default_is_declared():
    assert np.allclose(prf.param_values(RC(R=1.0, C=2.0))["C"], 2.0)


def test_param_values_bad_space_raises():
    with pytest.raises(ValueError, match="space"):
        prf.param_values(RC(R=1.0, C=2.0), space="unconstrained")


def test_params_free_only():
    rc = RC(R=prf.Fixed(1.0), C=prf.Unconstrained(2.0))
    assert set(prf.params(rc, free_only=True)) == {"C"}
    assert set(prf.param_values(rc, free_only=True)) == {"C"}


def test_params_on_tuple_and_dict_of_models():
    r = Resistor(R=prf.Unconstrained(50.0), name="r")
    c = Capacitor(C=prf.Unconstrained(1.0), name="c")
    assert set(prf.params((r, c))) == {"r.R", "c.C"}
    assert set(prf.param_values({"a": r, "b": c})) == {"r.R", "c.C"}


def test_params_collision_raises():
    r = Resistor(R=prf.Unconstrained(50.0), name="r")
    with pytest.raises(ValueError, match="name collision"):
        prf.params((r, r))


def test_nested_named_modules_join_with_underscore():
    cas = Cascade([Cascade([Resistor(2.0, name="myR")], name="myCas")])
    assert any(k.startswith("myCas_myR") for k in prf.params(cas))


@pytest.mark.parametrize("where, expected", [
    ("R", {"R"}),
    ("C*", {"C"}),
    (["R", "C"], {"R", "C"}),
    (lambda m: m.R, {"R"}),
    (lambda m: (m.R, m.C), {"R", "C"}),
])
def test_params_where(where, expected):
    assert set(prf.params(_rc(), where)) == expected
    assert set(prf.param_values(_rc(), where)) == expected


def test_params_where_callable_selects_submodule():
    load = Resistor(R=prf.Unconstrained(50.0), name="load")
    cable = Resistor(R=prf.Unconstrained(1.0), name="cable")
    assert set(prf.params(load ** cable, lambda m: m.cascade[0])) == {"load.R"}


def test_params_where_unknown_name_raises():
    with pytest.raises(ValueError, match="nope"):
        prf.params(_rc(), "nope")


def test_removed_module_and_param_methods():
    for name in ("named_params", "values", "with_values", "with_free", "with_fixed"):
        assert not hasattr(prf.Module, name), name
    p = prf.Unconstrained(1.0)
    for name in ("as_fixed", "as_free", "wrap"):
        assert not hasattr(p, name), name


def test_replace_docstring_points_to_update():
    assert "update" in prf.replace.__doc__


# ---- Log prior ----------------------------------------------------------------------


def _scaled_bounded():
    # Uniform(1, 3) prior on a declared value 2.0, with scale 1e-3 and bounds (1, 3).
    return RC(R=prf.Fixed(1.0), C=prf.Random(Uniform(1.0, 3.0), value=2.0, scale=1e-3))


def test_log_prior_declared_by_hand():
    expected = np.log(1 / 2.0)
    assert np.allclose(prf.log_prior(_scaled_bounded()), expected)
    assert np.allclose(prf.log_prior(_scaled_bounded(), space="declared"), expected)


def test_log_prior_physical_by_hand():
    expected = np.log(1 / 2.0) - np.log(1e-3)
    assert np.allclose(prf.log_prior(_scaled_bounded(), space="physical"), expected)


def test_log_prior_raw_by_hand():
    m = _scaled_bounded()
    # Raw → declared for Uniform(1, 3) is x = 1 + 2 Φ(z); d x / d z = 2 φ(z).
    z = float(m.C.raw_value)
    log_det = np.log(2.0 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi))
    expected = np.log(1 / 2.0) + log_det
    assert np.allclose(prf.log_prior(m, space="raw"), expected, atol=1e-6)


def test_log_prior_array_param_scales_per_element():
    m = RC(R=prf.Fixed(1.0), C=prf.Random(dist.Normal(jnp.zeros(3), jnp.ones(3)), value=jnp.zeros(3), scale=10.0))
    declared = 3 * -0.5 * np.log(2 * np.pi)
    assert np.allclose(prf.log_prior(m), declared)
    assert np.allclose(prf.log_prior(m, space="physical"), declared - 3 * np.log(10.0))


def test_log_prior_on_tuple():
    a = Resistor(R=prf.Random(Uniform(0.0, 2.0), value=1.0), name="a")
    b = Resistor(R=prf.Random(Uniform(0.0, 4.0), value=1.0), name="b")
    assert np.allclose(prf.log_prior((a, b)), np.log(0.5) + np.log(0.25))


def test_log_prior_raw_agrees_with_prior_penalized():
    """PriorPenalized scores the physical density; with no scale that is the declared
    one, so the raw log prior differs from it by the Jacobian term alone."""
    from pmrf.problems import PriorPenalized, SummedTerms

    m = RC(R=prf.Random(Uniform(0.0, 100.0), value=40.0), C=prf.Fixed(2.0))
    problem = SummedTerms(model=m, terms=(lambda model: jnp.asarray(0.0),))
    penalty = PriorPenalized(problem)()
    log_det = m.R.raw_to_declared_bijector.forward_log_det_jacobian(m.R.raw_value)
    assert np.allclose(prf.log_prior(m, space="raw") - log_det, -penalty, atol=1e-6)


# ---- Update: value forms ------------------------------------------------------------


def test_update_mapping():
    m = prf.update(_rc(), {"R": 51.0, "C": 2.5})
    assert np.allclose(m.R.value, 51.0)
    assert np.allclose(m.C.value, 2.5) and np.allclose(m.C.physical_value, 2.5e-12)


def test_update_mapping_with_param_values():
    other = prf.update(_rc(), {"C": 2.5})
    m = prf.update(_rc(), prf.params(other))
    assert np.allclose(m.C.value, 2.5)


def test_update_where_value():
    m = prf.update(_rc(), "*", value=3.0)
    assert np.allclose(m.R.value, 3.0) and np.allclose(m.C.value, 3.0)


@pytest.mark.parametrize("space", ["declared", "physical", "raw"])
def test_update_mapping_space(space):
    src = prf.update(_rc(), {"C": 2.5})
    m = prf.update(_rc(), {"C": prf.param_values(src, space=space)["C"]}, space=space)
    assert np.allclose(m.C.value, 2.5, atol=1e-5)


@pytest.mark.parametrize("space, v", [("declared", 2.5), ("physical", 2.5e-12)])
def test_update_where_value_space(space, v):
    m = prf.update(_rc(), "C", value=v, space=space)
    assert np.allclose(m.C.value, 2.5)


def test_update_where_value_raw_space():
    m = _rc()
    raw = prf.param_values(prf.update(m, {"C": 2.5}), space="raw")["C"]
    assert np.allclose(prf.update(m, "C", value=raw, space="raw").C.value, 2.5, atol=1e-5)


def test_update_root_param():
    p = prf.Random(Uniform(45.0, 55.0), value=50.0, name="R", scale=2.0, metadata={"a": 1})
    q = prf.update(p, value=51.0)
    assert np.allclose(q.value, 51.0) and np.allclose(q.physical_value, 102.0)
    assert q.name == "R" and q.scale == 2.0 and q.metadata == {"a": 1}
    assert q.distribution == p.distribution


def test_update_keeps_everything_else():
    system = _rc()
    m = prf.update(system, {"C": 2.5})
    for name, p in prf.params(system).items():
        q = prf.params(m)[name]
        assert q.fixed == p.fixed and q.scale == p.scale and q.name == p.name
        assert q.metadata == p.metadata and q.distribution == p.distribution
        assert np.allclose(q.bounds, p.bounds) if p.bounds is not None else q.bounds is None


def test_update_on_tuple_of_models():
    r = Resistor(R=prf.Unconstrained(50.0), name="r")
    c = Capacitor(C=prf.Unconstrained(1.0), name="c")
    r2, c2 = prf.update((r, c), {"r.R": 60.0, "c.C": 2.0})
    assert np.allclose(r2.R.value, 60.0) and np.allclose(c2.C.value, 2.0)


def test_update_on_frozen_tree():
    frozen = jax.tree.map(prf.freeze, _rc(), is_leaf=prf.is_param)
    m = prf.update(frozen, {"R": 52.0})
    assert np.allclose(prf.param_values(m)["R"], 52.0)
    assert prf.params(m, free_only=True) == {}


def test_update_fixed_param_value():
    p = prf.Random(Uniform(45.0, 55.0), value=50.0, fixed=True)
    q = prf.update(p, value=51.0)
    assert q.fixed and np.allclose(q.value, 51.0)
    assert prf.update(q, fixed=False).distribution is not None


# ---- Update: fixed form -------------------------------------------------------------


def test_update_fixed_true():
    m = prf.update(_rc(), "C", fixed=True)
    assert m.C.fixed and not m.R.fixed
    assert np.allclose(m.C.value, 2.0)
    assert prf.update(m, "C", fixed=False).C.distribution is not None
    assert set(prf.params(m)) == {"R", "C"}
    assert set(prf.params(m, free_only=True)) == {"R"}


def test_update_fixed_false_frees_param_created_fixed():
    m = RC(R=prf.Fixed(1.0), C=prf.Fixed(2.0))
    freed = prf.update(m, "R", fixed=False)
    assert not freed.R.fixed and freed.C.fixed


def test_update_fixed_is_additive():
    m = prf.update(_rc(), "R", fixed=True)
    m = prf.update(m, "C", fixed=True)
    assert m.R.fixed and m.C.fixed
    only_c = prf.update(prf.update(m, "*", fixed=True), ["C"], fixed=False)
    assert set(prf.params(only_c, free_only=True)) == {"C"}


def test_update_fixed_twice_is_idempotent():
    once = prf.update(_rc(), "C", fixed=True)
    assert bool(eqx.tree_equal(prf.update(once, "C", fixed=True), once))


def test_update_fixed_root_param():
    assert prf.update(prf.Unconstrained(1.0), fixed=True).fixed


# ---- Update: errors -----------------------------------------------------------------


def test_update_out_of_bounds_raises():
    with pytest.raises(Exception, match="outside the constraint"):
        prf.update(_rc(), {"R": 200.0})
    with pytest.raises(Exception, match="outside the constraint"):
        prf.update(_rc(), "R", value=-1.0)


def test_update_out_of_bounds_raises_under_jit():
    @eqx.filter_jit
    def set_value(model, v):
        return prf.param_values(prf.update(model, {"R": v}))["R"]

    assert np.allclose(set_value(_rc(), 51.0), 51.0)
    with pytest.raises(Exception, match="outside the constraint"):
        set_value(_rc(), 200.0)


def test_update_unknown_name_raises():
    with pytest.raises(ValueError, match="nope"):
        prf.update(_rc(), {"nope": 1.0})
    with pytest.raises(ValueError, match="nope"):
        prf.update(_rc(), "nope", value=1.0)


@pytest.mark.parametrize("call", [
    lambda m: prf.update(m, {1: 2.0}),
    lambda m: prf.update(m, "R"),
    lambda m: prf.update(m, 3.0),
    lambda m: prf.update(m, {"R": 1.0}, value=2.0),
    lambda m: prf.update(m, "R", value=1.0, fixed=True),
    lambda m: prf.update(m, value=1.0),
])
def test_update_form_mismatch_lists_forms(call):
    with pytest.raises(TypeError, match=r"(?s)update\(model, \{.*where, value=.*fixed="):
        call(_rc())


# ---- Update keeps the jit cache key -------------------------------------------------


@pytest.mark.parametrize("space", ["declared", "physical", "raw"])
def test_update_param_values_round_trip_keeps_jit_key(space):
    m = _rc()
    same = prf.update(m, prf.param_values(m, space=space), space=space)
    assert_same_jit_key(m, same)
    for name, v in prf.param_values(m).items():
        assert np.allclose(prf.param_values(same)[name], v, rtol=1e-5)


@pytest.mark.parametrize("v", [3.0, np.float64(3.0), jnp.asarray(3.0), jnp.asarray(3, dtype=jnp.int32)])
def test_update_value_form_keeps_jit_key(v):
    m = _rc()
    assert_same_jit_key(m, prf.update(m, "*", value=v))
    assert_same_jit_key(m.C, prf.update(m.C, value=v))
