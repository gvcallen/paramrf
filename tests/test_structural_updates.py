"""Structural updates, `prf.tie`, and the method rule (ADR-0002, decisions 1 to 3)."""
import inspect
import re

import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.models import Capacitor, Cascade, Resistor, Short, Wrapped
from pmrf.modules import Tied


def _rc():
    return Resistor(prf.Unconstrained(50.0)) ** Capacitor(prf.Unconstrained(1e-12))


# ---- Structural update ----------------------------------------------------------------


def test_update_swaps_a_sub_model_in_a_cascade():
    rc = _rc()
    shorted = prf.update(rc, "cascade[1]", Short())
    assert isinstance(shorted.cascade[1], Short)
    assert isinstance(shorted.cascade[0], Resistor)
    assert set(prf.params(shorted)) == {"cascade[0].R"}


def test_update_swaps_a_named_sub_model():
    load = Resistor(prf.Unconstrained(50.0), name="load")
    model = Cascade([Capacitor(prf.Unconstrained(1e-12), name="c"), load])
    swapped = prf.update(model, "load", Short())
    assert isinstance(swapped.cascade[1], Short)


def test_update_replaces_a_parameter_with_a_new_param():
    rc = _rc()
    updated = prf.update(rc, "cascade[0].R", prf.Bounded(0.0, 100.0, value=75.0))
    new = prf.params(updated)["cascade[0].R"]
    assert np.allclose(new.value, 75.0)
    assert new.bounds is not None


def test_update_node_bypasses_validation():
    class Holder(prf.Module):
        gain: prf.Param = prf.param()

    # A plain string is not a parameter; the structural form does not check.
    updated = prf.update(Holder(gain=1.0), "gain", "not a parameter")
    assert updated.gain == "not a parameter"


def test_update_node_with_callable_selector():
    rc = _rc()
    shorted = prf.update(rc, lambda m: m.cascade[1], Short())
    assert isinstance(shorted.cascade[1], Short)


def test_update_fn_replaces_each_selected_part():
    load = Resistor(prf.Unconstrained(50.0, scale=2.0), name="load")
    cap = Capacitor(prf.Unconstrained(1e-12), name="c")
    model = load ** cap
    updated = prf.update(model, "load.*", fn=lambda p: prf.update(p, fixed=True))
    assert prf.params(updated)["load.R"].fixed
    assert not prf.params(updated)["c.C"].fixed


def test_update_fn_on_a_sub_model():
    rc = _rc()
    updated = prf.update(rc, lambda m: m.cascade[0], fn=lambda r: r.flipped())
    assert not isinstance(updated.cascade[0], Resistor)


def test_update_fn_on_several_names():
    rc = Resistor(prf.Unconstrained(1.0), name="r") ** Capacitor(prf.Unconstrained(2.0), name="c")
    updated = prf.update(rc, ["r.R", "c.C"], fn=lambda p: prf.update(p, value=p.value * 3))
    assert prf.param_values(updated) == pytest.approx({"r.R": 3.0, "c.C": 6.0})


def test_update_structural_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown"):
        prf.update(_rc(), "nope", Short())


def test_update_structural_form_mismatches_raise():
    rc = _rc()
    with pytest.raises(TypeError, match="forms"):
        prf.update(rc, "cascade[1]", Short(), fn=lambda x: x)
    with pytest.raises(TypeError, match="forms"):
        prf.update(rc, "cascade[0].R", 3.0, value=3.0)
    with pytest.raises(TypeError, match="forms"):
        prf.update(rc, {"cascade[0].R": 3.0}, Short())
    with pytest.raises(TypeError, match="forms"):
        prf.update(rc, fn=lambda x: x)


def test_update_overlapping_structural_selection_raises():
    rc = _rc()
    with pytest.raises(ValueError, match="overlap"):
        prf.update(rc, ["cascade[1]", "cascade[1].C"], fn=lambda x: x)


def test_update_docstring_says_structural_forms_bypass_validation():
    doc = prf.update.__doc__
    assert "bypass" in doc and "validation" in doc and "converters" in doc


# ---- tie ------------------------------------------------------------------------------


def test_tie_is_top_level():
    from pmrf import parameters
    assert prf.tie is parameters.tie


def test_tie_model_keeps_rf_interface():
    frequency = prf.Frequency(1.0, 2.0, 3, unit="GHz")
    model = _rc()
    tied = prf.tie(model, "cascade[0].R", "cascade[1].C", fn=lambda c: c * 50e12)
    assert isinstance(tied, Wrapped)
    assert isinstance(tied.wrapped, Tied)
    assert jnp.allclose(tied.s(frequency), prf.unwrap(tied.wrapped).s(frequency))
    assert set(prf.params(tied)) == {"cascade[1].C"}
    assert np.allclose(tied.build().cascade[0].R, 50.0)


def test_tie_default_is_identity():
    rc = Resistor(prf.Unconstrained(50.0), name="r1") ** Resistor(prf.Unconstrained(75.0), name="r2")
    tied = prf.tie(rc, "r2.R", "r1.R")
    assert np.allclose(tied.build().cascade[1].R, 50.0)


def test_tie_after_update_tracks_source():
    rc = Resistor(prf.Unconstrained(50.0), name="r") ** Capacitor(prf.Unconstrained(1e-12), name="c")
    tied = prf.tie(rc, "r.R", "c.C", fn=lambda c: c * 5e13)
    assert np.allclose(prf.update(tied, {"c.C": 2e-12}).build().cascade[0].R, 100.0)


def test_tie_module_returns_tied():
    class Pair(prf.Module):
        first: prf.Param = prf.param(default=1.0, as_free=True)
        second: prf.Param = prf.param(default=2.0, as_free=True)

    tied = prf.tie(Pair(), "second", "first", fn=lambda v: 3 * v)
    assert isinstance(tied, Tied)
    resolved = prf.unwrap(tied)
    assert jnp.allclose(resolved.second, 3 * resolved.first)


def test_tie_unknown_name_raises():
    with pytest.raises(ValueError, match="not found"):
        prf.tie(_rc(), "cascade[0].R", "nope")


# ---- The method rule (ADR-0002, decision 1) --------------------------------------------


def _public_methods(cls):
    names = set()
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(cls, name)
        if isinstance(attr, property):
            continue
        if callable(attr) or isinstance(attr, (staticmethod, classmethod)):
            names.add(name)
    return names


def test_module_has_no_public_methods():
    assert _public_methods(prf.Module) == set()


def test_module_docstring_states_its_purpose():
    doc = prf.Module.__doc__.lower()
    assert "nam" in doc and "validat" in doc and "base class" in doc


# ADR-0002 decision 1: the RF methods `Model` keeps. `nports` and `port_tuples` are
# properties, and `**` and `@` are dunders, so neither appears here.
MODEL_METHODS = {
    "s", "a", "z", "y", "mna", "primary_matrix", "build", "expand",
    "cascaded", "flipped", "renumbered", "terminated", "to_skrf", "export_touchstone",
}

# ADR-0002 decision 1: the methods `Model.__init_subclass__` generates per class
# (`s_db`, `s_mn_mag`, ...) are RF and allowed; the plotting names `__getattr__`
# serves are not class attributes. Both are out of scope for the allowlist.
_GENERATED = re.compile(r"^[a-z]+(_mn)?_[a-z0-9_]+$")


def test_model_public_methods_equal_allowlist():
    class Sub(prf.Model):
        def s(self, freq):
            return jnp.zeros((freq.npoints, 1, 1))

    assert _public_methods(prf.Model) == MODEL_METHODS
    generated = {n for n in _public_methods(Sub) - MODEL_METHODS if getattr(getattr(Sub, n), "_pmrf_auto", False)}
    assert all(_GENERATED.match(n) for n in generated)
    assert _public_methods(Sub) - generated == MODEL_METHODS


def test_param_has_no_at():
    assert not hasattr(prf.Param, "at")


# ---- repr (ADR-0002, decision 5) -------------------------------------------------------


def test_module_repr_shows_declared_values():
    class RC(prf.Model):
        R: prf.Param = prf.param()
        C: prf.Param = prf.param(scale=1e-12)

        def s(self, freq):
            return jnp.zeros((freq.npoints, 1, 1))

    text = repr(RC(R=50.0, C=2.0))
    assert "2." in text and "e-12" not in text


def test_tied_repr_shows_derived_target():
    rc = Resistor(prf.Unconstrained(50.0), name="r") ** Capacitor(prf.Unconstrained(1e-12), name="c")
    text = repr(prf.tie(rc, "r.R", "c.C", fn=lambda c: c * 5e13).wrapped)
    assert "50." in text
