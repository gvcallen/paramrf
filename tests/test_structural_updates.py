"""Structural updates, `prf.tie`, and the method rule (ADR-0002, decisions 1 to 3)."""
import inspect
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.math import CONVERSION_LOOKUP
from pmrf.models.base import PLOT_DOMAINS
from pmrf.distributions import Normal
from pmrf.distributions import RelativeTruncatedNormal as RTNormal
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


# ---- resolve --------------------------------------------------------------------------


def _named_resistor(value, name):
    return Resistor(prf.Unconstrained(value), name=name)


def test_resolve_is_top_level():
    from pmrf import parameters
    assert prf.resolve is parameters.resolve


def test_resolve_docstring_separates_structure_from_evaluation():
    doc = prf.resolve.__doc__
    assert "pmrf.unwrap" in doc
    assert "structural" in doc and "value" in doc


def test_resolve_returns_a_container_to_its_own_shape():
    parts = {"a": _named_resistor(50.0, "a"), "b": _named_resistor(1.0, "b")}
    resolved = prf.resolve(prf.tie(parts, "b.R", "a.R"))

    assert isinstance(resolved, dict)
    assert list(resolved) == ["a", "b"]
    assert np.allclose(resolved["b"].R, 50.0)


def test_resolve_keeps_untied_parameters_and_values_the_target():
    parts = {"a": _named_resistor(50.0, "a"), "b": _named_resistor(1.0, "b")}
    resolved = prf.resolve(prf.tie(parts, "b.R", "a.R"))

    assert prf.is_param(resolved["a"].R)
    assert not prf.is_param(resolved["b"].R)


def test_resolve_after_update_carries_the_update_through_the_tie():
    parts = {"a": _named_resistor(50.0, "a"), "b": _named_resistor(1.0, "b")}
    tied = prf.tie(parts, "b.R", "a.R")

    resolved = prf.resolve(prf.update(tied, {"a.R": 75.0}))
    assert np.allclose(resolved["b"].R, 75.0)


def test_resolve_applies_a_non_identity_tie_function():
    parts = {"a": _named_resistor(50.0, "a"), "b": _named_resistor(1.0, "b")}
    tied = prf.tie(parts, "b.R", "a.R", fn=lambda r: r * 2.0)

    assert np.allclose(prf.resolve(tied)["b"].R, 100.0)


def test_resolve_applies_stacked_ties():
    parts = {
        "a": _named_resistor(50.0, "a"),
        "b": _named_resistor(1.0, "b"),
        "c": _named_resistor(1.0, "c"),
    }
    tied = prf.tie(parts, "b.R", "a.R", fn=lambda r: r * 2.0)
    tied = prf.tie(tied, "c.R", "a.R", fn=lambda r: r * 3.0)

    resolved = prf.resolve(tied)
    assert set(prf.params(tied)) == {"a.R"}
    assert np.allclose(resolved["b"].R, 100.0)
    assert np.allclose(resolved["c"].R, 150.0)


def test_resolve_returns_a_list_as_a_list():
    parts = [_named_resistor(50.0, "a"), _named_resistor(1.0, "b")]
    resolved = prf.resolve(prf.tie(parts, "b.R", "a.R"))

    assert isinstance(resolved, list)
    assert np.allclose(resolved[1].R, 50.0)
    assert prf.is_param(resolved[0].R)


def test_resolve_returns_a_nested_container_in_its_own_shape():
    parts = {"group": {"a": _named_resistor(50.0, "a"), "b": _named_resistor(1.0, "b")}}
    resolved = prf.resolve(prf.tie(parts, "b.R", "a.R", fn=lambda r: r * 2.0))

    assert isinstance(resolved, dict) and isinstance(resolved["group"], dict)
    assert np.allclose(resolved["group"]["b"].R, 100.0)
    assert prf.is_param(resolved["group"]["a"].R)


def test_resolve_returns_a_model_whose_rf_methods_work():
    frequency = prf.Frequency(1.0, 2.0, 3, unit="GHz")
    tied = prf.tie(_rc(), "cascade[0].R", "cascade[1].C", fn=lambda c: c * 50e12)

    resolved = prf.resolve(tied)
    assert isinstance(resolved, prf.Model)
    assert jnp.allclose(resolved.s(frequency), tied.s(frequency))
    assert prf.is_param(prf.params(resolved)["cascade[1].C"])
    # The tie is gone: the RF wrapper stays, because it carries the RF interface,
    # but nothing below it is still unwrappable.
    assert not isinstance(resolved.wrapped, Tied)
    assert not prf.is_param(prf.params(resolved)["cascade[0].R"])


def test_resolve_without_ties_returns_the_tree_unchanged():
    parts = {"a": _named_resistor(50.0, "a"), "b": _named_resistor(75.0, "b")}
    resolved = prf.resolve(parts)

    assert bool(eqx.tree_equal(resolved, parts))
    assert prf.is_param(resolved["a"].R) and prf.is_param(resolved["b"].R)


def test_resolve_leaves_a_probabilistic_subtree_wrapped():
    """`Probabilistic` bears a prior: discharging it is evaluation, not structure.

    It absorbs the parameters below it and holds one raw value, so unwrapping it
    turns a joint prior into a number the same way `prf.unwrap` turns a `Param`
    into its value. `resolve` is structural, so it leaves the wrapper standing.
    """
    joint = prf.modules.Probabilistic(
        _named_resistor(50.0, "p"), Normal(50.0, 1.0), target=lambda m: m.R
    )
    parts = {"p": joint, "a": _named_resistor(50.0, "a"), "b": _named_resistor(1.0, "b")}

    resolved = prf.resolve(prf.tie(parts, "b.R", "a.R"))
    assert isinstance(resolved["p"], prf.modules.Probabilistic)
    assert np.allclose(resolved["b"].R, 50.0)


def test_resolve_keeps_the_tie_predicate_in_one_private_helper():
    from pmrf import parameters

    source = inspect.getsource(parameters)
    assert len(re.findall(r"prx\.Tie\b", source)) == 1
    assert "prx.Tie" in inspect.getsource(parameters._is_tie)


def test_handoff_acceptance_a_tied_container_reads_back_parameterised():
    """The handoff note's acceptance test (`notes/paramrf_tie_containers_handoff.md`).

    The note ties the sub-models `b` and `a` in one call; that is #187, so the
    tie is expanded per parameter here. The block below is this issue's: the tie
    is applied and the untied parts keep their parameters.
    """
    parts = {
        "a": Resistor(R=prf.Random(RTNormal(50.0, 0.1)), name="a"),
        "b": Resistor(R=prf.Random(RTNormal(50.0, 0.1)), name="b"),
        "c": Capacitor(C=prf.Random(RTNormal(1.0, 0.1), scale=1e-12), name="c"),
    }

    tied = parts
    for name in prf.params(parts, "a.*"):
        tied = prf.tie(tied, "b" + name[len("a"):], name)
    assert set(prf.params(tied)) == {"a.R", "c.C"}

    moved = prf.resolve(prf.update(tied, {"a.R": 55.0}))
    assert float(moved["b"].R) == 55.0
    assert prf.is_param(moved["c"].C)


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
_GENERATED = re.compile(
    rf"^({'|'.join(PLOT_DOMAINS)})(_mn)?_({'|'.join(map(re.escape, CONVERSION_LOOKUP))})$"
)


def test_model_public_methods_equal_allowlist():
    class Sub(prf.Model):
        def s(self, freq):
            return jnp.zeros((freq.npoints, 1, 1))

    assert _public_methods(prf.Model) == MODEL_METHODS
    generated = {n for n in _public_methods(Sub) if _GENERATED.match(n)}
    assert generated and all(getattr(Sub, n)._pmrf_auto for n in generated)
    assert _public_methods(Sub) - generated == MODEL_METHODS


def test_param_has_no_at():
    assert not hasattr(prf.Param, "at")


def test_all_names_only_existing_attributes():
    missing = [name for name in prf.__all__ if not hasattr(prf, name)]
    assert missing == []


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


# ---- Mapping form with sub-models (#168) ----------------------------------------------


def _named():
    return Cascade([
        Capacitor(prf.Unconstrained(1e-12), name="c"),
        Cascade([Resistor(prf.Unconstrained(10.0), name="inner"), Resistor(prf.Unconstrained(20.0))], name="sub"),
        Resistor(prf.Unconstrained(50.0), name="load"),
    ])


def _same(a, b):
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    assert ta == tb
    assert all(np.array_equal(x, y) for x, y in zip(la, lb))


def test_update_mapping_replaces_a_single_sub_model():
    model = _named()
    new = Resistor(prf.Unconstrained(75.0), name="load")
    _same(prf.update(model, {"load": new}), prf.update(model, "load", new))


def test_update_mapping_replaces_nested_and_indexed_sub_models():
    rc = _rc()
    _same(prf.update(rc, {"cascade[1]": Short()}), prf.update(rc, "cascade[1]", Short()))
    model = _named()
    short = Short()
    via_map = prf.update(model, {"sub_inner": short, "load": Short()})
    one_by_one = prf.update(prf.update(model, "sub_inner", short), "load", Short())
    _same(via_map, one_by_one)
    assert isinstance(via_map.cascade[1].cascade[0], Short)


def test_update_mapping_mixes_model_and_value_entries():
    model = _named()
    mixed = prf.update(model, {"load": Short(), "c.C": 2e-12}, space="declared")
    _same(mixed, prf.update(prf.update(model, "load", Short()), {"c.C": 2e-12}))
    assert isinstance(mixed.cascade[2], Short)


def test_update_mapping_model_for_parameter_name_raises():
    with pytest.raises(TypeError, match="parameter name"):
        prf.update(_named(), {"c.C": Short()})


def test_update_mapping_array_for_sub_model_name_raises():
    with pytest.raises(TypeError, match="names a sub-model"):
        prf.update(_named(), {"load": 3.0})


def test_update_mapping_unknown_sub_model_raises():
    with pytest.raises(ValueError, match="Unknown sub-model"):
        prf.update(_named(), {"nope": Short()})


def test_update_mapping_ambiguous_sub_model_raises():
    model = Cascade([Resistor(prf.Unconstrained(1.0, name="r1"), name="x"), Resistor(prf.Unconstrained(2.0, name="r2"), name="x")])
    with pytest.raises(ValueError, match="ambiguous"):
        prf.update(model, {"x": Short()})


def test_update_mapping_overlapping_parts_raise():
    with pytest.raises(ValueError, match="overlap"):
        prf.update(_named(), {"sub": Short(), "sub_inner": Short()})
    with pytest.raises(ValueError, match="overlap"):
        prf.update(_named(), {"load": Short(), "load.R": 1.0})
