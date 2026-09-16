import jax
import pytest
import pmrf as prf
from pmrf.models import Resistor, Inductor, Capacitor, Cascade

def test_explicit_parameter_name():
    """Test that directly naming a parameter overrides the JAX path."""
    val = prf.Unconstrained(2.0, name="custom_param")
    res = Resistor(val)
    params = prf.param_values(res)
    
    assert "custom_param" in params
    assert params["custom_param"] == 2.0

def test_model_namespace():
    """Test that naming a model prefixes its child parameters when contained in another model."""
    res = Resistor(2.0, name="myR")
    cas1 = Cascade([res], name="myCas")
    cas2 = Cascade([cas1])
    params = prf.params(cas2)
    
    assert any(k.startswith("myCas_myR") for k in params.keys())

def test_namespace_separator_is_not_configurable():
    """Nested named modules are always joined with `_`."""
    res = Resistor(2.0, name="myR")
    cas2 = Cascade([Cascade([res], name="myCas")])
    with pytest.raises(TypeError):
        prf.params(cas2, namespace_separator='*')

def test_name_collision_raises_error():
    """Test that identical names in the same hierarchy raise a ValueError."""
    res = Resistor(2.0, name="myR")
    ind = Inductor(2.0)
    cap = Capacitor(2.0)

    cas1 = Cascade([res, ind, cap])
    cas2 = Cascade([res, ind, cap])
    
    combined_model = cas1 ** cas2

    with pytest.raises(ValueError, match="name collision"):
        prf.params(combined_model)


def test_at_string_target():
    """Test that .at() accepts a string parameter name."""
    r = Resistor(prf.Unconstrained(50.0, name="custom_R"))
    
    val = r.at("custom_R").get()
    assert prf.unwrap(val) == 50.0
    
    new_r = r.at("custom_R").set(prf.Unconstrained(100.0, name="custom_R"))
    assert prf.unwrap(new_r.at("custom_R").get()) == 100.0

def test_at_multiple_string_targets():
    """Test that .at() accepts an iterable of string parameter names."""
    rc = Resistor(prf.Unconstrained(50.0, name="custom_R")) ** Capacitor(prf.Unconstrained(10.0, name="custom_C"))
    
    # Get multiple using a tuple
    vals = rc.at(("custom_R", "custom_C")).get()
    unwrapped_vals = tuple(prf.unwrap(v) for v in vals)
    assert unwrapped_vals == (50.0, 10.0)
    
    # Set multiple using a list
    new_rc = rc.at(["custom_R", "custom_C"]).set((
        prf.Unconstrained(100.0, name="custom_R"), 
        prf.Unconstrained(20.0, name="custom_C")
    ))
    assert prf.unwrap(new_rc.at("custom_R").get()) == 100.0
    assert prf.unwrap(new_rc.at("custom_C").get()) == 20.0

def test_tied_string_targets():
    """Test that .tied() accepts string parameter names for source and target."""
    from pmrf.models import Wrapped
    from pmrf.modules import Tied
    
    rc = Resistor(prf.Unconstrained(50.0, name="custom_R")) ** Capacitor(prf.Unconstrained(10.0, name="custom_C"))
    
    # Tie custom_R to custom_C using strings
    tied_rc = rc.tied(target="custom_R", source="custom_C", tie_fn=lambda c: c * 5.0)
    
    # If the resolution failed, it would throw an error before instantiation
    assert isinstance(tied_rc, Wrapped)
    assert isinstance(tied_rc.wrapped, Tied)

def test_target_resolution_errors():
    """Test that invalid target formats or non-existent names raise appropriate errors."""
    r = Resistor(prf.Unconstrained(50.0, name="custom_R"))
    
    # Test non-existent string name
    with pytest.raises(ValueError, match="not resolve parameter name"):
        r.at("nonexistent_param")
        
    with pytest.raises(ValueError, match="not resolve parameter name"):
        r.at(123)
        
    # Test that tied checks both target and source
    with pytest.raises(ValueError, match="not found in the provided lookup"):
        r.tied(target="custom_R", source="nonexistent_param")

def test_at_nested_namespace():
    """Test that .at() resolves string targets using nested model namespaces."""
    r = Resistor(prf.Unconstrained(50.0, name="res_val"), name="myR")
    cas1 = Cascade([r], name="myCas")
    
    # Go a level deeper: cas2 acts as the root, so cas1's name ("myCas") 
    # will be properly traversed and added to the namespace.
    cas2 = Cascade([cas1])
    
    expected_namespace_name = "myCas_myR_res_val"
    
    # Verify the value can be retrieved using the fully namespaced string
    val = cas2.at(expected_namespace_name).get()
    assert prf.unwrap(val) == 50.0
    
    # Verify the value can be updated using the fully namespaced string
    new_cas2 = cas2.at(expected_namespace_name).set(prf.Unconstrained(100.0, name="res_val"))
    assert prf.unwrap(new_cas2.at(expected_namespace_name).get()) == 100.0


def test_tied_nested_namespace():
    """Test that .tied() resolves string targets using nested model namespaces."""
    from pmrf.models import Wrapped
    from pmrf.modules import Tied
    
    r = Resistor(prf.Unconstrained(50.0, name="res_val"), name="myR")
    c = Capacitor(prf.Unconstrained(10.0, name="cap_val"), name="myC")
    cas1 = Cascade([r, c], name="myCas")
    
    # Go a level deeper so "myCas" acts as a namespace prefix for its children
    cas2 = Cascade([cas1])
    
    target_name = "myCas_myR_res_val"
    source_name = "myCas_myC_cap_val"
    
    # Tie the nested resistor's value to the nested capacitor's value
    tied_cas = cas2.tied(
        target=target_name,
        source=source_name,
        tie_fn=lambda val: val * 5.0
    )
    
    assert isinstance(tied_cas, Wrapped)
    assert isinstance(tied_cas.wrapped, Tied)


# ---- One name resolver (#133) ---------------------------------------------------------

import numpy as np
import skrf
from pmrf.distributions import Uniform
from pmrf.models import DatasheetLine, FloatingLine, Touchstone


class _System(prf.Module):
    components: dict


def _cable(name="cable"):
    return DatasheetLine(
        zn=prf.Random(Uniform(45, 55), value=50.0), vf=prf.Fixed(0.7),
        k1=prf.Random(Uniform(1.0, 3.0), value=2.4),
        k2=prf.Random(Uniform(1.0, 10.0), value=3.5, scale=1e-3),
        length=prf.Random(Uniform(120, 130), value=127.0, scale=1e-3), name=name,
    )


def _system():
    load = Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load")
    return _System(components={"cable": _cable(), "load": load})


def _assert_names_resolve(tree, expected=None):
    names = prf.params(tree)
    assert names, "no names resolved"
    if expected is not None:
        assert set(names) == set(expected)
    full = prf.params(tree)
    for name in names:
        node = full[name]
        physical = node.physical_value if prf.is_param(node) else node
        assert np.allclose(prf.unwrap(tree.at(name).get()), physical)


def test_names_resolve_on_frozen_tree():
    """B1: freezing every parameter must not hide their names."""
    system = _system()
    frozen = system.map(prf.freeze, is_target=prf.is_param)

    assert set(prf.params(frozen)) == set(prf.params(system))
    _assert_names_resolve(frozen, prf.params(system))
    assert prf.params(frozen, free_only=True) == {}


def test_names_resolve_on_frozen_submodule():
    system = _system()
    frozen = system.at(lambda m: m.components["load"]).apply(prf.freeze)

    assert set(prf.params(frozen)) == set(prf.params(system))
    _assert_names_resolve(frozen)
    assert "load.R" not in prf.params(frozen, free_only=True)


def test_frozen_subtree_arrays_are_not_named():
    """Raw arrays frozen as constants are data, not parameters."""
    import jax.numpy as jnp

    class Holder(prf.Module):
        data: object
        gain: prf.Param

    holder = Holder(data=prf.freeze(jnp.ones(3)), gain=prf.Unconstrained(2.0))
    assert set(prf.params(holder)) == {"gain"}


def test_unfreeze_submodule_unfreezes_nested_params():
    system = _system()
    frozen = system.map(prf.freeze, is_target=prf.is_param)
    thawed = frozen.at(
        lambda m: (m.components["load"], m.components["cable"])
    ).apply(lambda ms: tuple(prf.unfreeze(x) for x in ms))

    assert set(prf.params(thawed, free_only=True)) == set(prf.params(system, free_only=True))


def test_names_resolve_after_evaluating_touchstone_output(tmp_path):
    """B2: evaluating an output must not mutate the pytree."""
    f = skrf.Frequency(1, 100, 11, "MHz")
    path = tmp_path / "dut"
    skrf.Network(frequency=f, s=np.full((11, 1, 1), 0.1 + 0.05j), name="dut").write_touchstone(str(path))
    freq = prf.Frequency(1, 100, 11, "MHz")

    class Sys(prf.Module):
        line: DatasheetLine
        ts: Touchstone

        @property
        def out(self):
            return self.line.terminated(self.ts)

    sys = Sys(line=DatasheetLine(zn=50.0, vf=0.7, k1=2.4, k2=3.5e-3, length=0.1, name="cable"),
              ts=Touchstone(str(path) + ".s1p"))
    before = set(vars(sys.ts))
    sys.out.s(freq)

    assert set(vars(sys.ts)) == before
    assert sys.ts.nports == 1
    _assert_names_resolve(sys)


def test_dict_identifier_keys_are_dotted():
    """U5: identifier string keys use dotted names, others keep brackets."""
    class D(prf.Module):
        params: dict

    d = D(params={"R": prf.Param(value=1.0), "not valid": prf.Param(value=2.0)})
    assert set(prf.params(d)) == {"params.R", "params['not valid']"}
    _assert_names_resolve(d)

    root = {"R": prf.Unconstrained(1.0)}
    assert set(prf.params(root)) == {"R"}


def test_named_module_collapses_through_containers():
    line = _cable()
    expected = {"cable.zn", "cable.vf", "cable.k1", "cable.k2", "cable.length"}
    assert set(prf.params(_System(components={"cable": line}))) == expected
    assert set(prf.params(Cascade([line]))) == expected
    assert set(prf.params(FloatingLine(floating=line))) == expected


def test_names_resolve_through_wrappers():
    """U1: one resolver, names relative to the wrapped module."""
    system = _system()
    names = prf.params(system)

    _assert_names_resolve(system, names)
    tied = system.tied("load.R", "cable.zn")
    assert set(prf.params(tied)) == set(names) - {"load.R"}
    _assert_names_resolve(tied)

    frozen_tied = tied.map(prf.freeze, is_target=prf.is_param)
    _assert_names_resolve(frozen_tied, prf.params(tied))

    rc = Resistor(prf.Unconstrained(50.0), name="r") ** Capacitor(prf.Unconstrained(1e-12), name="c")
    wrapped = rc.tied("r.R", "c.C", lambda c: c * 5e13)
    assert set(prf.params(wrapped)) == {"c.C"}
    _assert_names_resolve(wrapped)
    assert np.allclose(wrapped.at("c.C").set(prf.Unconstrained(2e-12)).build().cascade[0].R, 100.0)


def _two_resistor_circuit():
    from pmrf.models import Port, Circuit
    r1 = Resistor(R=prf.Unconstrained(50.0), name="r1")
    r2 = Resistor(R=prf.Unconstrained(75.0), name="r2")
    p0, p1 = Port(), Port()
    return Circuit([[(p0, 0), (r1, 0)], [(r1, 1), (r2, 0)], [(r2, 1), (p1, 0)]])


def test_names_resolve_after_evaluating_circuit():
    """Evaluating a Circuit builds its cached topology without mutating the pytree."""
    freq = prf.Frequency(1, 10, 5, "GHz")
    circuit = _two_resistor_circuit()
    frozen = circuit.map(prf.freeze, is_target=prf.is_param)
    tied = circuit.tied("r2.R", "r1.R")

    for tree in (circuit, frozen, tied):
        before = set(vars(tree))
        tree.s(freq)
        tree.y(freq)
        assert set(vars(tree)) == before
        _assert_names_resolve(tree)
    assert set(prf.params(frozen)) == set(prf.params(circuit))


def test_names_resolve_after_evaluating_tied_and_frozen_touchstone(tmp_path):
    f = skrf.Frequency(1, 100, 11, "MHz")
    path = tmp_path / "dut"
    skrf.Network(frequency=f, s=np.full((11, 1, 1), 0.1 + 0.05j), name="dut").write_touchstone(str(path))
    freq = prf.Frequency(1, 100, 11, "MHz")

    line = DatasheetLine(zn=50.0, vf=0.7, k1=2.4, k2=3.5e-3, length=0.1, name="cable")
    model = line.terminated(Touchstone(str(path) + ".s1p"))
    tied = model.tied("cable.k1", "cable.zn", lambda z: z * 0.048)
    frozen = model.map(prf.freeze, is_target=prf.is_param)

    for tree in (tied, frozen):
        tree.s(freq)
        _assert_names_resolve(tree)
    assert set(prf.params(frozen)) == set(prf.params(model))
# ---- Values and free sets by name (#134) -----------------------------------------------


def test_param_values_round_trip():
    system = _system()
    values = prf.param_values(system)
    assert set(values) == set(prf.params(system))
    for name, p in prf.params(system).items():
        assert np.allclose(values[name], p.value)

    same = prf.update(system, values)
    assert prf.params(same).keys() == prf.params(system).keys()
    for name, p in prf.params(system).items():
        q = prf.params(same)[name]
        assert np.allclose(q.value, p.value)
        assert q.fixed == p.fixed and q.scale == p.scale and q.name == p.name
        assert q.distribution == p.distribution


def test_param_values_free_only():
    system = _system()
    assert set(prf.param_values(system, free_only=True)) == set(prf.params(system, free_only=True))
    assert "cable.vf" not in prf.param_values(system, free_only=True)


def test_update_takes_declared_values():
    system = _system()
    updated = prf.update(system, {"load.R": 51.0, "cable.length": 125.0, "cable.vf": 0.8})
    names = prf.params(updated)
    assert np.allclose(names["load.R"].value, 51.0)
    assert np.allclose(names["cable.length"].physical_value, 0.125)
    assert np.allclose(names["cable.length"].value, 125.0)
    assert names["cable.vf"].fixed and np.allclose(names["cable.vf"].value, 0.8)
    assert names["cable.length"].distribution is not None


def test_update_on_frozen_tree():
    frozen = jax.tree.map(prf.freeze, _system(), is_leaf=prf.is_param)
    updated = prf.update(frozen, {"load.R": 52.0})
    assert np.allclose(prf.param_values(updated)["load.R"], 52.0)
    assert prf.params(updated, free_only=True) == {}


def test_update_unknown_name_raises():
    with pytest.raises(ValueError, match="nope"):
        prf.update(_system(), {"nope": 1.0, "load.R": 49.0})


def test_update_out_of_bounds_raises():
    with pytest.raises(Exception, match="outside the constraint"):
        prf.update(_system(), {"load.R": 60.0})


def test_update_fixed_does_not_unfreeze():
    """Fixed state is a parameter's; freezing is a sub-tree's, and update leaves it."""
    system = _system()
    frozen = jax.tree.map(prf.freeze, system, is_leaf=prf.is_param)
    freed = prf.update(frozen, ["load.*", "cable.length"], fixed=False)
    assert prf.params(freed, free_only=True) == {}
    free = prf.update(prf.update(system, "*", fixed=True), ["load.*", "cable.length"], fixed=False)
    assert set(prf.params(free, free_only=True)) == {"load.R", "cable.length"}
    assert set(prf.params(free)) == set(prf.params(system))


def test_update_fixed_false_frees_fixed_by_construction():
    free = prf.update(_system(), "cable.*", fixed=False)
    assert set(prf.params(free, free_only=True)) == {
        "cable.zn", "cable.vf", "cable.k1", "cable.k2", "cable.length", "load.R",
    }


def test_update_fixed():
    system = _system()
    fixed = prf.update(system, "cable.k*", fixed=True)
    assert set(prf.params(fixed, free_only=True)) == {"cable.zn", "cable.length", "load.R"}
    assert set(prf.params(fixed)) == set(prf.params(system))
    assert np.allclose(prf.params(fixed)["cable.k1"].value, 2.4)


def test_update_fixed_twice_is_idempotent():
    import equinox as eqx
    system = _system()
    once = prf.update(system, "cable.k*", fixed=True)
    twice = prf.update(once, "cable.k*", fixed=True)
    assert bool(eqx.tree_equal(twice, once))
    thawed = prf.update(twice, "cable.k*", fixed=False)
    assert set(prf.params(thawed, free_only=True)) == set(prf.params(system, free_only=True))


def test_update_fixed_on_nested_freezes_leaves_them_frozen():
    import equinox as eqx
    import parax as prx
    system = _system()
    nested = eqx.tree_at(
        lambda t: t.components["load"].R, system,
        prx.Freeze(prx.Freeze(system.components["load"].R)),
    )
    assert "load.R" not in prf.params(nested, free_only=True)
    free = prf.update(prf.unfreeze(nested), ["load.R", "cable.length"], fixed=False)
    assert {"load.R", "cable.length"} <= set(prf.params(free, free_only=True))
