import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import pmrf as prf
from pmrf.models import (
    CapacitorQ, CoaxialLine, InductorQ, MicrostripLine, MixedModeConverter,
)
from pmrf.utils import error_if


def _check(v):
    return error_if(v, v <= 0, "value {}", v)


def _stdout(capfd):
    jax.effects_barrier()
    return capfd.readouterr().out


@pytest.mark.parametrize(
    "transform, values",
    [
        pytest.param(jax.jit, jnp.array([1.0, 2.0]), id="jax-jit"),
        pytest.param(eqx.filter_jit, jnp.array([1.0, 2.0]), id="jit"),
        pytest.param(jax.vmap, jnp.array([1.0, 2.0]), id="vmap"),
        pytest.param(lambda f: jax.vmap(eqx.filter_jit(f)), jnp.array([1.0, 2.0]), id="vmap-jit"),
        pytest.param(lambda f: eqx.filter_jit(jax.vmap(f)), jnp.array([1.0, 2.0]), id="jit-vmap"),
        pytest.param(lambda f: jax.vmap(jax.vmap(f)), jnp.array([[1.0, 2.0], [3.0, 4.0]]), id="vmap-vmap"),
    ],
)
def test_passing_check_prints_nothing(transform, values, capfd):
    out = transform(_check)(values)
    out.block_until_ready()
    assert _stdout(capfd) == ""


def test_failing_check_under_jit_prints_value_and_raises(capfd):
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(_check)(jnp.array(-1.0)).block_until_ready()
    out = _stdout(capfd)
    assert out == "value -1.0\n"


def test_failing_check_under_vmap_raises_with_runtime_values(capfd):
    with pytest.raises(eqx.EquinoxRuntimeError) as excinfo:
        jax.vmap(_check)(jnp.array([1.0, -1.0])).block_until_ready()
    out = _stdout(capfd)
    assert out == "value -1.0\n"
    assert "Tracer" not in str(excinfo.value)


@pytest.mark.parametrize(
    "transform, values, expected",
    [
        pytest.param(jax.jit, jnp.array(-1.0), ["value -1.0"], id="jit"),
        pytest.param(jax.vmap, jnp.array([-1.0, 2.0]), ["value -1.0"], id="vmap"),
        pytest.param(lambda f: jax.jit(jax.vmap(f)), jnp.array([-1.0, 2.0]), ["value -1.0"], id="jit-vmap"),
        pytest.param(lambda f: jax.vmap(jax.jit(f)), jnp.array([-1.0, 2.0]), ["value -1.0"], id="vmap-jit"),
        pytest.param(lambda f: jax.vmap(jax.vmap(f)), jnp.array([[-1.0, 2.0], [3.0, -2.0]]), ["value -1.0", "value -2.0"], id="vmap-vmap"),
    ],
)
def test_failing_check_reports_each_bad_element(transform, values, expected, capfd):
    with pytest.raises(Exception):
        transform(_check)(values).block_until_ready()
    report = _stdout(capfd)
    for line in expected:
        assert report.count(line) == 1
    assert "Tracer" not in report


def test_passing_check_has_no_runtime_host_callback(monkeypatch):
    calls = []
    pure_callback = jax.pure_callback

    def count_callback(callback, *args, **kwargs):
        def counted(*values):
            calls.append(None)
            return callback(*values)

        return pure_callback(counted, *args, **kwargs)

    monkeypatch.setattr(jax, "pure_callback", count_callback)
    checked = jax.jit(jax.vmap(_check))
    checked(jnp.array([1.0, 2.0])).block_until_ready()
    assert calls == []


def test_scalar_and_nested_print_arguments_under_vmap(capfd):
    def check(value):
        return error_if(value, value < 0, "{} {bad}", {"value": value}, bad=1.0)

    with pytest.raises(eqx.EquinoxRuntimeError):
        jax.vmap(check)(jnp.array([2.0, -1.0])).block_until_ready()
    report = _stdout(capfd)
    assert report.count("-1.") == 1
    assert report.endswith(" 1.0\n")


@pytest.mark.parametrize(
    "model, name, value",
    [
        pytest.param(CapacitorQ(C=1e-12, Q=30.0), "Q", 30.0, id="capacitor-q"),
        pytest.param(InductorQ(L=1e-9, Q=30.0), "Q", 30.0, id="inductor-q"),
        pytest.param(MicrostripLine(length=0.1), "w", 1e-3, id="microstrip"),
        pytest.param(CoaxialLine(length=0.1), "d_in", 1e-3, id="coaxial"),
    ],
)
@pytest.mark.parametrize("batched", [False, True], ids=["jit", "jit-vmap"])
def test_updated_models_make_no_runtime_callback(model, name, value, batched, monkeypatch):
    calls = []
    pure_callback = jax.pure_callback

    def count_callback(callback, *args, **kwargs):
        def counted(*values):
            calls.append(None)
            return callback(*values)

        return pure_callback(counted, *args, **kwargs)

    monkeypatch.setattr(jax, "pure_callback", count_callback)
    freq = prf.Frequency(1, 2, 3, "GHz")
    evaluate = lambda v: prf.update(model, {name: v}).s(freq)
    if batched:
        evaluate = jax.vmap(evaluate)
        value = jnp.array([value, value * 1.1])
    jax.jit(evaluate)(jnp.asarray(value)).block_until_ready()
    assert calls == []


@pytest.mark.parametrize("batched", [False, True], ids=["jit", "jit-vmap"])
def test_mixed_mode_converter_makes_no_runtime_callback(batched, monkeypatch):
    calls = []
    pure_callback = jax.pure_callback

    def count_callback(callback, *args, **kwargs):
        def counted(*values):
            calls.append(None)
            return callback(*values)

        return pure_callback(counted, *args, **kwargs)

    monkeypatch.setattr(jax, "pure_callback", count_callback)
    freq = prf.Frequency(1, 2, 3, "GHz")
    evaluate = lambda z0: MixedModeConverter().s(freq, z0=jnp.full((4,), z0))
    if batched:
        evaluate = jax.vmap(evaluate)
        z0 = jnp.array([50.0, 75.0])
    else:
        z0 = jnp.array(50.0)
    jax.jit(evaluate)(z0).block_until_ready()
    assert calls == []


def test_equal_reference_impedances_are_silent_under_vmap(capfd):
    freq = prf.Frequency(1, 2, 3, "GHz")
    s = jax.vmap(lambda z0: MixedModeConverter().s(freq, z0))(jnp.array([50.0, 75.0]))
    s.block_until_ready()
    assert _stdout(capfd) == ""


def test_differentiable_through_print_arguments():
    f = lambda v: error_if(jnp.sin(v), v <= 0, "value {}", v)
    assert jnp.allclose(jax.grad(f)(1.0), jnp.cos(1.0))
    assert jnp.allclose(jax.jacfwd(f)(1.0), jnp.cos(1.0))


def test_grad_through_parameter_update():
    model = CapacitorQ(C=1e-12, Q=prf.Unconstrained(30.0))
    objective = lambda q: prf.update(model, {"Q": q}).Q.value
    assert jnp.allclose(jax.grad(objective)(30.0), 1.0)
