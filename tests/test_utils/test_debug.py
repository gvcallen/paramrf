import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import pmrf as prf
from pmrf.models import MixedModeConverter
from pmrf.utils import error_if


def _check(v):
    return error_if(v, v <= 0, "value {}", v)


def _stdout(capfd):
    jax.effects_barrier()
    return capfd.readouterr().out


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(jax.jit, id="jax-jit"),
        pytest.param(eqx.filter_jit, id="jit"),
        pytest.param(jax.vmap, id="vmap"),
        pytest.param(lambda f: jax.vmap(eqx.filter_jit(f)), id="vmap-jit"),
        pytest.param(lambda f: eqx.filter_jit(jax.vmap(f)), id="jit-vmap"),
    ],
)
def test_passing_check_prints_nothing(transform, capfd):
    out = transform(_check)(jnp.array([1.0, 2.0]))
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


def test_equal_reference_impedances_are_silent_under_vmap(capfd):
    freq = prf.Frequency(1, 2, 3, "GHz")
    s = jax.vmap(lambda z0: MixedModeConverter().s(freq, z0))(jnp.array([50.0, 75.0]))
    s.block_until_ready()
    assert _stdout(capfd) == ""


def test_differentiable_through_print_arguments():
    f = lambda v: error_if(jnp.sin(v), v <= 0, "value {}", v)
    assert jnp.allclose(jax.grad(f)(1.0), jnp.cos(1.0))
    assert jnp.allclose(jax.jacfwd(f)(1.0), jnp.cos(1.0))
