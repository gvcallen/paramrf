import jax.numpy as jnp

import pmrf as prf
from pmrf.fitting import fit_minimize
from pmrf.models import Capacitor, Resistor, Wrapped
from pmrf.modules import Tied


class GainModule(prf.Module):
    gain: prf.Param = prf.param(default=1.0, as_free=True)


def test_model_is_a_module():
    assert issubclass(prf.Model, prf.Module)


def test_module_parameter_helpers():
    module = GainModule(gain=2.0, name="gain")

    assert module.named_params() == {"gain": 2.0}
    updated = module.at("gain").set(prf.Unconstrained(3.0, name="gain"))
    assert jnp.allclose(updated.gain, 3.0)


def test_module_parameters_can_be_tied():
    class Pair(prf.Module):
        first: prf.Param = prf.param(default=1.0, as_free=True)
        second: prf.Param = prf.param(default=2.0, as_free=True)

    tied = Pair().tied(
        target=lambda item: item.second,
        source=lambda item: item.first,
        tie_fn=lambda value: 3 * value,
    )

    resolved = prf.unwrap(tied)
    assert jnp.allclose(resolved.second, 3 * resolved.first)
    assert set(tied.named_params(free_only=True)) == {"first"}


def test_tied_model_preserves_rf_interface():
    frequency = prf.Frequency(1.0, 2.0, 3, unit="GHz")
    model = Resistor(R=prf.Unconstrained(50.0)) ** Capacitor(
        C=prf.Unconstrained(1e-12)
    )
    tied = model.tied(
        target=lambda item: item.cascade[0].R,
        source=lambda item: item.cascade[1].C,
        tie_fn=lambda value: value * 50e12,
    )

    assert isinstance(tied, Wrapped)
    assert isinstance(tied, prf.models.AbstractBuilder)
    assert isinstance(tied.wrapped, Tied)
    assert isinstance(tied.build(), prf.Model)
    assert jnp.allclose(tied.s(frequency), prf.unwrap(tied.wrapped).s(frequency))
    assert isinstance(tied ** model, prf.models.Cascade)


def test_tied_model_stacking_keeps_one_rf_wrapper():
    model = Resistor(R=prf.Unconstrained(50.0)) ** Capacitor(
        C=prf.Unconstrained(1e-12)
    )
    once = model.tied(
        target=lambda item: item.cascade[0].R,
        source=lambda item: item.cascade[1].C,
    )
    twice = once.tied(
        target=lambda item: item.cascade[0].R,
        source=lambda item: item.cascade[1].C,
    )

    assert isinstance(twice, Wrapped)
    assert isinstance(twice.wrapped, Tied)
    assert not isinstance(twice.wrapped.module, Wrapped)


def test_wrapped_probabilistic_model_preserves_rf_interface():
    frequency = prf.Frequency(1.0, 2.0, 3, unit="GHz")
    model = Resistor(R=prf.Unconstrained(50.0))
    probabilistic = prf.modules.Probabilistic(
        module=model,
        distribution=prf.distributions.Normal(50.0, 1.0),
        target=lambda item: item.R,
    )
    wrapped = Wrapped(wrapped=probabilistic)

    assert jnp.allclose(wrapped.s(frequency), prf.unwrap(probabilistic).s(frequency))


def test_fit_minimize_accepts_module():
    module = GainModule(gain=1.0)
    frequency = prf.Frequency(1.0, 2.0, 3, unit="GHz")
    target = jnp.full((3,), 4.0)

    result = fit_minimize(
        module,
        target,
        frequency=frequency,
        features=lambda item, freq: jnp.full((freq.npoints,), item.gain),
    )

    assert isinstance(result.model, GainModule)
    assert jnp.allclose(result.model.gain, 4.0, atol=1e-3)


def test_fit_minimize_accepts_arbitrary_pytree():
    params = {"gain": jnp.asarray(1.0)}
    frequency = prf.Frequency(1.0, 2.0, 3, unit="GHz")
    target = jnp.full((3,), 4.0)

    result = fit_minimize(
        params,
        target,
        frequency=frequency,
        features=lambda tree, freq: jnp.full((freq.npoints,), tree["gain"]),
    )

    assert isinstance(result.model, dict)
    assert jnp.allclose(result.model["gain"], 4.0, atol=1e-3)
