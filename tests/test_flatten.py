import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import parax as prx
import pytest

import pmrf as prf
from pmrf.distributions import Normal, Uniform
from pmrf.models import Capacitor, Resistor
from pmrf.problems import PriorPenalized, SummedTerms


class System(prf.Module):
    load: Resistor
    cap: Capacitor
    coeffs: prf.Param
    w: prf.Param
    gain: prf.Param


def make_system():
    return System(
        load=Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load"),
        cap=Capacitor(C=prf.Random(Normal(1.0, 0.1), value=1.1, scale=1e-12)),
        coeffs=prf.Bounded(0.0, 2.0, value=jnp.array([0.5, 1.0, 1.5])),
        w=prf.Unconstrained(value=jnp.arange(6.0).reshape(2, 3)),
        gain=prf.Fixed(3.0),
    )


def test_names_expand_arrays_in_c_order():
    flat = prf.flatten(make_system())
    assert flat.names == (
        "load.R", "cap.C", "coeffs[0]", "coeffs[1]", "coeffs[2]",
        "w[0,0]", "w[0,1]", "w[0,2]", "w[1,0]", "w[1,1]", "w[1,2]",
    )


def test_names_follow_free_named_params_order():
    model = make_system()
    base = [name.split("[")[0] for name in prf.flatten(model).names]
    assert list(dict.fromkeys(base)) == list(model.named_params(free_only=True))


@pytest.mark.parametrize("space", ["physical", "unconstrained"])
def test_names_align_with_theta(space):
    model = make_system()
    flat = prf.flatten(model, space=space)
    assert flat.theta0.shape == (len(flat.names),)

    values = model.values(free_only=True)
    physical = prf.flatten(model).theta0
    for i, name in enumerate(flat.names):
        base, _, index = name.partition("[")
        expected = jnp.asarray(values[base])
        if index:
            expected = expected[tuple(int(k) for k in index[:-1].split(","))]
        assert np.allclose(physical[i], expected, rtol=1e-6), name


def test_physical_theta_is_scaled():
    flat = prf.flatten(make_system())
    assert np.isclose(flat.theta0[flat.names.index("cap.C")], 1.1e-12)


def test_unconstrained_theta_maps_through_bijector():
    flat = prf.flatten(make_system(), space="unconstrained")
    # A Uniform(45, 55) prior at its midpoint sits at zero on the real line.
    assert np.isclose(flat.theta0[flat.names.index("load.R")], 0.0, atol=1e-6)
    assert np.allclose(flat.theta0[flat.names.index("w[1,2]")], 5.0)


def test_invalid_space_raises():
    with pytest.raises(ValueError, match="space"):
        prf.flatten(make_system(), space="cube")


@pytest.mark.parametrize("space", ["physical", "unconstrained"])
def test_round_trip(space):
    model = make_system()
    flat = prf.flatten(model, space=space)
    theta = flat.theta0 + 0.01

    wrapped = flat.wrap(theta)
    assert isinstance(wrapped, System)
    assert wrapped.load.R.distribution == model.load.R.distribution
    assert wrapped.cap.C.scale == 1e-12
    assert wrapped.gain.fixed
    assert np.allclose(prf.flatten(wrapped, space=space).theta0, theta, rtol=1e-5)

    unwrapped = flat.unflatten(theta)
    assert jax.tree.structure(unwrapped) == jax.tree.structure(prx.unwrap(wrapped))
    for a, b in zip(jax.tree.leaves(unwrapped), jax.tree.leaves(prx.unwrap(wrapped))):
        assert np.allclose(a, b, rtol=1e-5)
    assert np.isclose(unwrapped.gain, 3.0)


def test_unflatten_evaluates_model():
    load = Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load")
    freq = prf.Frequency(1, 2, 3, "GHz")
    flat = prf.flatten(load)
    s = flat.unflatten(jnp.array([51.0])).s(freq)
    assert np.allclose(s, load.with_values({"R": 51.0}).s(freq))


def test_frozen_parameters_are_excluded():
    model = make_system().with_fixed("cap.*")
    assert "cap.C" not in prf.flatten(model).names


def test_generic_parax_tree():
    tree = {
        "x": prx.Random(Normal(0.0, 1.0), value=jnp.array([0.1, 0.2])),
        "y": jnp.array(1.0),
        "z": prx.Fixed(prx.Real(jnp.array(2.0))),
    }
    flat = prf.flatten(tree)
    assert flat.names == ("x[0]", "x[1]", "y")
    assert np.allclose(flat.unflatten(jnp.array([1.0, 2.0, 3.0]))["x"], [1.0, 2.0])


def _log_prior_reference(model, theta_physical):
    flat = prf.flatten(model)
    tree = flat.unflatten(theta_physical)
    return (
        Uniform(45.0, 55.0).log_prob(tree.load.R)
        + Normal(1.0, 0.1).log_prob(tree.cap.C / 1e-12) - jnp.log(1e-12)
    )


def test_log_prior_physical_folds_in_scale():
    model = make_system()
    flat = prf.flatten(model)
    assert np.isclose(flat.log_prior(flat.theta0), _log_prior_reference(model, flat.theta0))


def test_log_prior_includes_jacobian():
    model = make_system()
    flat_u = prf.flatten(model, space="unconstrained")
    flat_p = prf.flatten(model)
    theta = flat_u.theta0 + 0.1

    def to_physical(t):
        return prf.flatten(flat_u.wrap(t)).theta0

    _, log_det = jnp.linalg.slogdet(jax.jacfwd(to_physical)(theta))
    expected = flat_p.log_prior(to_physical(theta)) + log_det
    assert np.isclose(flat_u.log_prior(theta), expected, rtol=1e-6)


def test_log_prior_jit_and_grad():
    flat = prf.flatten(make_system(), space="unconstrained")
    value = eqx.filter_jit(flat.log_prior)(flat.theta0)
    grad = eqx.filter_jit(jax.grad(flat.log_prior))(flat.theta0)
    assert np.isfinite(value)
    assert grad.shape == flat.theta0.shape and np.all(np.isfinite(grad))


def test_grad_through_unflatten_s():
    load = Resistor(R=prf.Random(Uniform(45.0, 55.0), value=50.0), name="load")
    freq = prf.Frequency(1, 2, 3, "GHz")
    flat = prf.flatten(load, space="unconstrained")

    @eqx.filter_jit
    def objective(theta):
        return jnp.sum(jnp.abs(flat.unflatten(theta).s(freq)) ** 2) + flat.log_prior(theta)

    assert np.all(np.isfinite(jax.grad(objective)(flat.theta0)))


def test_joint_prior_from_probabilistic_module():
    inner = Resistor(R=prf.Unconstrained(value=50.0), name="load")
    dist = Normal(50.0, 2.0)
    model = prf.modules.Probabilistic(inner, dist, target=lambda m: m.R)
    flat = prf.flatten(model)
    assert len(flat.names) == 1
    theta = jnp.array([51.0])
    assert np.isclose(flat.log_prior(theta), dist.log_prob(51.0))


def test_unconstrained_log_prior_matches_prior_penalized():
    model = make_system()
    flat = prf.flatten(model, space="unconstrained")
    problem = PriorPenalized(SummedTerms(model, (lambda m: jnp.asarray(0.0),)))
    theta = flat.theta0 + 0.2

    log_det = jnp.linalg.slogdet(
        jax.jacfwd(lambda t: prf.flatten(flat.wrap(t)).theta0)(theta)
    )[1]
    penalized = eqx.tree_at(lambda p: p.problem.model, problem, flat.wrap(theta))
    assert np.isclose(-penalized() + log_det, flat.log_prior(theta), rtol=1e-6)


@pytest.mark.parametrize("space", ["physical", "unconstrained"])
def test_no_free_parameters(space):
    model = make_system().with_free([])
    flat = prf.flatten(model, space=space)
    assert flat.names == () and flat.theta0.shape == (0,)
    assert np.isfinite(flat.log_prior(flat.theta0))


def test_joint_prior_unconstrained_includes_jacobian():
    inner = Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="load")
    model = prf.modules.Probabilistic(inner, Normal(50.0, 2.0), target=lambda m: m.R)
    flat_u = prf.flatten(model, space="unconstrained")
    theta = flat_u.theta0 + 0.3

    def to_physical(t):
        return prf.flatten(flat_u.wrap(t)).theta0

    log_det = jnp.log(jnp.abs(jax.jacfwd(to_physical)(theta)[0, 0]))
    expected = Normal(50.0, 2.0).log_prob(to_physical(theta)[0]) + log_det
    assert np.isclose(flat_u.log_prior(theta), expected, rtol=1e-6)
