"""The minimiser and samplers on raw-space parameter values (ADR-0002, decision 8).

Both go through `prf.values`, `prf.update` and `prf.log_prior` in raw space, so
these tests check the solvers against those public functions directly.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

import pmrf as prf
from pmrf.constraints import Interval
from pmrf.distributions import Uniform
from pmrf.infer import base as infer_base
from pmrf.models import Resistor
from pmrf.optimize import base as optimize_base
from pmrf.optimize.solvers.optimistix import BFGS
from pmrf.optimize.solvers.scipy import ScipyMinimize


class RLC(prf.Model):
    R: prf.Param = prf.param()
    C: prf.Param = prf.param(scale=1e-12)
    L: prf.Param = prf.param(scale=1e-9)

    def s(self, freq):
        w = freq.w
        z = self.R + 1 / (1j * w * self.C) + 1j * w * self.L
        g = (z - 50.0) / (z + 50.0)
        return g[:, None, None]


FREQ = prf.Frequency(1.0, 10.0, 21, "GHz")


def _start():
    return RLC(
        R=prf.Bounded(0.0, 100.0, value=50.0),
        C=prf.Random(Uniform(1.0, 3.0), value=2.0),
        L=prf.Fixed(0.5),
        name="rlc",
    )


def _objective(target):
    s_target = RLC(R=30.0, C=1.5, L=0.5).s(FREQ) if target is None else target

    def fn(model, args):
        return jnp.sum(jnp.abs(model.s(FREQ) - s_target) ** 2)

    return fn


# ---- Minimiser ------------------------------------------------------------------------


def test_raw_values_share_the_declared_dtype():
    """A Uniform prior's bijector clips with a float32 epsilon; the raw value must still
    have the declared value's dtype, or JAX-native solvers reject the mixed tree."""
    raw = prf.values(_start(), free_only=True, space="raw")
    assert raw["C"].dtype == raw["R"].dtype == jnp.asarray(2.0).dtype


@pytest.mark.parametrize("solver", [BFGS(), ScipyMinimize(show_progress=False)], ids=["bfgs", "scipy"])
def test_minimizer_recovers_rc_optimum(solver):
    model = _start()
    fitted, result = optimize_base.run_minimizer(_objective(None), model, solver, max_iter=2000)

    values = prf.values(fitted)
    assert values["R"] == pytest.approx(30.0, rel=1e-3)
    assert values["C"] == pytest.approx(1.5, rel=1e-3)


def test_minimizer_keeps_fixed_names_scales_and_priors():
    model = _start()
    fitted, _ = optimize_base.run_minimizer(_objective(None), model, BFGS(), max_iter=2000)

    assert fitted.name == "rlc"
    assert fitted.L.fixed and np.allclose(fitted.L.value, 0.5)
    assert fitted.C.scale == 1e-12 and fitted.L.scale == 1e-9
    assert fitted.C.distribution == model.C.distribution
    assert fitted.R.bounds is not None and np.allclose(fitted.R.bounds, model.R.bounds)
    assert jax.tree.structure(fitted) == jax.tree.structure(model)


class _RecordingMinimizer(optimize_base.AbstractUnconstrainedMinimizer):
    """Records what it is given and returns `y0` unchanged."""

    def run(self, fn, y0, args, max_iter=1024, **kwargs):
        _RecordingMinimizer.seen = (y0, fn(y0, args))
        return optimize_base.MinimizeResult(y=y0)


def test_jax_native_minimizer_receives_name_keyed_raw_values():
    model = _start()
    optimize_base.run_minimizer(_objective(None), model, _RecordingMinimizer())
    y0, value = _RecordingMinimizer.seen

    expected = prf.values(model, free_only=True, space="raw")
    assert isinstance(y0, dict) and set(y0) == {"R", "C"}
    for name in expected:
        assert np.allclose(y0[name], expected[name])
    assert np.allclose(value, _objective(None)(prf.unwrap(model), None))


def test_minimizer_start_on_a_bound_raises():
    model = prf.update(_start(), {"R": 0.0})
    with pytest.raises(ValueError, match=r"'R' start on a bound"):
        optimize_base.run_minimizer(_objective(None), model, BFGS())


def test_minimizer_start_at_nan_raises():
    """NaN is a different bug from an infinite raw value, and says so."""
    model = prf.update(_start(), {"R": jnp.nan})
    with pytest.raises(ValueError, match=r"'R' start at NaN"):
        optimize_base.run_minimizer(_objective(None), model, BFGS())


def test_minimizer_without_free_parameters_raises():
    model = prf.update(_start(), "*", fixed=True)
    with pytest.raises(ValueError, match="no free parameters"):
        optimize_base.run_minimizer(_objective(None), model, BFGS())


# ---- Samplers -------------------------------------------------------------------------


def _loglikelihood(model, args):
    return -0.5 * _objective(None)(model, args) / 0.01


def _raw_log_posterior(model, v):
    at = prf.update(model, v, space="raw")
    return _loglikelihood(prf.unwrap(at), None) + prf.log_prior(at, space="raw")


def _offsets(y0):
    return jax.tree.map(lambda x: jnp.stack([x, x + 0.1, x - 0.2]), y0)


class _StubJointSampler(infer_base.AbstractJointSampler):
    """Evaluates the log posterior at a fixed batch around `y0`."""

    def run(self, logposterior_fn, y0, args, key, init_samples=None, max_steps=None, **kwargs):
        samples = _offsets(y0)
        fn_values = jax.vmap(lambda y: logposterior_fn(y, args))(samples)
        return infer_base.SampleResult(samples=samples, fn_values=fn_values)


class _StubSplitSampler(infer_base.AbstractSplitSampler):
    """Evaluates the likelihood and prior separately at a fixed batch around `y0`."""

    def run(self, loglikelihood_fn, logprior_fn, y0, args, key, init_samples=None, max_steps=None, **kwargs):
        samples = _offsets(y0)
        fn_values = jax.vmap(lambda y: loglikelihood_fn(y, args) + logprior_fn(y, args))(samples)
        return infer_base.SampleResult(samples=samples, fn_values=fn_values)


@pytest.mark.parametrize("sampler", [_StubJointSampler(), _StubSplitSampler()], ids=["joint", "split"])
def test_sampler_log_posterior_is_likelihood_plus_raw_log_prior(sampler):
    model = _start()
    batched, results = infer_base.run_sampler(_loglikelihood, model, sampler, jax.random.key(0))

    y0 = prf.values(model, free_only=True, space="raw")
    assert set(results.samples) == {"R", "C"}
    for i in range(3):
        v = jax.tree.map(lambda x: x[i], results.samples)
        assert np.allclose(results.fn_values[i], _raw_log_posterior(model, v), rtol=1e-6)
    assert np.allclose(results.fn_values[0], _raw_log_posterior(model, y0), rtol=1e-6)


def test_sampler_writes_back_free_parameters_only():
    model = _start()
    batched, results = infer_base.run_sampler(_loglikelihood, model, _StubJointSampler(), jax.random.key(0))

    assert batched.R.raw_value.shape == (3,)
    assert np.allclose(batched.R.raw_value, results.samples["R"])
    assert batched.L.fixed and np.shape(batched.L.value) == ()
    assert batched.C.scale == 1e-12 and batched.C.distribution == model.C.distribution
    assert batched.name == "rlc"


def test_sampler_init_samples_are_read_in_raw_space():
    model = _start()
    init = prf.update(model, {"R": jnp.array([10.0, 20.0]), "C": jnp.array([1.5, 2.5])})

    class Recording(infer_base.AbstractJointSampler):
        def run(self, logposterior_fn, y0, args, key, init_samples=None, max_steps=None, **kwargs):
            Recording.init = init_samples
            return infer_base.SampleResult(samples=_offsets(y0), fn_values=jnp.zeros(3))

    infer_base.run_sampler(_loglikelihood, model, Recording(), jax.random.key(0), init_samples=init)
    expected = prf.values(init, free_only=True, space="raw")
    assert set(Recording.init) == {"R", "C"}
    for name in expected:
        assert np.allclose(Recording.init[name], expected[name])


def test_sampler_init_samples_missing_a_parameter_raises():
    """`init_samples` promises the same parameter names as `model`; a bare KeyError on
    an internal name would not say which parameter the caller left out."""
    model = _start()
    init = prf.update(prf.update(model, {"R": jnp.array([10.0, 20.0])}), "C", fixed=True)
    with pytest.raises(ValueError, match=r"'C' is free in the model but missing or fixed"):
        infer_base.run_sampler(_loglikelihood, model, _StubJointSampler(), jax.random.key(0), init_samples=init)


class _StubHypercubeSampler(infer_base.AbstractHypercubeSampler):
    """Maps a fixed batch of cube points through the prior transform."""

    def run(self, loglikelihood_fn, prior_transform_fn, u0, args, key, init_cube_samples=None, max_steps=None, **kwargs):
        cubes = jax.tree.map(lambda u: jnp.stack([u, jnp.full_like(u, 0.25)]), u0)
        samples = jax.vmap(lambda u: prior_transform_fn(u, args))(cubes)
        fn_values = jax.vmap(lambda y: loglikelihood_fn(y, args))(samples)
        return infer_base.SampleResult(samples=samples, fn_values=fn_values)


def _all_priors():
    return RLC(
        R=prf.Random(Uniform(0.0, 100.0), value=50.0),
        C=prf.Random(Uniform(1.0, 3.0), value=2.0),
        L=prf.Fixed(0.5),
    )


def test_hypercube_sampler_moves_through_declared_priors():
    model = _all_priors()
    batched, results = infer_base.run_sampler(_loglikelihood, model, _StubHypercubeSampler(), jax.random.key(0))

    # The cube point u0 is the prior CDF at the starting values; 0.25 is a quarter of the way.
    assert np.allclose(results.samples["R"], [50.0, 25.0], atol=1e-4)
    assert np.allclose(results.samples["C"], [2.0, 1.5], atol=1e-4)
    assert np.allclose(batched.R.value, [50.0, 25.0], atol=1e-4)
    assert batched.L.fixed and np.shape(batched.L.value) == ()
    expected = _loglikelihood(prf.unwrap(prf.update(model, {"R": 25.0, "C": 1.5})), None)
    assert np.allclose(results.fn_values[1], expected, rtol=1e-5)


def test_hypercube_sampler_accepts_a_start_on_a_bound():
    """A hypercube sampler moves through declared space, so an infinite raw value —
    a parameter sitting exactly on a bound — is no obstacle to it."""
    # An Interval constraint's bijector has no clipping epsilon, so a value exactly on
    # a bound really is infinite in raw space.
    model = RLC(
        R=prf.Random(Uniform(0.0, 100.0), constraint=Interval(0.0, 100.0), value=0.0),
        C=prf.Random(Uniform(1.0, 3.0), value=2.0),
        L=prf.Fixed(0.5),
    )
    assert not np.isfinite(prf.values(model, free_only=True, space="raw")["R"])
    # The same model is out of bounds for a sampler that does move raw values.
    with pytest.raises(ValueError, match=r"'R' start on a bound"):
        infer_base.run_sampler(_loglikelihood, model, _StubJointSampler(), jax.random.key(0))

    batched, results = infer_base.run_sampler(_loglikelihood, model, _StubHypercubeSampler(), jax.random.key(0))
    assert np.allclose(results.samples["R"], [0.0, 25.0], atol=1e-4)
    assert np.allclose(batched.R.value, [0.0, 25.0], atol=1e-4)


def test_hypercube_sampler_names_free_parameters_without_a_prior():
    with pytest.raises(ValueError, match=r"'R'"):
        infer_base.run_sampler(_loglikelihood, _start(), _StubHypercubeSampler(), jax.random.key(0))


# ---- Joint priors ---------------------------------------------------------------------


def _joint_prior():
    """A joint prior of event size one, over the raw value of a bounded parameter with no
    prior of its own, so no raw value leaves the bounds.

    The model is a :class:`pmrf.models.Wrapped`, so solvers see it unwrapped as one, and
    read its parameter through `build()`."""
    import parax.distributions as dist

    inner = Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="load")
    return prf.prior(inner, ["R"], dist.MultivariateNormalDiag(jnp.array([0.2]), jnp.array([0.5])), space="raw")


def _declared_R(model):
    return prf.values(model)["R"]


def test_parameter_under_a_joint_prior_is_a_free_raw_value():
    model = _joint_prior()
    raw = prf.values(model, free_only=True, space="raw")
    assert list(raw) == ["R"]
    moved = prf.update(model, {"R": raw["R"] + 0.3}, space="raw")
    assert not np.allclose(_declared_R(moved), _declared_R(model))


def test_joint_prior_raw_log_prior_includes_jacobian():
    """Raw space is the joint prior's whitened space, and the raw log prior carries the
    Jacobian of the map from it to declared space."""
    model = _joint_prior()
    z = prf.values(model, free_only=True, space="raw")["R"] + 0.3

    def declared(z):
        return _declared_R(prf.update(model, {"R": z}, space="raw"))

    moved = prf.update(model, {"R": z}, space="raw")
    expected = prf.log_prior(moved, space="declared") + jnp.log(jnp.abs(jax.grad(declared)(z)))
    actual = prf.log_prior(moved, space="raw")
    assert np.allclose(actual, expected, rtol=1e-6)


def test_minimizer_moves_a_parameter_under_a_joint_prior():
    model = _joint_prior()

    def fn(m, args):
        return (m.build().R - 45.0) ** 2

    fitted, _ = optimize_base.run_minimizer(fn, model, BFGS(), max_iter=500)
    assert _declared_R(fitted) == pytest.approx(45.0, rel=1e-4)


def test_joint_sampler_moves_and_scores_a_parameter_under_a_joint_prior():
    model = _joint_prior()
    loglik = lambda m, args: -((m.build().R - 45.0) ** 2)
    batched, results = infer_base.run_sampler(loglik, model, _StubJointSampler(), jax.random.key(0))

    for i in range(3):
        v = jax.tree.map(lambda x: x[i], results.samples)
        at = prf.update(model, v, space="raw")
        expected = loglik(prf.unwrap(at), None) + prf.log_prior(at, space="raw")
        assert np.allclose(results.fn_values[i], expected, rtol=1e-6)
    assert np.shape(_declared_R(batched)) == (3,)


def test_hypercube_sampler_moves_a_parameter_under_a_joint_prior():
    """The cube goes through the standard normal's inverse CDF, the whitening and the
    parameter's old raw-to-declared map, and the starting value maps to the cube and back."""
    model = _joint_prior()
    _, results = infer_base.run_sampler(lambda m, a: 0.0, model, _StubHypercubeSampler(), jax.random.key(0))
    to_declared = prf.params(model)["R"].raw_to_declared_bijector
    expected = to_declared.forward(0.2 + 0.5 * norm.ppf(0.25))
    assert np.allclose(results.samples["R"], [50.0, expected], atol=1e-4)
