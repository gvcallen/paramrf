"""Attaching one-dimensional priors by name with `prf.prior` (ADR-0005, #192)."""
import parax.distributions as dist
import jax
import jax.numpy as jnp
import numpy as np
import parax as prx
import pytest

import pmrf as prf
from pmrf.distributions import Normal, Uniform, truncate
from pmrf.infer import base as infer_base
from pmrf.models import Resistor


class RC(prf.Model):
    R: prf.Param = prf.param()
    C: prf.Param = prf.param(scale=1e-12)

    def s(self, freq):
        w = freq.w
        z = self.R + 1 / (1j * w * self.C)
        g = (z - 50.0) / (z + 50.0)
        return g[:, None, None]


SPACES = ["declared", "physical", "raw"]


def _assert_same_prior(attached, built):
    """`attached` scores and moves exactly as `built`, parameter for parameter."""
    for space in SPACES:
        assert np.allclose(prf.log_prior(attached, space=space), prf.log_prior(built, space=space)), space
    raw_a, raw_b = prf.values(attached, space="raw"), prf.values(built, space="raw")
    assert set(raw_a) == set(raw_b)
    for name in raw_a:
        assert np.allclose(raw_a[name], raw_b[name]), name


def test_prior_is_top_level():
    from pmrf import parameters
    assert prf.prior is parameters.prior


def test_prior_on_unconstrained_is_random():
    m = RC(R=prf.Unconstrained(40.0), C=prf.Unconstrained(2.0))
    prior = Normal(50.0, 5.0)
    attached = prf.prior(m, "R", prior)
    built = RC(R=prf.Random(prior, value=40.0), C=prf.Unconstrained(2.0))
    _assert_same_prior(attached, built)
    assert attached.R.bounds is None or np.all(np.isinf(np.asarray(attached.R.bounds)))


def _unbounded(param):
    return param.bounds is None or np.all(np.isinf(np.asarray(param.bounds)))


def test_prior_replaces_the_range():
    """A range is prior information, so the prior replaces it. Truncation to a field's
    validity is covered in `test_parameters_validity.py`."""
    m = RC(R=prf.Bounded(0.0, 100.0, value=40.0), C=prf.Bounded(1.0, 3.0, value=2.0, scale=1e-12))
    prior = Normal(50.0, 30.0)
    attached = prf.prior(m, "R", prior)
    built = RC(R=prf.Random(prior, value=40.0), C=m.C)
    _assert_same_prior(attached, built)
    assert _unbounded(attached.R)
    assert np.allclose(attached.C.bounds, (1.0, 3.0))


def test_prior_replaces_an_existing_prior():
    m = RC(R=prf.Random(Uniform(0.0, 100.0), value=40.0), C=prf.Unconstrained(2.0))
    prior = Normal(50.0, 30.0)
    attached = prf.prior(m, "R", prior)
    built = RC(R=prf.Random(prior, value=40.0), C=prf.Unconstrained(2.0))
    _assert_same_prior(attached, built)
    assert isinstance(prx.as_unwrapped(attached.R.distribution), dist.Normal)
    assert _unbounded(attached.R)


def test_prior_already_within_bounds_is_not_truncated():
    """A prior whose support lies inside the bounds needs no truncation, so it is kept
    as given, even when it cannot be truncated."""
    m = RC(R=prf.Constrained(prf.constraints.Positive(), 40.0), C=prf.Unconstrained(2.0))
    prior = dist.Gamma(4.0, 0.1)
    attached = prf.prior(m, "R", prior)
    assert isinstance(prx.as_unwrapped(attached.R.distribution), dist.Gamma)
    _assert_same_prior(attached, RC(R=prf.Random(prior, value=40.0), C=prf.Unconstrained(2.0)))


def test_prior_by_glob_gives_each_its_own():
    a = Resistor(R=prf.Unconstrained(40.0), name="cable_a")
    b = Resistor(R=prf.Bounded(0.0, 100.0, value=60.0), name="cable_b")
    load = Resistor(R=prf.Unconstrained(50.0), name="load")
    m = a ** b ** load
    prior = Normal(50.0, 30.0)
    attached = prf.prior(m, "cable_*", prior)
    built = (
        Resistor(R=prf.Random(prior, value=40.0), name="cable_a")
        ** Resistor(R=prf.Random(prior, value=60.0), name="cable_b")
        ** load
    )
    _assert_same_prior(attached, built)
    assert prf.params(attached)["load.R"].distribution is None


def test_prior_keeps_name_scale_metadata_and_fixed_state():
    m = RC(R=prf.Fixed(40.0), C=prf.Unconstrained(2.0, metadata={"k": 1}))
    attached = prf.prior(m, ["R", "C"], Normal(2.0, 1.0))
    assert attached.R.fixed and not attached.C.fixed
    assert attached.C.scale == 1e-12 and attached.C.metadata == {"k": 1}
    assert np.allclose(prf.values(attached)["R"], 40.0)
    assert np.allclose(prf.values(attached)["C"], 2.0)


def test_prior_in_physical_space():
    m = RC(R=prf.Fixed(1.0), C=prf.Bounded(1.0, 3.0, value=2.0))
    attached = prf.prior(m, "C", Normal(2e-12, 0.5e-12), space="physical")
    built = RC(R=prf.Fixed(1.0), C=prf.Random(Normal(2.0, 0.5), value=2.0))
    for space in SPACES:
        assert np.allclose(prf.log_prior(attached, space=space), prf.log_prior(built, space=space)), space
    assert np.allclose(prf.values(attached)["C"], 2.0)
    assert _unbounded(attached.C)


def test_prior_in_raw_space_is_over_the_raw_space_before_attaching():
    m = RC(R=prf.Fixed(1.0), C=prf.Random(Uniform(1.0, 3.0), value=2.5))
    old = m.C.raw_to_declared_bijector
    z = m.C.raw_value
    prior = Normal(0.3, 0.7)
    attached = prf.prior(m, "C", prior, space="raw")
    expected = prior.log_prob(z) - old.forward_log_det_jacobian(z)
    assert np.allclose(prf.log_prior(attached), expected)
    assert np.allclose(prf.values(attached)["C"], 2.5)
    assert np.allclose(attached.C.bounds, (1.0, 3.0))


def test_prior_unknown_name_raises():
    with pytest.raises(ValueError, match="nope"):
        prf.prior(RC(R=1.0, C=2.0), "nope", Normal(0.0, 1.0))


def test_prior_bad_space_raises():
    with pytest.raises(ValueError, match="space"):
        prf.prior(RC(R=1.0, C=2.0), "R", Normal(0.0, 1.0), space="unconstrained")


def test_prior_joint_event_size_attaches_a_joint_prior():
    """Joint priors are covered in `test_parameters_joint_prior.py`."""
    joint = dist.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2))
    model = prf.prior(RC(R=prf.Unconstrained(1.0), C=prf.Unconstrained(2.0)), ["R", "C"], joint)
    assert isinstance(model.wrapped, prf.modules.Probabilistic)
    assert model.wrapped.names == ("R", "C")


def test_prior_other_event_size_raises():
    three = dist.MultivariateNormalDiag(jnp.zeros(3), jnp.ones(3))
    with pytest.raises(ValueError, match=r"event size 3.*2 parameters"):
        prf.prior(RC(R=1.0, C=2.0), ["R", "C"], three)


# ---- Hypercube samplers ---------------------------------------------------------------


class _StubHypercubeSampler(infer_base.AbstractHypercubeSampler):
    """Maps a fixed batch of cube points through the prior transform."""

    def run(self, loglikelihood_fn, prior_transform_fn, u0, args, key, init_cube_samples=None, max_steps=None, **kwargs):
        cubes = jax.tree.map(lambda u: jnp.stack([u, jnp.full_like(u, 0.25)]), u0)
        samples = jax.vmap(lambda u: prior_transform_fn(u, args))(cubes)
        fn_values = jax.vmap(lambda y: loglikelihood_fn(y, args))(samples)
        return infer_base.SampleResult(samples=samples, fn_values=fn_values)


def _loglikelihood(model, args=None):
    return -((model["R"] - 30.0) ** 2) - ((model["C"] - 2.0) ** 2)


def _dict_with_attached_priors():
    m = {"R": prf.Bounded(0.0, 100.0, value=50.0), "C": prf.Unconstrained(2.0)}
    m = prf.prior(m, "R", truncate(Normal(50.0, 30.0), 0.0, 100.0))  # it replaces the range
    return prf.prior(m, "C", Normal(2.0, 1.0))


def test_hypercube_sampler_on_attached_priors():
    _, results = infer_base.run_sampler(_loglikelihood, _dict_with_attached_priors(), _StubHypercubeSampler(), jax.random.key(0))
    r_prior = truncate(Normal(50.0, 30.0), 0.0, 100.0)
    assert np.allclose(results.samples["R"], [50.0, r_prior.icdf(0.25)], atol=1e-4)
    assert np.allclose(results.samples["C"], [2.0, Normal(2.0, 1.0).icdf(0.25)], atol=1e-4)


def test_polychord_on_attached_priors(tmp_path):
    polychord_backend = pytest.importorskip("pmrf.infer.solvers.polychord")
    if not getattr(polychord_backend, "MPI_AVAILABLE", False):
        pytest.skip("PolyChord, anesthetic, or mpi4py not installed.")

    solver = polychord_backend.PolyChord(nlive=50, num_repeats=2, do_clustering=False, base_dir=str(tmp_path), seed=0)
    batched, results = infer_base.run_sampler(_loglikelihood, _dict_with_attached_priors(), solver, jax.random.key(0))
    assert results.samples["R"].ndim == 1
    assert np.all((batched["R"].value >= 0.0) & (batched["R"].value <= 100.0))


def test_hypercube_sampler_on_a_prior_over_raw_space():
    """A prior over raw space reaches declared space through the raw-to-declared map, so
    the cube goes through the base's inverse CDF and then that map."""
    before = {"R": prf.Bounded(0.0, 100.0, value=50.0), "C": prf.Unconstrained(2.0)}
    m = prf.prior(prf.prior(before, "R", Normal(0.5, 2.0), space="raw"), "C", Normal(2.0, 1.0))
    _, results = infer_base.run_sampler(_loglikelihood, m, _StubHypercubeSampler(), jax.random.key(0))
    to_declared = prf.params(before)["R"].raw_to_declared_bijector
    assert np.allclose(results.samples["R"], [50.0, to_declared.forward(Normal(0.5, 2.0).icdf(0.25))], atol=1e-4)


def test_hypercube_sampler_on_a_prior_over_physical_space_of_a_scaled_parameter():
    m = {"R": prf.Unconstrained(50.0), "C": prf.Unconstrained(2.0, scale=1e-12)}
    m = prf.prior(prf.prior(m, "C", Normal(2e-12, 1e-13), space="physical"), "R", Normal(50.0, 1.0))
    _, results = infer_base.run_sampler(_loglikelihood, m, _StubHypercubeSampler(), jax.random.key(0))
    assert np.allclose(results.samples["C"], [2.0, Normal(2.0, 0.1).icdf(0.25)], atol=1e-4)


def test_hypercube_sampler_names_a_mapped_prior_whose_base_has_no_inverse_cdf():
    m = {"R": prf.Bounded(0.0, 100.0, value=50.0), "C": prf.Unconstrained(2.0)}
    m = prf.prior(prf.prior(m, "R", dist.Gamma(2.0, 1.0), space="raw"), "C", Normal(2.0, 1.0))
    with pytest.raises(ValueError, match=r"'R'.*space='raw' or space='physical'.*Gamma"):
        infer_base.run_sampler(_loglikelihood, m, _StubHypercubeSampler(), jax.random.key(0))
