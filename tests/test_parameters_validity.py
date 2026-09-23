"""Validity and range (ADR-0005, §Bounds; #200).

Validity is a model field's constraint and always holds. A range is what the user
supplies, and is prior information: a declared- or physical-space prior replaces it.
Every prior is truncated to its parameter's constraint and renormalised.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import parax as prx
import parax.bijectors as db
import parax.distributions as dd
import pytest

import pmrf as prf
from pmrf.constraints import Interval, Positive
from pmrf.distributions import Normal, Uniform
from pmrf.modules import Probabilistic
from pmrf.models import Resistor


class Width(prf.Model):
    """A model whose parameter has a validity: a width is positive."""
    w: prf.Param = prf.param(constraint=Positive())

    def s(self, freq):
        return jnp.zeros((len(freq), 1, 1), dtype=complex)


class Unit(prf.Model):
    """A model whose parameter is valid only in [0, 10]."""
    x: prf.Param = prf.param(constraint=Interval(0.0, 10.0))

    def s(self, freq):
        return jnp.zeros((len(freq), 1, 1), dtype=complex)


def _dist(param):
    return prx.as_unwrapped(param.distribution)


def _validity(param):
    return None if param.validity is None else prx.as_unwrapped(param.validity)


# ---- Recording validity ---------------------------------------------------------------


def test_a_field_constraint_is_the_validity():
    m = Width(prf.Bounded(1.0, 5.0, value=2.0))
    assert np.allclose(_validity(m.w).bounds, (0.0, np.inf))
    assert np.allclose(m.w.bounds, (1.0, 5.0))


def test_a_range_is_not_validity():
    assert prf.Bounded(1.0, 5.0, value=2.0).validity is None
    assert prf.Random(Normal(0.0, 1.0), constraint=Interval(-1.0, 1.0)).validity is None
    assert Resistor(prf.Bounded(1.0, 5.0, value=2.0)).R.validity is None


def test_a_parameter_passed_on_to_another_field_intersects_the_validities():
    w = Width(prf.Bounded(1.0, 5.0, value=2.0)).w
    m = Unit(w)
    assert np.allclose(_validity(m.x).bounds, (0.0, 10.0))
    assert np.allclose(m.x.bounds, (1.0, 5.0))


def test_a_range_outside_validity_is_clipped():
    m = Width(prf.Bounded(-5.0, 5.0, value=2.0))
    assert np.allclose(m.w.bounds, (0.0, 5.0))


def test_validity_survives_update_and_replace():
    m = Width(prf.Bounded(1.0, 5.0, value=2.0))
    moved = prf.update(m, {"w": 3.0})
    assert np.allclose(_validity(moved.w).bounds, (0.0, np.inf))
    assert np.allclose(_validity(prf.replace(m.w, value=4.0)).bounds, (0.0, np.inf))
    fixed = prf.update(m, "w", fixed=True)
    assert np.allclose(_validity(fixed.w).bounds, (0.0, np.inf))


# ---- A declared-space prior replaces the range and keeps validity ---------------------


def test_a_declared_prior_keeps_validity_and_drops_the_range():
    m = prf.prior(Width(prf.Bounded(0.0, 60.0, value=40.0)), "w", Normal(50.0, 10.0))
    d = _dist(m.w)
    assert isinstance(d, dd.TruncatedNormal)
    assert np.allclose([d.low, d.high], [0.0, np.inf])
    assert np.allclose(prf.values(prf.update(m, {"w": 80.0}))["w"], 80.0)
    with pytest.raises(Exception, match="outside the constraint"):
        prf.update(m, {"w": -1.0})


def test_a_declared_prior_over_a_range_without_validity_is_untruncated():
    m = prf.prior(Resistor(prf.Bounded(0.0, 60.0, value=40.0)), "*", Normal(50.0, 10.0))
    r = prf.params(m)["R"]
    assert isinstance(_dist(r), dd.Normal)
    assert np.all(np.isinf(np.asarray(r.bounds)))


def test_a_physical_prior_keeps_validity_and_drops_the_range():
    m = Width(prf.Bounded(1.0, 3.0, value=2.0, scale=1e-12))
    m = prf.prior(m, "w", Normal(2e-12, 2e-12), space="physical")
    assert np.allclose(m.w.bounds[0], 0.0, atol=1e-12)
    assert np.isinf(m.w.bounds[1])
    expected = dd.TruncatedNormal(2.0, 2.0, 0.0, jnp.inf).log_prob(2.0) + jnp.log(1e12)
    assert np.allclose(prf.log_prior(m, space="physical"), expected)


def test_a_raw_prior_keeps_the_range():
    m = Width(prf.Bounded(1.0, 5.0, value=2.0))
    m = prf.prior(m, "w", Normal(0.0, 1.0), space="raw")
    assert np.allclose(m.w.bounds, (1.0, 5.0))
    assert np.allclose(_validity(m.w).bounds, (0.0, np.inf))


# ---- Truncation ------------------------------------------------------------------------


def test_a_random_is_truncated_to_its_constraint():
    p = prf.Random(Normal(50.0, 10.0), constraint=Interval(0.0, 60.0))
    d = _dist(p)
    x = jnp.linspace(0.0, 60.0, 20001)
    assert np.allclose(jnp.trapezoid(jnp.exp(d.log_prob(x)), x), 1.0, atol=1e-6)
    assert 0.0 <= float(d.icdf(0.999)) <= 60.0


def test_a_random_through_a_field_constraint_is_truncated():
    m = Width(prf.Random(Normal(1.0, 2.0)))
    d = _dist(m.w)
    assert isinstance(d, dd.TruncatedNormal)
    assert np.allclose([d.low, d.high], [0.0, np.inf])


def test_a_random_whose_support_lies_inside_its_constraint_is_kept():
    m = Width(prf.Random(dd.Gamma(2.0, 1.0)))
    assert isinstance(_dist(m.w), dd.Gamma)


def test_a_non_truncatable_random_with_a_narrowing_constraint_raises():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Gamma.*cannot be truncated"):
            prf.Random(dd.Gamma(2.0, 1.0), constraint=Interval(0.0, 1.0))


def test_a_non_truncatable_random_through_a_narrowing_field_constraint_raises():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Gamma.*cannot be truncated"):
            Unit(prf.Random(dd.Gamma(2.0, 1.0)))


def test_a_uniform_random_is_truncated_to_its_constraint():
    p = prf.Random(Uniform(0.0, 100.0), constraint=Interval(10.0, 20.0))
    d = _dist(p)
    assert np.allclose([d.low, d.high], [10.0, 20.0])


# ---- The dropped-bounds case -------------------------------------------------------------


def test_a_prior_on_a_constrained_random_keeps_the_bounds_it_should():
    """Regression: `prf.prior` on a `prf.Random` built with an explicit constraint.

    Over raw space the old range is kept. Over declared space the range is replaced,
    and the validity of the field is kept."""
    random = prf.Random(Normal(50.0, 10.0), constraint=Interval(0.0, 60.0))
    raw = prf.prior(Resistor(random), "*", Normal(0.0, 1.0), space="raw")
    assert np.allclose(prf.params(raw)["R"].bounds, (0.0, 60.0))
    declared = prf.prior(Width(random), "w", Normal(40.0, 5.0))
    assert np.allclose(declared.w.bounds[0], 0.0, atol=1e-12)
    assert np.isinf(declared.w.bounds[1])


# ---- Joint priors -------------------------------------------------------------------------


NAMES = ["a.w", "b.w"]


def _gaussian(mean, cov):
    return dd.MultivariateNormalFullCovariance(jnp.asarray(mean), jnp.asarray(cov))


def test_a_declared_joint_prior_over_ranges_is_accepted():
    parts = {
        "a": Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="a"),
        "b": Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="b"),
    }
    model = prf.prior(parts, ["a.R", "b.R"], _gaussian([50.0, 50.0], [[4.0, 1.0], [1.0, 4.0]]))
    assert isinstance(model, Probabilistic)
    # The ranges are replaced, so a value outside them can be written.
    moved = prf.update(model, {"a.R": 80.0})
    assert np.allclose(prf.values(moved)["a.R"], 80.0)
    assert prf.params(model)["a.R"].bounds is None


def test_a_declared_joint_prior_whose_support_leaves_validity_raises():
    parts = {"a": Width(prf.Bounded(1.0, 5.0, value=2.0), name="a"), "b": Width(prf.Unconstrained(2.0), name="b")}
    with pytest.raises(ValueError, match=r"'a\.w', 'b\.w'.*space='raw'"):
        prf.prior(parts, NAMES, _gaussian([2.0, 2.0], [[1.0, 0.0], [0.0, 1.0]]))


def test_a_declared_joint_prior_inside_validity_keeps_validity():
    parts = {"a": Width(prf.Bounded(1.0, 5.0, value=2.0), name="a"), "b": Width(prf.Unconstrained(2.0), name="b")}
    squash = db.Chain([db.Block(db.Shift(jnp.array([1.0, 1.0])), 1), db.Block(db.Exp(), 1)])
    positive = dd.Transformed(dd.Independent(dd.Normal(jnp.zeros(2), jnp.ones(2))), squash)
    model = prf.prior(parts, NAMES, positive)
    a = prf.params(model)["a.w"]
    assert np.allclose(_validity(a).bounds, (0.0, np.inf))
    assert np.allclose(a.bounds, (0.0, np.inf))
    with pytest.raises(Exception, match="outside the constraint"):
        prf.update(model, {"a.w": -1.0})


def test_a_raw_joint_prior_keeps_the_ranges():
    parts = {
        "a": Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="a"),
        "b": Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="b"),
    }
    model = prf.prior(parts, ["a.R", "b.R"], _gaussian([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]), space="raw")
    assert np.allclose(prf.params(model)["a.R"].bounds, (40.0, 60.0))
