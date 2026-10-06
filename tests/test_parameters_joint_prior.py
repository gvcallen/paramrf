"""Joint priors attached by name with `prf.prior` (ADR-0005, #193), the whitened raw
space of their parameters (#194), and array-valued parameters under them (#261)."""
import parax.bijectors as db
import parax.distributions as dd
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import parax as prx
import pytest
from jax.flatten_util import ravel_pytree

import pmrf as prf
from pmrf.distributions import Normal
from pmrf.distributions import RelativeTruncatedNormal as RTNormal
from pmrf.infer import base as infer_base
from pmrf.models import Capacitor, Resistor, Wrapped
from pmrf.modules import Probabilistic
from pmrf.parameters import tree_param_distributions, tree_param_log_prob
from pmrf.problems import PriorPenalized, SummedTerms


MU = jnp.array([3.9, 3.9])
L = jnp.array([[0.10, 0.0], [0.08, 0.06]])   # correlation 0.8
NAMES = ("a.R", "b.R")


def _gaussian(mean, tril):
    """A correlated Gaussian as an affine bijector over a standard-normal base, the
    structure of a trained flow. `Block` sums the shift's log-determinant over the event."""
    n = len(mean)
    base = dd.Independent(dd.Normal(jnp.zeros(n), jnp.ones(n)))
    return dd.Transformed(base, db.Chain([db.Block(db.Shift(jnp.asarray(mean)), 1), db.TriangularLinear(jnp.asarray(tril))]))


def _gaussian_log_prob(x, mean, tril):
    """The closed-form log density of N(mean, tril tril^T) at `x`."""
    x, mean, tril = (np.asarray(v, dtype=float) for v in (x, mean, tril))
    y = np.linalg.solve(tril, x - mean)
    return -0.5 * y @ y - np.sum(np.log(np.abs(np.diag(tril)))) - 0.5 * len(x) * np.log(2 * np.pi)


def _parts():
    return {
        "a": Resistor(R=prf.Random(RTNormal(50.0, 0.1)), name="a"),
        "b": Resistor(R=prf.Random(RTNormal(50.0, 0.1)), name="b"),
        "c": Capacitor(C=prf.Random(RTNormal(1.0, 0.1), scale=1e-12), name="c"),
    }


def _unbounded_parts():
    """As :func:`_parts`, but with `a.R` and `b.R` unbounded, so a joint prior over their
    declared or physical values fits inside their bounds."""
    return {
        "a": Resistor(R=prf.Unconstrained(50.0), name="a"),
        "b": Resistor(R=prf.Unconstrained(50.0), name="b"),
        "c": _parts()["c"],
    }


def _example():
    """The correlated-Gaussian example of #193: a joint prior over the raw values of two
    sibling sub-models' parameters, with `c.C` keeping its own prior."""
    return prf.prior(_parts(), NAMES, _gaussian(MU, L), space="raw")


# ---- Scoring --------------------------------------------------------------------------


def _in_prior_space(parts, model=None):
    """The values of `a.R` and `b.R` of `model`, by default `parts`, possibly batched, in
    the example joint prior's space: the raw space of `parts`, before it was attached."""
    own = prf.params(parts)
    declared = prf.values(parts if model is None else model)
    return jnp.stack([own[name].raw_to_declared_bijector.inverse(declared[name]) for name in NAMES], axis=-1)


def _log_det_to_declared(parts, x):
    """log|det J| of the map from the example prior's space to declared space at `x`."""
    own = prf.params(parts)
    return sum(own[name].raw_to_declared_bijector.forward_log_det_jacobian(x[..., i]) for i, name in enumerate(NAMES))


def _whitened(x):
    """The whitened values `L^-1 (x - MU)` of the example's values `x` in its prior's space."""
    return jnp.linalg.solve(L, x - MU)


def test_raw_joint_prior_scores_declared_values_through_the_old_raw_space():
    """The declared density of the parameters under the example prior is the Gaussian at
    their old raw values, less the Jacobian of the map from those to declared values."""
    parts, model = _parts(), _example()
    x = _in_prior_space(parts)
    expected = _gaussian(MU, L).log_prob(x) - _log_det_to_declared(parts, x)
    expected += prf.log_prior({"c": parts["c"]}, space="declared")
    assert np.allclose(prf.log_prior(model, space="declared"), expected, rtol=1e-10)
    assert np.allclose(eqx.filter_jit(lambda m: prf.log_prior(m, space="declared"))(model), expected, rtol=1e-10)


@pytest.mark.parametrize("space", ["declared", "physical"])
def test_joint_prior_matches_the_closed_form(space):
    parts = _unbounded_parts()
    mean, tril = [50.5, 49.0], [[2.0, 0.0], [1.6, 1.2]]
    model = prf.prior(parts, NAMES, _gaussian(mean, tril), space=space)
    values = prf.values(parts, space=space)
    expected = _gaussian_log_prob([values["a.R"], values["b.R"]], mean, tril)
    expected += float(prf.log_prior({"c": parts["c"]}, space=space))
    assert np.allclose(prf.log_prior(model, space=space), expected, rtol=1e-10)


def test_joint_prior_over_a_scaled_parameter_in_physical_space():
    """A physical-space joint prior differs from its declared-space density by the
    scale, whether or not the parameter had a prior of its own."""
    parts = {
        "a": Resistor(R=prf.Unconstrained(50.0), name="a"),
        "c": Capacitor(C=prf.Unconstrained(2.0, scale=1e-12), name="c"),
    }
    mean, tril = [49.0, 2.2e-12], [[2.0, 0.0], [0.0, 0.5e-12]]
    model = prf.prior(parts, ["a.R", "c.C"], _gaussian(mean, tril), space="physical")
    expected = _gaussian_log_prob([50.0, 2e-12], mean, tril)
    assert np.allclose(prf.log_prior(model, space="physical"), expected, rtol=1e-10)
    assert np.allclose(prf.log_prior(model, space="declared"), expected + np.log(1e-12), rtol=1e-10)


def test_joint_prior_replaces_the_parameters_own_priors():
    parts, model = _parts(), _example()
    independent = prf.log_prior(parts, space="raw")
    assert not np.allclose(prf.log_prior(model, space="raw"), independent)
    # The own priors stay attached, unused.
    assert prf.params(model)["a.R"].distribution is not None


@pytest.mark.parametrize("space", ["declared", "raw", "physical"])
def test_batched_raw_joint_prior_sums_closed_form_densities_in_every_space(space):
    """Three whitened samples with a bounded map, scales, and an independent prior."""
    mean = np.array([0.3, -0.1])
    tril = np.array([[0.8, 0.0], [0.4, 0.6]])
    z = np.array([[0.1, -0.2], [0.3, 0.4], [-1.0, 0.5]])
    c = np.array([0.2, -0.5, 0.8])
    parts = {
        "a": prf.Bounded(-2.0, 4.0, value=0.0, scale=2.0),
        "b": prf.Bounded(-2.0, 4.0, value=0.0, scale=3.0),
        "c": prf.Random(Normal(0.0, 1.0), value=0.0, scale=5.0),
    }
    model = prf.prior(parts, ["a", "b"], _gaussian(mean, tril), space="raw")
    model = prf.update(model, {"a": z[:, 0], "b": z[:, 1]}, space="raw")
    model = prf.update(model, {"c": c})
    t = mean + z @ tril.T
    sigmoid = 1 / (1 + np.exp(-t))
    log_det = np.log(6 * sigmoid * (1 - sigmoid)).sum()
    independent = np.sum(-0.5 * c ** 2 - 0.5 * np.log(2 * np.pi))
    expected = sum(_gaussian_log_prob(sample, mean, tril) for sample in t) - log_det + independent
    if space == "physical":
        expected -= len(z) * np.log(2 * 3 * 5)
    elif space == "raw":
        expected += log_det + len(z) * np.log(np.diag(tril)).sum()
    actual = prf.log_prior(model, space=space)
    assert actual.shape == ()
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_vector_order_follows_explicit_names():
    parts = prf.update(_parts(), {"a.R": 52.0})
    mean = MU + jnp.array([0.2, -0.3])
    raw = prf.values(parts, space="raw")
    log_det = _log_det_to_declared(parts, _in_prior_space(parts))
    joint = lambda m: prf.log_prior(m) - prf.log_prior({"c": parts["c"]})
    for order in (("a.R", "b.R"), ("b.R", "a.R")):
        model = prf.prior(parts, list(order), _gaussian(mean, L), space="raw")
        assert model.names == order
        expected = _gaussian(mean, L).log_prob(jnp.stack([raw[name] for name in order])) - log_det
        assert np.allclose(joint(model), expected, rtol=1e-10)


def test_a_glob_expands_to_sorted_names():
    model = prf.prior(_parts(), ["[ba].R"], _gaussian(MU, L), space="raw")
    assert model.names == ("a.R", "b.R")


def test_a_value_outside_the_bounds_scores_minus_infinity():
    parts = {
        "a": Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="a"),
        "b": Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="b"),
    }
    model = prf.prior(parts, NAMES, _gaussian([0.0, 0.0], [[2.0, 0.0], [0.0, 2.0]]), space="raw")
    distributions = tree_param_distributions(model)
    inside = prf.unwrap(model)
    assert np.isfinite(tree_param_log_prob(distributions, inside))
    outside = eqx.tree_at(lambda m: m["a"].R, inside, jnp.asarray(70.0))
    assert tree_param_log_prob(distributions, outside) == -jnp.inf


class _PositiveResistor(prf.Model):
    """A resistor whose resistance has a validity: it is positive."""
    R: prf.Param = prf.param(constraint=prf.constraints.Positive())

    def s(self, freq):
        return jnp.zeros((len(freq), 1, 1), dtype=complex)


class _ArrayPriorModel(prf.Model):
    value: prf.Param = prf.param(as_free=True, constraint=prf.constraints.Positive())


class _OutsidePriorModel(prf.Model):
    value: prf.Param = prf.param(as_free=True)


def _positive_parts():
    """`a.R` and `b.R` with a positive validity, and ranges inside it."""
    return {
        "a": _PositiveResistor(R=prf.Random(RTNormal(50.0, 0.1)), name="a"),
        "b": _PositiveResistor(R=prf.Random(RTNormal(50.0, 0.1)), name="b"),
    }


def test_nan_update_checks_values_after_raw_joint_log_normal_whitening():
    base = dd.Independent(dd.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    prior = dd.Transformed(base, db.Block(db.Exp(), 1))
    parts = {
        "a": _PositiveResistor(R=prf.Unconstrained(50.0), name="a"),
        "b": _PositiveResistor(R=prf.Unconstrained(50.0), name="b"),
    }
    model = prf.prior(parts, NAMES, prior, space="raw")

    moved = prf.update(model, {"a.R": 1000.0}, space="raw", on_invalid="nan")

    assert np.isnan(prf.values(moved)["a.R"])
    assert np.isfinite(prf.values(moved)["b.R"])
    with pytest.raises(Exception, match="outside the constraint"):
        prf.update(model, {"a.R": 1000.0}, space="raw")


@pytest.mark.parametrize("space", ["declared", "physical"])
def test_a_joint_prior_whose_support_leaves_validity_raises(space):
    """A correlated Gaussian over declared or physical values reaches outside positive
    parameters, and cannot be truncated to them exactly."""
    with pytest.raises(ValueError, match=r"'a\.R', 'b\.R'.*space='raw'"):
        prf.prior(_positive_parts(), NAMES, _gaussian([50.0, 50.0], [[2.0, 0.0], [1.6, 1.2]]), space=space)
    with pytest.raises(ValueError, match=r"'a\.R', 'b\.R'.*space='raw'"):
        prf.prior(_positive_parts(), NAMES, dd.MultivariateNormalTri(jnp.array([50.0, 50.0]), jnp.eye(2)), space=space)


def test_the_attach_error_names_only_the_parameters_whose_validity_is_left():
    parts = {"a": Resistor(R=prf.Unconstrained(50.0), name="a"), "b": _positive_parts()["b"]}
    with pytest.raises(ValueError, match=r"validity of 'b\.R'\. "):
        prf.prior(parts, NAMES, _gaussian([50.0, 50.0], [[2.0, 0.0], [1.6, 1.2]]))


def test_a_joint_prior_whose_support_fits_the_bounds_is_accepted():
    """Over parameters with only a range, whatever the support, and over bounded ones
    for a support inside the bounds."""
    model = prf.prior(_unbounded_parts(), NAMES, _gaussian(MU, L))
    assert isinstance(model, Probabilistic)
    bounded = {
        "a": Resistor(R=prf.Bounded(0.0, 1.0, value=0.5), name="a"),
        "b": Resistor(R=prf.Bounded(-1.0, 3.0, value=0.5), name="b"),
    }
    # A sigmoid over a standard-normal base, shifted: support (0, 1) x (1, 2).
    squash = db.Chain([db.Block(db.Shift(jnp.array([0.0, 1.0])), 1), db.Block(db.Sigmoid(), 1)])
    box = dd.Transformed(dd.Independent(dd.Normal(jnp.zeros(2), jnp.ones(2))), squash)
    assert isinstance(prf.prior(bounded, NAMES, box), Probabilistic)


def test_unnormalised_joint_prior_opts_out_of_support_check_but_keeps_validity():
    mean, covariance = jnp.array([50.0, 50.0]), jnp.array([[4.0, 1.0], [1.0, 4.0]])
    distribution = dd.MultivariateNormalFullCovariance(mean, covariance)
    with pytest.raises(ValueError, match="support leaves the validity"):
        prf.prior(_positive_parts(), NAMES, distribution)

    model = prf.prior(_positive_parts(), NAMES, distribution, truncate="unnormalised")
    assert model.normalised is False
    assert prf.params(model)["a.R"].distribution is None
    assert prf.params(model)["a.R"].bounds[0] == 0.0
    assert prf.params(model)["a.R"].bounds[1] == np.inf
    np.testing.assert_allclose(prf.log_prior(model), distribution.log_prob(mean), rtol=1e-12)
    invalid = prf.update(model, {"a.R": -1.0}, on_invalid="nan")
    assert prf.log_prior(invalid) == -jnp.inf
    assert eqx.filter_jit(prf.log_prior)(invalid) == -jnp.inf


def test_unnormalised_joint_prior_mirror_scores_the_same_and_counts_precision_once():
    mean = jnp.array([50.0, 49.0])
    covariance = jnp.array([[4.0, 1.5], [1.5, 3.0]])
    distribution = dd.MultivariateNormalFullCovariance(mean, covariance)
    model = prf.prior(
        _positive_parts(), NAMES, distribution, truncate="unnormalised"
    )

    def penalized(values):
        moved = prf.update(model, dict(zip(NAMES, values)))
        return PriorPenalized(SummedTerms(moved, (lambda _: jnp.asarray(0.0),)))()

    np.testing.assert_allclose(
        jax.grad(penalized)(mean), jnp.zeros_like(mean), atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        jax.hessian(penalized)(mean), np.linalg.inv(covariance), atol=1e-10, rtol=1e-10
    )
    invalid = prf.update(model, {"a.R": -1.0}, on_invalid="nan")
    assert jnp.isposinf(PriorPenalized(SummedTerms(invalid, (lambda _: 0.0,)))())


def test_unnormalised_joint_prior_masks_array_member_and_keeps_outside_prior():
    mean = jnp.array([50.0, 0.2, 0.3])
    covariance = jnp.array(
        [[4.0, 0.2, 0.1], [0.2, 0.04, 0.01], [0.1, 0.01, 0.09]]
    )
    distribution = dd.MultivariateNormalFullCovariance(mean, covariance)
    parts = {
        "scalar": _PositiveResistor(
            R=prf.Random(Normal(50.0, 2.0), value=50.0), name="scalar"
        ),
        "array": _ArrayPriorModel(value=jnp.array([0.2, 0.3]), name="array"),
        "outside": _OutsidePriorModel(
            value=prf.Random(Normal(0.0, 0.5), value=0.4), name="outside"
        ),
    }
    model = prf.prior(parts, ["scalar.R", "array.value"], distribution, truncate="unnormalised")
    expected = distribution.log_prob(mean) + Normal(0.0, 0.5).log_prob(0.4)
    np.testing.assert_allclose(prf.log_prior(model), expected, rtol=1e-12, atol=1e-12)
    assert prf.params(model)["outside.value"].distribution is not None

    invalid = prf.update(model, {"array.value": jnp.array([-1.0, 0.3])}, on_invalid="nan")
    assert prf.log_prior(invalid) == -jnp.inf

    trials = jnp.array([[0.2, 0.3], [-1.0, 0.3], [0.2, jnp.nan]])
    batched = jax.vmap(
        lambda value: prf.log_prior(
            prf.update(model, {"array.value": value}, on_invalid="nan")
        )
    )(trials)
    assert jnp.isfinite(batched[0])
    assert jnp.all(jnp.isneginf(batched[1:]))


def test_unnormalised_joint_prior_penalized_mirror_counts_array_and_outside_prior():
    mean = jnp.array([50.0, 0.2, 0.3])
    covariance = jnp.array(
        [[4.0, 0.2, 0.1], [0.2, 0.04, 0.01], [0.1, 0.01, 0.09]]
    )
    distribution = dd.MultivariateNormalFullCovariance(mean, covariance)
    parts = {
        "scalar": _PositiveResistor(
            R=prf.Random(Normal(50.0, 2.0), value=50.0), name="scalar"
        ),
        "array": _ArrayPriorModel(value=jnp.array([0.2, 0.3]), name="array"),
        "outside": _OutsidePriorModel(
            value=prf.Random(Normal(0.0, 0.5), value=0.4), name="outside"
        ),
    }
    model = prf.prior(parts, ["scalar.R", "array.value"], distribution, truncate="unnormalised")

    def penalized(values):
        moved = prf.update(
            model,
            {
                "scalar.R": values[0],
                "array.value": values[1:3],
                "outside.value": values[3],
            },
        )
        return PriorPenalized(SummedTerms(moved, (lambda _: jnp.asarray(0.0),)))()

    values = jnp.array([50.0, 0.2, 0.3, 0.0])
    precision = np.zeros((4, 4))
    precision[:3, :3] = np.linalg.inv(covariance)
    precision[3, 3] = 4.0
    np.testing.assert_allclose(jax.grad(penalized)(values), jnp.zeros_like(values), atol=1e-11)
    np.testing.assert_allclose(jax.hessian(penalized)(values), precision, rtol=1e-10, atol=1e-10)


def test_unnormalised_joint_prior_metadata_survives_copy_wrappers_and_serialization(tmp_path):
    distribution = dd.MultivariateNormalDiag(jnp.array([50.0, 50.0]), jnp.array([2.0, 2.0]))
    model = prf.prior(
        _positive_parts(), NAMES, distribution, truncate="unnormalised"
    )
    updated = prf.update(model, {"a.R": 51.0})
    resolved = prf.resolve(updated)
    nested = {"transfer": resolved}
    path = tmp_path / "unnormalised.prf"
    prf.save(path, nested)
    loaded = prf.load(path)

    assert updated.normalised is False
    assert resolved.normalised is False
    assert loaded["transfer"].normalised is False


def test_unnormalised_joint_prior_requires_declared_or_physical_event():
    model = _positive_parts()
    distribution = dd.MultivariateNormalDiag(jnp.array([50.0, 50.0]), jnp.array([2.0, 2.0]))
    with pytest.raises(ValueError, match="unnormalised.*scalar-event"):
        prf.prior(model, "a.R", Normal(50.0, 2.0), truncate="unnormalised")
    with pytest.raises(ValueError, match="unnormalised.*raw-space"):
        prf.prior(model, NAMES, distribution, space="raw", truncate="unnormalised")
    with pytest.raises(ValueError, match="truncate.*normalised.*unnormalised"):
        prf.prior(model, NAMES, distribution, truncate="normalise")


def test_unnormalised_physical_joint_prior_keeps_density_and_applies_scale_jacobian_once():
    parts = {
        "a": prf.as_param(50.0, constraint=prf.constraints.Positive(), as_free=True),
        "c": prf.as_param(
            2.0,
            constraint=prf.constraints.Positive(),
            scale=1e-12,
            as_free=True,
        ),
    }
    mean = jnp.array([49.0, 2.2e-12])
    covariance = jnp.diag(jnp.array([2.0**2, (0.5e-12)**2]))
    distribution = dd.MultivariateNormalFullCovariance(mean, covariance)
    model = prf.prior(
        parts,
        ["a", "c"],
        distribution,
        space="physical",
        truncate="unnormalised",
    )
    expected = distribution.log_prob(jnp.array([50.0, 2e-12]))

    assert model.normalised is False
    np.testing.assert_allclose(prf.log_prior(model, space="physical"), expected, rtol=1e-12)
    np.testing.assert_allclose(
        prf.log_prior(model, space="declared"), expected + jnp.log(1e-12), rtol=1e-12
    )


def test_a_raw_joint_prior_over_bounded_parameters_is_accepted():
    parts = {
        "a": Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="a"),
        "b": Resistor(R=prf.Bounded(40.0, 60.0, value=50.0), name="b"),
    }
    model = prf.prior(parts, NAMES, _gaussian([0.0, 0.0], [[2.0, 0.0], [1.6, 1.2]]), space="raw")
    assert isinstance(model, Probabilistic)


def test_two_disjoint_joint_priors_are_both_scored():
    parts = {name: Resistor(R=prf.Unconstrained(v), name=name) for name, v in zip("abcd", (1.0, 2.0, 3.0, 4.0))}
    tril = [[1.0, 0.0], [0.5, 1.0]]
    model = prf.prior(parts, ["a.R", "b.R"], _gaussian([0.0, 0.0], tril))
    model = prf.prior(model, ["c.R", "d.R"], _gaussian([1.0, 1.0], tril))
    expected = _gaussian_log_prob([1.0, 2.0], [0.0, 0.0], tril) + _gaussian_log_prob([3.0, 4.0], [1.0, 1.0], tril)
    assert np.allclose(prf.log_prior(model), expected, rtol=1e-10)


# ---- Whitened raw space (#194) --------------------------------------------------------


def test_raw_values_are_the_whitened_values_in_the_prior_space():
    """For `L z + mu`, the raw values of `a.R` and `b.R` are `L^-1 (x - mu)`, with `x`
    their values in the prior's space. `c.C` keeps its own raw value."""
    parts = prf.update(_parts(), {"a.R": 52.0, "b.R": 49.0})
    model = prf.prior(parts, NAMES, _gaussian(MU, L), space="raw")
    raw = prf.values(model, space="raw")
    np.testing.assert_allclose([raw[name] for name in NAMES], _whitened(_in_prior_space(parts)), rtol=1e-12)
    assert np.allclose(raw["c.C"], prf.values(parts, space="raw")["c.C"])
    # Reading raw values only is unchanged by `where`.
    assert np.allclose(prf.values(model, "b.R", space="raw")["b.R"], raw["b.R"])


@pytest.mark.parametrize("space", ["declared", "physical"])
def test_raw_values_are_whitened_in_every_prior_space(space):
    parts = {
        "a": Resistor(R=prf.Unconstrained(50.0), name="a"),
        "c": Capacitor(C=prf.Unconstrained(2.0, scale=1e-12), name="c"),
    }
    names = ["a.R", "c.C"]
    mean = jnp.array([49.0, 2.2]) if space == "declared" else jnp.array([49.0, 2.2e-12])
    tril = jnp.array([[2.0, 0.0], [0.1, 0.5]]) * (1.0 if space == "declared" else jnp.array([[1.0], [1e-12]]))
    model = prf.prior(parts, names, _gaussian(mean, tril), space=space)
    values = prf.values(parts, space=space)
    raw = prf.values(model, space="raw")
    expected = jnp.linalg.solve(tril, jnp.array([values[n] for n in names]) - mean)
    np.testing.assert_allclose([raw[n] for n in names], expected, rtol=1e-10)


def test_raw_round_trip_returns_the_model():
    model = _example()
    again = prf.update(model, prf.values(model, space="raw"), space="raw")
    assert jax.tree.structure(again) == jax.tree.structure(model)
    for x, y in zip(jax.tree.leaves(again), jax.tree.leaves(model)):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_a_raw_update_moves_one_whitened_coordinate():
    """Writing one raw value keeps the other whitened coordinates, so under a correlated
    prior every parameter under it can move."""
    model = _example()
    raw = prf.values(model, space="raw")
    moved = prf.update(model, {"a.R": raw["a.R"] + 0.5}, space="raw")
    after = prf.values(moved, space="raw")
    np.testing.assert_allclose(after["a.R"], raw["a.R"] + 0.5, rtol=1e-12)
    np.testing.assert_allclose(after["b.R"], raw["b.R"], rtol=1e-12)
    # Their values in the prior's space are `L z + mu`, so both move.
    before, now = prf.values(model), prf.values(moved)
    assert not np.allclose(before["a.R"], now["a.R"]) and not np.allclose(before["b.R"], now["b.R"])
    # A value selector writes the same raw value to every parameter it selects.
    both = prf.values(prf.update(model, "[ab].R", value=0.25, space="raw"), space="raw")
    np.testing.assert_allclose([both[name] for name in NAMES], [0.25, 0.25], rtol=1e-12)


def _raw_log_prior_matches_declared_through_the_jacobian(model, names):
    """Checks raw = declared + log|det dx/dz| for the parameters `names` under a joint
    prior, with the Jacobian of their declared values `x` in their raw values `z` taken
    by `jax.jacobian`. Other parameters carry their own raw-to-declared Jacobian."""
    raw = prf.values(model, space="raw")
    z0 = jnp.stack([raw[name] for name in names]) + 0.1

    def at(z):
        return prf.update(model, dict(zip(names, z)), space="raw")

    def declared(z):
        values = prf.values(at(z))
        return jnp.stack([values[name] for name in names])

    moved = at(z0)
    expected = prf.log_prior(moved, space="declared") + jnp.linalg.slogdet(jax.jacobian(declared)(z0))[1]
    for name, p in prf.params(moved).items():
        if name not in names and p.raw_to_declared_bijector is not None:
            expected += p.raw_to_declared_bijector.forward_log_det_jacobian(p.raw_value)
    np.testing.assert_allclose(prf.log_prior(moved, space="raw"), expected, rtol=1e-9)


def test_raw_log_prior_carries_the_jacobian_of_the_whitening():
    _raw_log_prior_matches_declared_through_the_jacobian(_example(), NAMES)


@pytest.mark.parametrize("space", ["declared", "physical"])
def test_raw_log_prior_carries_the_jacobian_in_every_prior_space(space):
    parts = {
        "a": Resistor(R=prf.Unconstrained(50.0), name="a"),
        "c": Capacitor(C=prf.Unconstrained(2.0, scale=1e-12), name="c"),
    }
    mean = [49.0, 2.2] if space == "declared" else [49.0, 2.2e-12]
    tril = [[2.0, 0.0], [0.1, 0.5]] if space == "declared" else [[2.0, 0.0], [0.1e-12, 0.5e-12]]
    model = prf.prior(parts, ["a.R", "c.C"], _gaussian(mean, tril), space=space)
    _raw_log_prior_matches_declared_through_the_jacobian(model, ("a.R", "c.C"))


def test_raw_log_prior_is_the_base_density_of_a_flow():
    """For a flow, the raw log prior of its parameters is its base density at `z`."""
    parts, model = _parts(), _example()
    z = jnp.stack([prf.values(model, space="raw")[name] for name in NAMES])
    base = dd.Independent(dd.Normal(jnp.zeros(2), jnp.ones(2)))
    expected = base.log_prob(z) + prf.log_prior({"c": parts["c"]}, space="raw")
    np.testing.assert_allclose(prf.log_prior(model, space="raw"), expected, rtol=1e-10)
    jitted = eqx.filter_jit(lambda m: prf.log_prior(m, space="raw"))(model)
    np.testing.assert_allclose(jitted, expected, rtol=1e-10)


def test_a_multivariate_normal_is_whitened_by_its_cholesky_factor():
    parts = prf.update(_parts(), {"a.R": 52.0, "b.R": 49.0})
    model = prf.prior(parts, NAMES, dd.MultivariateNormalTri(MU, L), space="raw")
    raw = prf.values(model, space="raw")
    np.testing.assert_allclose([raw[name] for name in NAMES], _whitened(_in_prior_space(parts)), rtol=1e-10)
    _raw_log_prior_matches_declared_through_the_jacobian(model, NAMES)


def test_a_distribution_with_no_known_whitening_keeps_its_own_space():
    """An independent normal has no registered whitening, so raw stays the space it is
    over, and everything else still works."""
    parts = prf.update(_parts(), {"a.R": 52.0, "b.R": 49.0})
    distribution = dd.Independent(dd.Normal(MU, jnp.array([0.1, 0.2])))
    model = prf.prior(parts, NAMES, distribution, space="raw")
    before, after = prf.values(parts, space="raw"), prf.values(model, space="raw")
    assert all(np.allclose(before[name], after[name]) for name in before)
    again = prf.update(model, after, space="raw")
    assert all(np.allclose(prf.values(again)[n], prf.values(model)[n]) for n in after)
    _raw_log_prior_matches_declared_through_the_jacobian(model, NAMES)


def test_raw_values_of_a_batched_model_are_whitened_per_sample():
    model = _example()
    z = jnp.array([[0.1, -0.2], [0.3, 0.4], [-1.0, 0.5]])
    batched = prf.update(model, {"a.R": z[:, 0], "b.R": z[:, 1]}, space="raw")
    assert np.shape(prf.values(batched)["a.R"]) == (3,)
    raw = prf.values(batched, space="raw")
    np.testing.assert_allclose(np.stack([raw[name] for name in NAMES], axis=-1), z, rtol=1e-10, atol=1e-12)


# ---- A constant whitening log-determinant (#260) --------------------------------------


def _mvn_example(space):
    """A multivariate normal joint prior over `a.R` and `c.C` in `space`, with the
    Cholesky factor of its covariance. `c.C` is scaled, so the spaces differ, and over
    raw space `a.R` has a range, so its raw-to-declared map is not the identity."""
    # A joint prior over declared or physical space must fit inside `a.R`'s bounds.
    a = prf.Random(RTNormal(50.0, 0.1)) if space == "raw" else prf.Unconstrained(50.0)
    parts = {
        "a": Resistor(R=a, name="a"),
        "c": Capacitor(C=prf.Unconstrained(2.0, scale=1e-12), name="c"),
    }
    mean, tril = {
        "raw": ([3.9, 2.0], [[0.10, 0.0], [0.08, 0.06]]),
        "declared": ([49.0, 2.2], [[2.0, 0.0], [0.1, 0.5]]),
        "physical": ([49.0, 2.2e-12], [[2.0, 0.0], [0.1e-12, 0.5e-12]]),
    }[space]
    tril = jnp.asarray(tril)
    return prf.prior(parts, ["a.R", "c.C"], dd.MultivariateNormalTri(jnp.asarray(mean), tril), space=space), tril


def _flow_example():
    """A joint prior over raw space whose whitening, a softplus after an affine map, has
    a Jacobian that varies with the whitened values."""
    n = 2
    base = dd.Independent(dd.Normal(jnp.zeros(n), jnp.ones(n)))
    bijector = db.Chain([db.Block(db.Softplus(), 1), db.Block(db.Shift(MU), 1), db.TriangularLinear(L)])
    return prf.prior(_parts(), NAMES, dd.Transformed(base, bijector), space="raw")


def _dense_raw_log_prior(model, names):
    """Returns the raw log prior of `model` as a function of the raw values `z` of the
    parameters `names` under its joint prior: the declared log prior plus the
    log-determinant of a dense Jacobian of their declared values in `z`, plus the
    raw-to-declared Jacobians of the other parameters."""

    def at(z):
        return prf.update(model, dict(zip(names, z)), space="raw")

    def declared(z):
        values = prf.values(at(z))
        return jnp.stack([values[name] for name in names])

    def log_prior(z):
        moved = at(z)
        total = prf.log_prior(moved, space="declared") + jnp.linalg.slogdet(jax.jacfwd(declared)(z))[1]
        for name, p in prf.params(moved).items():
            if name not in names and p.raw_to_declared_bijector is not None:
                total += p.raw_to_declared_bijector.forward_log_det_jacobian(p.raw_value)
        return total

    return at, log_prior


@pytest.mark.parametrize("space", ["raw", "declared", "physical"])
def test_a_multivariate_normal_holds_its_whitening_log_det(space):
    """Its whitening `L z + mu` has the constant log-determinant `sum log L_ii`."""
    model, tril = _mvn_example(space)
    np.testing.assert_allclose(
        prx.as_unwrapped(model.whitening_log_det), jnp.sum(jnp.log(jnp.diag(tril))), rtol=1e-12
    )


@pytest.mark.parametrize("make", [
    lambda: dd.MultivariateNormalDiag(MU, jnp.diag(L)),
    lambda: dd.MultivariateNormalFullCovariance(MU, L @ L.T),
], ids=["diag", "full_covariance"])
def test_every_multivariate_normal_holds_its_whitening_log_det(make):
    model = prf.prior(_parts(), NAMES, make(), space="raw")
    np.testing.assert_allclose(prx.as_unwrapped(model.whitening_log_det), jnp.sum(jnp.log(jnp.diag(L))), rtol=1e-12)


def test_a_flow_holds_no_whitening_log_det():
    assert prx.as_unwrapped(_flow_example().whitening_log_det) is None


@pytest.mark.parametrize("case", ["raw", "declared", "physical", "flow"])
def test_raw_log_prior_and_its_gradient_match_a_dense_jacobian(case):
    if case == "flow":
        model = prf.update(
            _flow_example(), dict(zip(NAMES, jnp.array([0.1, -0.2]))), space="raw"
        )
        names = NAMES
        round_trip = prf.update(model, prf.values(model, space="raw"), space="raw")
        for name, value in prf.values(model).items():
            np.testing.assert_allclose(prf.values(round_trip)[name], value)
    else:
        model, names = _mvn_example(case)[0], ("a.R", "c.C")
    at, dense = _dense_raw_log_prior(model, names)
    raw = prf.values(model, space="raw")
    z = jnp.stack([raw[name] for name in names]) + jnp.array([0.1, -0.05])
    held = lambda z: prf.log_prior(at(z), space="raw")
    np.testing.assert_allclose(held(z), dense(z), rtol=1e-12)
    np.testing.assert_allclose(jax.grad(held)(z), jax.grad(dense)(z), rtol=1e-12, atol=1e-12)
    jitted = eqx.filter_jit(lambda m: prf.log_prior(m, space="raw"))(at(z))
    np.testing.assert_allclose(jitted, dense(z), rtol=1e-12)


# ---- Names ----------------------------------------------------------------------------


def test_names_are_kept():
    parts, model = _parts(), _example()
    assert list(prf.params(model)) == list(prf.params(parts)) == ["a.R", "b.R", "c.C"]
    for space in ("raw", "declared", "physical"):
        before, after = prf.values(parts, space=space), prf.values(model, space=space)
        assert list(before) == list(after)
        # Raw space is redefined for the parameters under the joint prior only.
        kept = ("c.C",) if space == "raw" else tuple(before)
        assert all(np.allclose(before[name], after[name]) for name in kept)


def test_names_are_kept_for_a_named_module_at_the_root():
    """A root module's own name is not part of its parameters' names, and wrapping it
    in a joint prior does not make it so."""
    root = Resistor(R=prf.Unconstrained(50.0), name="load") ** Resistor(R=prf.Unconstrained(20.0), name="b")
    root = prf.replace(root, name="chain")
    model = prf.prior(root, ["load.R", "b.R"], _gaussian([50.0, 20.0], [[1.0, 0.0], [0.0, 1.0]]))
    assert list(prf.params(model)) == list(prf.params(root)) == ["load.R", "b.R"]
    single = Resistor(R=prf.Unconstrained(50.0), name="load")
    joint = prf.prior(single, ["R"], dd.MultivariateNormalDiag(jnp.array([50.0]), jnp.array([1.0])))
    assert list(prf.params(joint)) == list(prf.params(single)) == ["R"]
    assert np.allclose(prf.log_prior(joint), Normal(50.0, 1.0).log_prob(50.0))


def test_update_by_name_writes_through_the_joint_prior():
    model = _example()
    moved = prf.update(model, {"a.R": 55.0})
    assert isinstance(moved, Probabilistic)
    assert list(prf.params(moved)) == ["a.R", "b.R", "c.C"]
    assert np.allclose(prf.values(moved)["a.R"], 55.0)
    raw = prf.values(model, space="raw")
    again = prf.update(model, raw, space="raw")
    assert list(prf.values(again, space="raw")) == list(raw)


def test_joint_prior_across_siblings_leaves_their_parent_unwrapped():
    model = _example()
    assert isinstance(model, Probabilistic)
    assert isinstance(model.module, dict) and isinstance(model.module["a"], Resistor)


def test_a_model_keeps_its_rf_interface():
    frequency = prf.Frequency(1.0, 2.0, 3, unit="GHz")
    cascade = Resistor(R=prf.Unconstrained(50.0), name="a") ** Resistor(R=prf.Unconstrained(20.0), name="b")
    model = prf.prior(cascade, NAMES, _gaussian([50.0, 20.0], [[1.0, 0.0], [0.0, 1.0]]))
    assert isinstance(model, Wrapped) and isinstance(model.wrapped, Probabilistic)
    assert np.allclose(model.s(frequency), cascade.s(frequency))
    assert list(prf.params(model)) == list(prf.params(cascade))


def test_resolve_keeps_the_joint_prior_and_unwrap_drops_it():
    model = prf.tie(_example(), "c.C", "a.R", fn=lambda r: r * 1e-14)
    resolved = prf.resolve(model)
    assert isinstance(resolved, Probabilistic)
    assert resolved.names == NAMES and prf.is_param(resolved.module["a"].R)
    assert np.allclose(resolved.module["c"].C, 50.0 * 1e-14)
    unwrapped = prf.unwrap(_example())
    assert isinstance(unwrapped, dict) and np.allclose(unwrapped["a"].R, 50.0)


# ---- Raising --------------------------------------------------------------------------


def test_a_parameter_under_two_joint_priors_raises():
    model = _example()
    with pytest.raises(ValueError, match=r"'b.R'.*already under a joint prior"):
        prf.prior(model, ["b.R", "c.C"], _gaussian(MU, L))


def test_a_scalar_prior_on_a_parameter_under_a_joint_prior_raises():
    with pytest.raises(ValueError, match=r"'a.R'.*already under a joint prior"):
        prf.prior(_example(), "a.R", Normal(50.0, 1.0))


def test_a_selected_name_that_is_not_a_free_parameter_raises():
    parts = _parts()
    fixed = prf.update(parts, "a.R", fixed=True)
    with pytest.raises(ValueError, match=r"'a.R'.*fixed or frozen"):
        prf.prior(fixed, NAMES, _gaussian(MU, L))
    frozen = prf.update(parts, "a.R", fn=prf.freeze)
    with pytest.raises(ValueError, match=r"'a.R'.*fixed or frozen"):
        prf.prior(frozen, NAMES, _gaussian(MU, L))
    with pytest.raises(ValueError, match="nope"):
        prf.prior(parts, ["a.R", "nope"], _gaussian(MU, L))


def test_fixing_a_parameter_under_a_joint_prior_raises():
    model = _example()
    with pytest.raises(ValueError, match=r"Cannot fix 'a.R'.*joint prior"):
        prf.update(model, "a.*", fixed=True)
    assert prf.params(prf.update(model, "c.C", fixed=True))["c.C"].fixed
    assert not prf.params(prf.update(model, "a.R", fixed=False))["a.R"].fixed


def test_a_structural_update_cannot_fix_or_drop_a_parameter_under_a_joint_prior():
    model = _example()
    with pytest.raises(ValueError, match=r"Cannot update: 'a.R'.*joint prior"):
        prf.update(model, {"a": Resistor(R=prf.Fixed(50.0), name="a")})
    with pytest.raises(ValueError, match=r"Cannot update: 'a.R'.*joint prior"):
        prf.update(model, "a.R", fn=prf.freeze)
    replaced = prf.update(model, {"a": Resistor(R=prf.Unconstrained(48.0), name="a")})
    assert np.allclose(prf.values(replaced)["a.R"], 48.0)


def test_tying_a_parameter_under_a_joint_prior_raises():
    with pytest.raises(ValueError, match=r"Cannot tie: 'a.R'.*joint prior"):
        prf.tie(_example(), "a.R", "c.C")
    # Tying from it is fine: only the distribution may set it.
    assert list(prf.params(prf.tie(_example(), "c.C", "a.R"))) == ["a.R", "b.R"]


def test_a_joint_prior_on_a_tied_parameter_raises():
    """A tie's target is derived, so it is no longer a parameter to put a prior on."""
    tied = prf.tie(_parts(), "b.R", "c.C")
    with pytest.raises(ValueError, match=r"Unknown parameter name: 'b.R'"):
        prf.prior(tied, NAMES, _gaussian(MU, L))


# ---- Solvers --------------------------------------------------------------------------


class _RandomWalkMetropolis(infer_base.AbstractJointSampler):
    """A plain random-walk Metropolis sampler, a joint sampler with no dependencies."""

    steps: int = 40_000
    burn: int = 4_000
    step_size: float = 0.07

    def run(self, logposterior_fn, y0, args, key, init_samples=None, max_steps=None, **kwargs):
        flat, unravel = ravel_pytree(y0)
        log_p = lambda x: logposterior_fn(unravel(x), args)

        def step(carry, k):
            x, lp = carry
            k_move, k_accept = jax.random.split(k)
            proposal = x + self.step_size * jax.random.normal(k_move, x.shape)
            lp_proposal = log_p(proposal)
            accept = jnp.log(jax.random.uniform(k_accept)) < lp_proposal - lp
            x, lp = jnp.where(accept, proposal, x), jnp.where(accept, lp_proposal, lp)
            return (x, lp), (x, lp)

        _, (xs, lps) = jax.lax.scan(step, (flat, log_p(flat)), jax.random.split(key, self.steps))
        xs, lps = xs[self.burn:], lps[self.burn:]
        return infer_base.SampleResult(samples=jax.vmap(unravel)(xs), fn_values=lps)


def test_mcmc_recovers_the_joint_prior():
    """With a flat likelihood the posterior is the prior, so in its space `a.R`
    and `b.R` have mean `MU` and covariance `L L^T` (standard deviations 0.1, correlation
    0.8), as before raw space was whitened. The sampler now moves in the whitened space,
    where they are independent standard normals, so its step is ten times that of the
    prior's space, whose unit is 0.1. The tolerances allow for the Monte Carlo error of 36 000
    correlated steps."""
    parts, model = _parts(), _example()
    batched, results = infer_base.run_sampler(
        lambda m, a: 0.0, model, _RandomWalkMetropolis(step_size=0.7), jax.random.key(0)
    )
    values = np.asarray(_in_prior_space(parts, batched))
    np.testing.assert_allclose(values.mean(axis=0), MU, atol=0.02)
    np.testing.assert_allclose(np.cov(values.T), L @ L.T, atol=0.002)
    assert np.corrcoef(values.T)[0, 1] == pytest.approx(0.8, abs=0.05)
    raw = np.stack([np.asarray(results.samples[name]) for name in NAMES], axis=-1)
    np.testing.assert_allclose(raw.mean(axis=0), [0.0, 0.0], atol=0.2)
    np.testing.assert_allclose(np.cov(raw.T), np.eye(2), atol=0.2)


def test_map_fit_is_unchanged_by_the_whitening():
    """A MAP fit maximises the declared log posterior, which does not depend on the raw
    space a minimiser moves in. It matches the fit made by hand in the prior's space, the
    raw space before this prior whitened it."""
    from scipy.optimize import minimize

    from pmrf.optimize import ScipyMinimize, base as optimize_base

    target, sigma = jnp.array([64.9, 64.8]), 0.05
    parts = prf.update(_parts(), "c.C", fixed=True)
    model = prf.prior(parts, NAMES, _gaussian(MU, L), space="raw")
    distributions = tree_param_distributions(model)

    def loss(m, args):
        x = jnp.stack([m["a"].R, m["b"].R])
        return 0.5 * jnp.sum(((x - target) / sigma) ** 2) - tree_param_log_prob(distributions, m)

    fitted, _ = optimize_base.run_minimizer(
        loss, model, ScipyMinimize(method="trust-constr"), max_iter=1000
    )
    fit = np.array([prf.values(fitted)[name] for name in NAMES])

    own = prf.params(parts)
    to_declared = [own[name].raw_to_declared_bijector for name in NAMES]

    def by_hand(t):
        x = jnp.stack([f.forward(t[i]) for i, f in enumerate(to_declared)])
        log_det = sum(f.forward_log_det_jacobian(t[i]) for i, f in enumerate(to_declared))
        log_declared = _gaussian(MU, L).log_prob(t) - log_det
        return 0.5 * jnp.sum(((x - target) / sigma) ** 2) - log_declared

    t0 = _in_prior_space(parts)
    reference = minimize(jax.jit(by_hand), t0, jac=jax.jit(jax.grad(by_hand)), method="BFGS", tol=1e-12)
    expected = [to_declared[i].forward(reference.x[i]) for i in range(2)]
    np.testing.assert_allclose(fit, expected, rtol=1e-6)


# ---- Hypercube samplers (#195) --------------------------------------------------------


class _CubeDraws(infer_base.AbstractHypercubeSampler):
    """Pushes uniform cube draws through the prior transform, as a hypercube sampler with
    a flat likelihood would. The draws can be pinned to fixed cube points instead."""

    n: int = 20_000
    points: dict | None = None

    def run(self, loglikelihood_fn, prior_transform_fn, u0, args, key, init_cube_samples=None, max_steps=None, **kwargs):
        if self.points is not None:
            cubes = self.points
        else:
            keys = jax.random.split(key, len(u0))
            cubes = {name: jax.random.uniform(k, (self.n,)) for k, name in zip(keys, u0)}
        samples = jax.vmap(lambda u: prior_transform_fn(u, args))(cubes)
        return infer_base.SampleResult(samples=samples, fn_values=jnp.zeros(len(next(iter(cubes.values())))))


class _RoundTrip(infer_base.AbstractHypercubeSampler):
    """Returns the prior transform of the starting cube point and of the starting cube samples."""

    def run(self, loglikelihood_fn, prior_transform_fn, u0, args, key, init_cube_samples=None, max_steps=None, **kwargs):
        start = jax.tree.map(lambda u: u[None], prior_transform_fn(u0, args))
        init = jax.vmap(lambda u: prior_transform_fn(u, args))(init_cube_samples)
        samples = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), start, init)
        return infer_base.SampleResult(samples=samples, fn_values=jnp.zeros(len(samples[NAMES[0]])))


def _cube_draws(model, sampler=None):
    batched, _ = infer_base.run_sampler(lambda m, a: 0.0, model, sampler or _CubeDraws(), jax.random.key(0))
    return batched


def test_cube_draws_follow_a_joint_prior_in_raw_space():
    """Uniform cube draws pushed through the prior transform have mean `MU` and covariance
    `L L^T` in the prior's space. The tolerances allow for the Monte Carlo error of 20 000
    independent draws, whose standard deviations are 0.1."""
    parts, model = _parts(), _example()
    batched = _cube_draws(model)
    values = np.asarray(_in_prior_space(parts, batched))
    np.testing.assert_allclose(values.mean(axis=0), MU, atol=0.005)
    np.testing.assert_allclose(np.cov(values.T), L @ L.T, atol=0.0005)
    assert np.corrcoef(values.T)[0, 1] == pytest.approx(0.8, abs=0.01)


def test_a_parameter_outside_the_joint_prior_follows_its_own_prior():
    points = {"a.R": jnp.full(3, 0.5), "b.R": jnp.full(3, 0.5), "c.C": jnp.array([0.1, 0.5, 0.9])}
    batched = _cube_draws(_example(), _CubeDraws(points=points))
    own = prx.as_unwrapped(prf.params(_parts())["c.C"].distribution)
    np.testing.assert_allclose(prf.values(batched)["c.C"], own.icdf(points["c.C"]), rtol=1e-5)


def test_cube_draws_land_inside_every_parameters_bounds():
    parts = _parts()
    extreme = jnp.array([0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0])
    points = {name: extreme for name in (*NAMES, "c.C")}
    for batched in (_cube_draws(_example()), _cube_draws(_example(), _CubeDraws(points=points))):
        values = prf.values(batched)
        for name, node in prf.params(parts).items():
            lower, upper = node.bounds
            assert np.all(np.isfinite(values[name]))
            assert np.all((values[name] >= lower) & (values[name] <= upper)), name


def test_a_starting_model_maps_to_the_cube_and_back():
    model = prf.update(_example(), {"a.R": 0.3, "b.R": -0.4}, space="raw")
    model = prf.update(model, {"c.C": 1.03})
    init = prf.update(model, {"a.R": jnp.array([-1.2, 0.5]), "b.R": jnp.array([0.1, 1.7])}, space="raw")
    init = prf.update(init, {"c.C": jnp.array([0.98, 1.05])})
    batched, _ = infer_base.run_sampler(lambda m, a: 0.0, model, _RoundTrip(), jax.random.key(0), init_samples=init)
    values, start, samples = prf.values(batched), prf.values(model), prf.values(init)
    for name in (*NAMES, "c.C"):
        np.testing.assert_allclose(values[name][0], start[name], rtol=1e-6)
        np.testing.assert_allclose(values[name][1:], samples[name], rtol=1e-6)


def test_a_multivariate_normal_joint_prior_is_sampled_through_its_cholesky_factor():
    parts = _parts()
    batched = _cube_draws(prf.prior(parts, NAMES, dd.MultivariateNormalTri(MU, L), space="raw"))
    values = np.asarray(_in_prior_space(parts, batched))
    np.testing.assert_allclose(values.mean(axis=0), MU, atol=0.005)
    np.testing.assert_allclose(np.cov(values.T), L @ L.T, atol=0.0005)


def test_a_joint_prior_without_a_normal_base_raises():
    uniform = dd.Independent(dd.Uniform(MU - 0.2, MU + 0.2), 1)
    model = prf.prior(_parts(), NAMES, uniform, space="raw")
    with pytest.raises(ValueError, match=r"'a.R', 'b.R'.*bijector over an independent normal base"):
        _cube_draws(model)


def test_cube_draws_follow_a_coupling_flow():
    """The user's pipeline: a flow trained over the raw values of fit 1's free parameters,
    in sorted-name order, is fit 2's prior. Training moves the flow's diagonal-Gaussian
    base, so the base here is moved off the standard normal. The cube draws match draws
    from the flow itself. The tolerances allow for the Monte Carlo error of two sets of
    20 000 draws, whose standard deviations are about 0.1 to 0.2."""
    fleqx = pytest.importorskip("fleqx", reason="fleqx is not installed")
    parts = _parts()
    names = sorted(NAMES)
    flow = fleqx.coupling_flow(jax.random.key(1), dim=2, flow_layers=2, nn_width=8)
    flow = eqx.tree_at(
        lambda f: (f.distribution.distribution.loc, f.distribution.distribution.scale),
        flow, (jnp.array([0.3, -0.2]), jnp.array([0.5, 1.5])),
    )
    shift = db.Block(db.Shift(MU), 1)
    flow = dd.Transformed(flow.distribution, db.Chain([shift, db.Block(db.ScalarAffine(jnp.array(0.0), jnp.array(0.1)), 1), flow.bijector]))
    batched = _cube_draws(prf.prior(parts, names, flow, space="raw"))
    values = np.asarray(_in_prior_space(parts, batched))
    direct = np.asarray(jax.vmap(flow.sample)(jax.random.split(jax.random.key(2), 20_000)))
    np.testing.assert_allclose(values.mean(axis=0), direct.mean(axis=0), atol=0.01)
    np.testing.assert_allclose(np.cov(values.T), np.cov(direct.T), atol=0.002)


# ---- Array-valued parameters (#261) ---------------------------------------------------


V = jnp.array([[0.3, -0.2, 0.1], [0.4, 0.0, -0.5]])
ARRAY_NAMES = ["a.R", "v"]
FLAT_NAMES = ["a.R", *(f"v{i}" for i in range(V.size))]


def _array_parts(flat=False):
    """A scalar `a.R` and a scaled, bounded array `v` of shape (2, 3), or with `flat`,
    the same values as one scalar parameter `v0` ... `v5` per element, in C order. A
    width `w` like `v` stays outside the joint prior."""
    a = Resistor(R=prf.Bounded(40.0, 60.0, value=50.5), name="a")
    w = prf.Unconstrained(jnp.zeros(V.shape))
    if flat:
        return {"a": a, "w": w, **{f"v{i}": prf.Bounded(-5.0, 5.0, value=x, scale=1e-3) for i, x in enumerate(V.ravel())}}
    return {"a": a, "w": w, "v": prf.Bounded(-5.0, 5.0, value=V, scale=1e-3)}


def _array_distribution(kind, space):
    """A correlated Gaussian over the 7 values of `a.R` and `v` in `space`, centred near
    their values there, as a multivariate normal or as a flow."""
    parts = _array_parts()
    values = prf.values(parts, space=space)
    mean = jnp.concatenate([jnp.ravel(values["a.R"]), jnp.ravel(values["v"])]) + 0.01
    width = jnp.abs(mean) * 0.2 + (1e-4 if space == "physical" else 0.1)
    corr = jnp.eye(7) + 0.3 * jnp.tril(jnp.ones((7, 7)), -1)
    tril = width[:, None] * corr
    return dd.MultivariateNormalTri(mean, tril) if kind == "mvn" else _gaussian(mean, tril)


def _array_prior(kind, space, flat=False):
    return prf.prior(_array_parts(flat), FLAT_NAMES if flat else ARRAY_NAMES, _array_distribution(kind, space), space=space)


def _at_raw(model, z, flat=False):
    """`model` with the joint prior's whitened vector `z`, possibly batched, written in."""
    if flat:
        return prf.update(model, dict(zip(FLAT_NAMES, jnp.moveaxis(z, -1, 0))), space="raw")
    return prf.update(model, {"a.R": z[..., 0], "v": z[..., 1:].reshape(*z.shape[:-1], *V.shape)}, space="raw")


Z = jnp.array([0.2, -0.4, 0.3, 0.1, -0.6, 0.5, 0.05])


@pytest.mark.parametrize("kind", ["flow", "tri", "diag", "full"])
@pytest.mark.parametrize("prior_space", ["raw", "declared", "physical"])
def test_array_joint_prior_sums_multiple_batch_axes_and_differentiates(kind, prior_space):
    parts = {
        "a": prf.Random(Normal(0.0, 1.0), value=0.0, scale=2.0),
        "v": prf.Random(Normal(0.0, 1.0), value=jnp.zeros(V.shape), scale=3.0),
    }
    mean = np.linspace(-0.3, 0.3, 7)
    tril = np.diag(np.linspace(0.6, 1.2, 7))
    if kind != "diag":
        tril += 0.1 * np.tril(np.ones((7, 7)), -1)
    distribution = {
        "flow": lambda: _gaussian(mean, tril),
        "tri": lambda: dd.MultivariateNormalTri(jnp.asarray(mean), jnp.asarray(tril)),
        "diag": lambda: dd.MultivariateNormalDiag(jnp.asarray(mean), jnp.asarray(np.diag(tril))),
        "full": lambda: dd.MultivariateNormalFullCovariance(jnp.asarray(mean), jnp.asarray(tril @ tril.T)),
    }[kind]()
    model = prf.prior(parts, ["a", "v"], distribution, space=prior_space)
    z = jnp.arange(42, dtype=float).reshape(2, 3, 7) / 30 - 0.5

    def at(z):
        return prf.update(model, {"a": z[..., 0], "v": z[..., 1:].reshape(*z.shape[:-1], *V.shape)}, space="raw")

    batched = at(z)
    assert prf.values(batched)["v"].shape == (2, 3, *V.shape)
    t = mean + np.asarray(z).reshape(-1, 7) @ tril.T
    density = sum(_gaussian_log_prob(sample, mean, tril) for sample in t)
    scale = len(t) * (np.log(2) + V.size * np.log(3))
    physical = density if prior_space == "physical" else density - scale
    expected = {
        "physical": physical,
        "declared": physical + scale,
        "raw": np.sum(-0.5 * np.asarray(z) ** 2 - 0.5 * np.log(2 * np.pi)),
    }
    for space in expected:
        score = eqx.filter_jit(lambda m: prf.log_prior(m, space=space))(batched)
        assert score.shape == ()
        np.testing.assert_allclose(score, expected[space], rtol=1e-12)
    gradient = jax.jit(jax.grad(lambda z: prf.log_prior(at(z), space="raw")))(z)
    np.testing.assert_allclose(gradient, -z, rtol=1e-12, atol=1e-14)


def test_attaching_to_batched_parameters_uses_the_attachment_shapes_as_the_event():
    parts = {"a": prf.Unconstrained(0.0, scale=2.0), "b": prf.Unconstrained(0.0, scale=3.0)}
    parts = prf.update(parts, {"a": jnp.zeros(3), "b": jnp.zeros(3)})
    mean, tril = np.arange(6) / 10, 0.7 * np.eye(6)
    model = prf.prior(parts, ["a", "b"], _gaussian(mean, tril))
    z = jnp.arange(12, dtype=float).reshape(2, 6) / 10
    batched = prf.update(model, {"a": z[:, :3], "b": z[:, 3:]}, space="raw")
    density = sum(_gaussian_log_prob(sample, mean, tril) for sample in mean + np.asarray(z) @ tril.T)
    for space, expected in {
        "declared": density,
        "physical": density - 2 * 3 * np.log(2 * 3),
        "raw": np.sum(-0.5 * np.asarray(z) ** 2 - 0.5 * np.log(2 * np.pi)),
    }.items():
        actual = prf.log_prior(batched, space=space)
        assert actual.shape == ()
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_one_out_of_bounds_sample_makes_the_batched_joint_total_minus_infinity():
    model = _at_raw(_array_prior("mvn", "raw"), Z + jnp.array([[0.0], [0.3], [-0.2]]))
    distributions = tree_param_distributions(model)
    inside = prf.unwrap(model)
    assert tree_param_log_prob(distributions, inside).shape == ()
    assert np.isfinite(tree_param_log_prob(distributions, inside))
    outside = eqx.tree_at(lambda m: m["v"], inside, inside["v"].at[1, 0, 2].set(6e-3))
    assert tree_param_log_prob(distributions, outside) == -jnp.inf


@pytest.mark.parametrize("kind", ["mvn", "flow"])
@pytest.mark.parametrize("prior_space", ["raw", "declared", "physical"])
def test_an_array_parameter_scores_as_one_scalar_per_value(kind, prior_space):
    array = _at_raw(_array_prior(kind, prior_space), Z)
    flat = _at_raw(_array_prior(kind, prior_space, flat=True), Z, flat=True)
    declared = prf.values(array)
    assert declared["v"].shape == V.shape
    np.testing.assert_allclose(
        jnp.ravel(declared["v"]), [prf.values(flat)[f"v{i}"] for i in range(V.size)], rtol=1e-12
    )
    for space in ("raw", "declared", "physical"):
        np.testing.assert_allclose(prf.log_prior(array, space=space), prf.log_prior(flat, space=space), rtol=1e-12)
    jitted = eqx.filter_jit(lambda m: prf.log_prior(m, space="raw"))(array)
    np.testing.assert_allclose(jitted, prf.log_prior(flat, space="raw"), rtol=1e-12)


@pytest.mark.parametrize("prior_space", ["raw", "declared", "physical"])
def test_an_array_parameters_raw_value_is_its_slice_of_the_whitened_vector(prior_space):
    model = _at_raw(_array_prior("mvn", prior_space), Z)
    raw = prf.values(model, space="raw")
    assert raw["v"].shape == V.shape
    np.testing.assert_allclose(jnp.concatenate([jnp.ravel(raw["a.R"]), jnp.ravel(raw["v"])]), Z, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("prior_space", ["raw", "declared", "physical"])
@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
def test_array_values_round_trip_through_update_in_every_space(prior_space, batched):
    z = Z + jnp.array([[0.0], [0.3], [-0.2]]) if batched else Z
    model = _at_raw(_array_prior("mvn", prior_space), z)
    batch = (3,) if batched else ()
    assert prf.values(model)["v"].shape == batch + V.shape
    assert prf.values(model, space="raw")["v"].shape == batch + V.shape
    for space in ("raw", "declared", "physical"):
        again = prf.update(model, prf.values(model, space=space), space=space)
        for check in ("raw", "declared", "physical"):
            before, after = prf.values(model, space=check), prf.values(again, space=check)
            for name in before:
                np.testing.assert_allclose(after[name], before[name], rtol=1e-10, atol=1e-12, err_msg=f"{space} {check} {name}")


def test_a_hypercube_sampler_maps_the_cube_through_an_array_parameters_block():
    n = 5
    u = jax.random.uniform(jax.random.key(3), (n, 7)).at[0].set(1e-12).at[1].set(1.0 - 1e-12)
    points = {"a.R": u[:, 0], "v": u[:, 1:].reshape(n, *V.shape), "w": jnp.full((n, *V.shape), 0.5)}
    flat_points = {"a.R": u[:, 0], "w": points["w"], **{f"v{i}": u[:, 1 + i] for i in range(V.size)}}
    model = prf.update(_array_prior("mvn", "raw"), "w", fn=lambda w: prf.Random(Normal(0.0, 1.0), value=jnp.zeros(V.shape)))
    flat = prf.update(_array_prior("mvn", "raw", flat=True), "w", fn=lambda w: prf.Random(Normal(0.0, 1.0), value=jnp.zeros(V.shape)))
    values = prf.values(_cube_draws(model, _CubeDraws(points=points)))
    expected = prf.values(_cube_draws(flat, _CubeDraws(points=flat_points)))
    assert values["v"].shape == (n, *V.shape)
    np.testing.assert_allclose(values["a.R"], expected["a.R"], rtol=1e-10)
    np.testing.assert_allclose(
        values["v"].reshape(n, -1), jnp.stack([expected[f"v{i}"] for i in range(V.size)], axis=-1), rtol=1e-10
    )
    assert np.all(np.isfinite(values["v"])) and np.all((values["v"] >= -5.0) & (values["v"] <= 5.0))
    assert np.all((values["a.R"] >= 40.0) & (values["a.R"] <= 60.0))


def test_an_event_size_matching_no_selection_names_both_sizes():
    distribution = dd.MultivariateNormalDiag(jnp.zeros(5), jnp.ones(5))
    with pytest.raises(ValueError, match=r"event size 5.*total size 7"):
        prf.prior(_array_parts(), ARRAY_NAMES, distribution)


def test_a_support_outside_an_array_parameters_validity_raises_naming_it():
    """The support is checked element by element: one element of `a.R` reaching below
    zero is enough."""
    parts = {
        "x": prf.Unconstrained(0.0),
        "a": _PositiveResistor(R=prf.Unconstrained(jnp.array([50.0, 51.0, 52.0])), name="a"),
    }

    def box(lower):
        # A sigmoid over a standard-normal base, shifted: support (lower, lower + 1).
        squash = db.Chain([db.Block(db.Shift(jnp.asarray(lower)), 1), db.Block(db.Sigmoid(), 1)])
        return dd.Transformed(dd.Independent(dd.Normal(jnp.zeros(4), jnp.ones(4))), squash)

    with pytest.raises(ValueError, match=r"validity of 'a\.R'\. "):
        prf.prior(parts, ["x", "a.R"], box([-0.5, 50.0, -0.5, 52.0]))
    assert isinstance(prf.prior(parts, ["x", "a.R"], box([-0.5, 50.0, 51.0, 52.0])), Probabilistic)


def test_fixing_or_tying_an_array_parameter_under_a_joint_prior_raises():
    model = _array_prior("mvn", "raw")
    with pytest.raises(ValueError, match=r"Cannot fix 'v'.*joint prior"):
        prf.update(model, "v", fixed=True)
    with pytest.raises(ValueError, match=r"Cannot tie: 'v'.*joint prior"):
        prf.tie(model, "v", "w")
