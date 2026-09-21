"""Joint priors attached by name with `prf.prior` (ADR-0005, #193), and the whitened raw
space of their parameters (#194)."""
import distreqx.bijectors as db
import distreqx.distributions as dd
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import pmrf as prf
from pmrf.distributions import Normal
from pmrf.distributions import RelativeTruncatedNormal as RTNormal
from pmrf.infer import base as infer_base
from pmrf.models import Capacitor, Resistor, Wrapped
from pmrf.modules import Probabilistic
from pmrf.parameters import tree_param_distributions, tree_param_log_prob


MU = jnp.array([3.9, 3.9])
L = jnp.array([[0.10, 0.0], [0.08, 0.06]])   # correlation 0.8
NAMES = ("a.R", "b.R")


def _gaussian(mean, tril):
    """A correlated Gaussian as an affine bijector over a standard-normal base, the
    structure of a trained flow. `Block` sums the shift's log-determinant over the event."""
    n = len(mean)
    base = dd.Independent(dd.Normal(jnp.zeros(n), jnp.ones(n)), 1)
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


def _example():
    """The correlated-Gaussian example of #193: a joint prior over the raw values of two
    sibling sub-models' parameters, with `c.C` keeping its own prior."""
    return prf.prior(_parts(), NAMES, _gaussian(MU, L), space="raw")


# ---- Scoring --------------------------------------------------------------------------


def _in_prior_space(parts, model=None):
    """The values of `a.R` and `b.R` of `model`, by default `parts`, possibly batched, in
    the example joint prior's space: the raw space of `parts`, before it was attached."""
    own = prf.params(parts)
    declared = prf.param_values(parts if model is None else model)
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
    parts = _parts()
    mean, tril = [50.5, 49.0], [[2.0, 0.0], [1.6, 1.2]]
    model = prf.prior(parts, NAMES, _gaussian(mean, tril), space=space)
    values = prf.param_values(parts, space=space)
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


def test_vector_order_follows_explicit_names():
    parts = prf.update(_parts(), {"a.R": 52.0})
    mean = MU + jnp.array([0.2, -0.3])
    raw = prf.param_values(parts, space="raw")
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
    model = prf.prior(parts, NAMES, _gaussian([50.0, 50.0], [[20.0, 0.0], [0.0, 20.0]]))
    distributions = tree_param_distributions(model)
    inside = prf.unwrap(model)
    assert np.isfinite(tree_param_log_prob(distributions, inside))
    outside = eqx.tree_at(lambda m: m["a"].R, inside, jnp.asarray(70.0))
    assert tree_param_log_prob(distributions, outside) == -jnp.inf


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
    raw = prf.param_values(model, space="raw")
    np.testing.assert_allclose([raw[name] for name in NAMES], _whitened(_in_prior_space(parts)), rtol=1e-12)
    assert np.allclose(raw["c.C"], prf.param_values(parts, space="raw")["c.C"])
    # Reading raw values only is unchanged by `where`.
    assert np.allclose(prf.param_values(model, "b.R", space="raw")["b.R"], raw["b.R"])


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
    values = prf.param_values(parts, space=space)
    raw = prf.param_values(model, space="raw")
    expected = jnp.linalg.solve(tril, jnp.array([values[n] for n in names]) - mean)
    np.testing.assert_allclose([raw[n] for n in names], expected, rtol=1e-10)


def test_raw_round_trip_returns_the_model():
    model = _example()
    again = prf.update(model, prf.param_values(model, space="raw"), space="raw")
    assert jax.tree.structure(again) == jax.tree.structure(model)
    for x, y in zip(jax.tree.leaves(again), jax.tree.leaves(model)):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_a_raw_update_moves_one_whitened_coordinate():
    """Writing one raw value keeps the other whitened coordinates, so under a correlated
    prior every parameter under it can move."""
    model = _example()
    raw = prf.param_values(model, space="raw")
    moved = prf.update(model, {"a.R": raw["a.R"] + 0.5}, space="raw")
    after = prf.param_values(moved, space="raw")
    np.testing.assert_allclose(after["a.R"], raw["a.R"] + 0.5, rtol=1e-12)
    np.testing.assert_allclose(after["b.R"], raw["b.R"], rtol=1e-12)
    # Their values in the prior's space are `L z + mu`, so both move.
    before, now = prf.param_values(model), prf.param_values(moved)
    assert not np.allclose(before["a.R"], now["a.R"]) and not np.allclose(before["b.R"], now["b.R"])
    # A value selector writes the same raw value to every parameter it selects.
    both = prf.param_values(prf.update(model, "[ab].R", value=0.25, space="raw"), space="raw")
    np.testing.assert_allclose([both[name] for name in NAMES], [0.25, 0.25], rtol=1e-12)


def _raw_log_prior_matches_declared_through_the_jacobian(model, names):
    """Checks raw = declared + log|det dx/dz| for the parameters `names` under a joint
    prior, with the Jacobian of their declared values `x` in their raw values `z` taken
    by `jax.jacobian`. Other parameters carry their own raw-to-declared Jacobian."""
    raw = prf.param_values(model, space="raw")
    z0 = jnp.stack([raw[name] for name in names]) + 0.1

    def at(z):
        return prf.update(model, dict(zip(names, z)), space="raw")

    def declared(z):
        values = prf.param_values(at(z))
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
        "a": Resistor(R=prf.Bounded(30.0, 70.0, value=50.0), name="a"),
        "c": Capacitor(C=prf.Unconstrained(2.0, scale=1e-12), name="c"),
    }
    mean = [49.0, 2.2] if space == "declared" else [49.0, 2.2e-12]
    tril = [[2.0, 0.0], [0.1, 0.5]] if space == "declared" else [[2.0, 0.0], [0.1e-12, 0.5e-12]]
    model = prf.prior(parts, ["a.R", "c.C"], _gaussian(mean, tril), space=space)
    _raw_log_prior_matches_declared_through_the_jacobian(model, ("a.R", "c.C"))


def test_raw_log_prior_is_the_base_density_of_a_flow():
    """For a flow, the raw log prior of its parameters is its base density at `z`."""
    parts, model = _parts(), _example()
    z = jnp.stack([prf.param_values(model, space="raw")[name] for name in NAMES])
    base = dd.Independent(dd.Normal(jnp.zeros(2), jnp.ones(2)), 1)
    expected = base.log_prob(z) + prf.log_prior({"c": parts["c"]}, space="raw")
    np.testing.assert_allclose(prf.log_prior(model, space="raw"), expected, rtol=1e-10)
    jitted = eqx.filter_jit(lambda m: prf.log_prior(m, space="raw"))(model)
    np.testing.assert_allclose(jitted, expected, rtol=1e-10)


def test_a_multivariate_normal_is_whitened_by_its_cholesky_factor():
    parts = prf.update(_parts(), {"a.R": 52.0, "b.R": 49.0})
    model = prf.prior(parts, NAMES, dd.MultivariateNormalTri(MU, L), space="raw")
    raw = prf.param_values(model, space="raw")
    np.testing.assert_allclose([raw[name] for name in NAMES], _whitened(_in_prior_space(parts)), rtol=1e-10)
    _raw_log_prior_matches_declared_through_the_jacobian(model, NAMES)


def test_a_distribution_with_no_known_whitening_keeps_its_own_space():
    """An independent normal has no registered whitening, so raw stays the space it is
    over, and everything else still works."""
    parts = prf.update(_parts(), {"a.R": 52.0, "b.R": 49.0})
    distribution = dd.Independent(dd.Normal(MU, jnp.array([0.1, 0.2])), 1)
    model = prf.prior(parts, NAMES, distribution, space="raw")
    before, after = prf.param_values(parts, space="raw"), prf.param_values(model, space="raw")
    assert all(np.allclose(before[name], after[name]) for name in before)
    again = prf.update(model, after, space="raw")
    assert all(np.allclose(prf.param_values(again)[n], prf.param_values(model)[n]) for n in after)
    _raw_log_prior_matches_declared_through_the_jacobian(model, NAMES)


def test_raw_values_of_a_batched_model_are_whitened_per_sample():
    model = _example()
    z = jnp.array([[0.1, -0.2], [0.3, 0.4], [-1.0, 0.5]])
    batched = prf.update(model, {"a.R": z[:, 0], "b.R": z[:, 1]}, space="raw")
    assert np.shape(prf.param_values(batched)["a.R"]) == (3,)
    raw = prf.param_values(batched, space="raw")
    np.testing.assert_allclose(np.stack([raw[name] for name in NAMES], axis=-1), z, rtol=1e-10, atol=1e-12)


# ---- Names ----------------------------------------------------------------------------


def test_names_are_kept():
    parts, model = _parts(), _example()
    assert list(prf.params(model)) == list(prf.params(parts)) == ["a.R", "b.R", "c.C"]
    for space in ("raw", "declared", "physical"):
        before, after = prf.param_values(parts, space=space), prf.param_values(model, space=space)
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
    assert np.allclose(prf.param_values(moved)["a.R"], 55.0)
    raw = prf.param_values(model, space="raw")
    again = prf.update(model, raw, space="raw")
    assert list(prf.param_values(again, space="raw")) == list(raw)


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


def test_a_non_scalar_parameter_raises():
    parts = {"a": Resistor(R=prf.Unconstrained(jnp.array([1.0, 2.0])), name="a")}
    with pytest.raises(ValueError, match=r"'a.R'.*not"):
        prf.prior(parts, ["a.R"], dd.MultivariateNormalDiag(jnp.zeros(1), jnp.ones(1)))


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
    assert np.allclose(prf.param_values(replaced)["a.R"], 48.0)


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

    fitted, _ = optimize_base.run_minimizer(loss, model, ScipyMinimize(), max_iter=1000)
    fit = np.array([prf.param_values(fitted)[name] for name in NAMES])

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
