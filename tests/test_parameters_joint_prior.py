"""Joint priors attached by name with `prf.prior` (ADR-0005, #193)."""
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


def test_raw_joint_prior_is_scored_at_the_stacked_raw_values():
    parts, model = _parts(), _example()
    raw = prf.param_values(parts, space="raw")
    expected = _gaussian(MU, L).log_prob(jnp.stack([raw["a.R"], raw["b.R"]]))
    expected += prf.log_prior({"c": parts["c"]}, space="raw")
    assert np.allclose(prf.log_prior(model, space="raw"), expected, rtol=1e-10)
    assert np.allclose(eqx.filter_jit(lambda m: prf.log_prior(m, space="raw"))(model), expected, rtol=1e-10)


def test_raw_joint_prior_carries_the_jacobian_to_declared_space():
    """Declared = raw - log|f'(z)| for the parameters under the joint prior, the same
    change of variables as for a parameter's own prior."""
    parts, model = _parts(), _example()
    to_declared = {name: prf.params(parts)[name].raw_to_declared_bijector for name in NAMES}
    raw = prf.param_values(parts, space="raw")
    log_det = sum(to_declared[name].forward_log_det_jacobian(raw[name]) for name in NAMES)
    joint_raw = prf.log_prior(model, space="raw") - prf.log_prior({"c": parts["c"]}, space="raw")
    joint_declared = prf.log_prior(model, space="declared") - prf.log_prior({"c": parts["c"]}, space="declared")
    assert np.allclose(joint_declared, joint_raw - log_det, rtol=1e-10)


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
    parts = _parts()
    mean = MU + jnp.array([0.2, -0.3])
    raw = prf.param_values(parts, space="raw")
    joint = lambda m: prf.log_prior(m, space="raw") - prf.log_prior({"c": parts["c"]}, space="raw")
    for order in (("a.R", "b.R"), ("b.R", "a.R")):
        model = prf.prior(parts, list(order), _gaussian(mean, L), space="raw")
        assert model.names == order
        expected = _gaussian(mean, L).log_prob(jnp.stack([raw[name] for name in order]))
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


# ---- Names ----------------------------------------------------------------------------


def test_names_are_kept():
    parts, model = _parts(), _example()
    assert list(prf.params(model)) == list(prf.params(parts)) == ["a.R", "b.R", "c.C"]
    for space in ("raw", "declared", "physical"):
        before, after = prf.param_values(parts, space=space), prf.param_values(model, space=space)
        assert list(before) == list(after)
        assert all(np.allclose(before[name], after[name]) for name in before)


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
    """With a flat likelihood the posterior is the prior, so the raw samples of `a.R`
    and `b.R` have mean `MU` and covariance `L L^T` (standard deviations 0.1, correlation
    0.8). The tolerances allow for the Monte Carlo error of 36 000 correlated steps."""
    model = _example()
    _, results = infer_base.run_sampler(lambda m, a: 0.0, model, _RandomWalkMetropolis(), jax.random.key(0))
    samples = np.stack([np.asarray(results.samples[name]) for name in NAMES], axis=-1)
    np.testing.assert_allclose(samples.mean(axis=0), MU, atol=0.02)
    np.testing.assert_allclose(np.cov(samples.T), L @ L.T, atol=0.002)
    assert np.corrcoef(samples.T)[0, 1] == pytest.approx(0.8, abs=0.05)
