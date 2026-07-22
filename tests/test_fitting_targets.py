import pytest
import numpy as np
import jax
import parax as prx
import jax.numpy as jnp

import pmrf as prf
from pmrf.models import Model
from pmrf.parameters import Bounded, Fixed, Param
from pmrf.frequency import Frequency
from pmrf.network_collection import NetworkCollection
from pmrf.fitting.routers import fit_joint
from pmrf.fitting.targets import resolve_datasets, union_frequency

skrf = pytest.importorskip("skrf")


class SubModel(Model):
    val: Param

    def s(self, freq: Frequency):
        return jnp.ones((freq.npoints, 1, 1), dtype=complex) * self.val


class CompositeModel(Model):
    wide: SubModel
    narrow: SubModel
    global_fixed: Param = Fixed(10.0)


@pytest.fixture
def starting_model():
    return CompositeModel(
        wide=SubModel(val=Bounded(0.0, 10.0, value=0.0)),
        narrow=SubModel(val=Bounded(0.0, 10.0, value=0.0)),
    )


@pytest.fixture
def wide_band():
    return Frequency(start=10.0, stop=500.0, npoints=21, unit='MHz')


@pytest.fixture
def narrow_band():
    return Frequency(start=50.0, stop=130.0, npoints=9, unit='MHz')


@pytest.fixture
def heterogeneous(wide_band, narrow_band):
    """Two networks whose bands only partially overlap."""
    wide = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)) * 3.0, name='wide')
    narrow = skrf.Network(frequency=narrow_band.to_skrf(), s=np.ones((9, 1, 1)) * 7.0, name='narrow')
    return NetworkCollection([wide, narrow])


# ---------------------------------------------------------
# `resolve_datasets`
# ---------------------------------------------------------

def test_resolve_keeps_each_network_on_its_native_grid(heterogeneous):
    datasets = resolve_datasets('s', heterogeneous)

    assert len(datasets) == 2
    assert [d.frequency.npoints for d in datasets] == [21, 9]
    assert [d.target.shape[0] for d in datasets] == [21, 9]

def test_resolve_prefixes_the_predictor_with_the_network_name(heterogeneous, starting_model, wide_band):
    datasets = resolve_datasets('s', heterogeneous)

    # The first dataset must predict from `wide`, whose value differs from `narrow`.
    model = prf.replace(starting_model, wide=SubModel(val=Fixed(4.0)))
    predicted = datasets[0].predictor(model, wide_band)

    assert jnp.allclose(predicted, 4.0)

def test_resolve_single_network_yields_one_dataset(wide_band):
    ntwk = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)) * 2.0, name='solo')
    datasets = resolve_datasets('s', ntwk)

    assert len(datasets) == 1
    assert datasets[0].frequency.npoints == 21

def test_resolve_raw_array_passes_through(wide_band):
    array = np.ones((21, 1, 1))
    datasets = resolve_datasets('s', array, wide_band)

    assert len(datasets) == 1
    assert datasets[0].target is array

def test_resolve_callable_features_are_not_split(heterogeneous, wide_band):
    """An opaque predictor governs its own grids, so the collection is left whole."""
    predictor = lambda model, frequency: jnp.zeros(1)
    datasets = resolve_datasets(predictor, heterogeneous, wide_band)

    assert len(datasets) == 1
    assert datasets[0].predictor is predictor

def test_resolve_rejects_duplicate_names(wide_band):
    a = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)), name='dup')
    collection = NetworkCollection([a])
    collection.networks.append(a)

    with pytest.raises(ValueError, match="unique"):
        resolve_datasets('s', collection)


# ---------------------------------------------------------
# `union_frequency`
# ---------------------------------------------------------

def test_union_spans_every_band(heterogeneous):
    union = union_frequency(resolve_datasets('s', heterogeneous))

    assert union.f.min() == pytest.approx(10e6)
    assert union.f.max() == pytest.approx(500e6)

def test_union_of_a_shared_grid_is_that_grid(wide_band):
    ntwk = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)), name='solo')
    union = union_frequency(resolve_datasets('s', ntwk))

    assert union.npoints == 21


# ---------------------------------------------------------
# `fit_joint` end to end
# ---------------------------------------------------------

def test_fit_joint_across_bands_recovers_both_targets(starting_model, heterogeneous):
    """The whole point: one solve, two grids, neither truncated to the overlap."""
    result = fit_joint(starting_model, heterogeneous, solver=prf.optimize.ScipyMinimize())

    assert jnp.allclose(result.model.wide.val.value, 3.0, atol=1e-3)
    assert jnp.allclose(result.model.narrow.val.value, 7.0, atol=1e-3)

def test_fit_joint_uses_every_measured_point(starting_model, heterogeneous):
    """Under the old intersection behaviour the wide band lost most of its points."""
    result = fit_joint(starting_model, heterogeneous, solver=prf.optimize.ScipyMinimize())
    observed = [len(term.evaluator.target) for term in result.solution.objective]

    assert sorted(observed) == [9, 21]

def test_fit_joint_reports_the_full_span(starting_model, heterogeneous):
    result = fit_joint(starting_model, heterogeneous, solver=prf.optimize.ScipyMinimize())

    assert result.frequency.f.max() == pytest.approx(500e6)


# ---------------------------------------------------------
# Priors
# ---------------------------------------------------------

def test_map_problem_penalizes_the_prior(wide_band):
    """PriorPenalized is SummedTerms plus the negative log prior of its parameters."""
    import distreqx.distributions as dist
    from pmrf.parameters import Random
    from pmrf.problems import SummedTerms, PriorPenalized
    from pmrf.terms import BoundEvaluator

    model = CompositeModel(
        wide=SubModel(val=Random(dist.Normal(jnp.array(3.0), jnp.array(1.0)), value=3.0)),
        narrow=SubModel(val=Fixed(7.0)),
    )
    term = BoundEvaluator(lambda m, f: jnp.asarray(0.0), wide_band)
    expected = -dist.Normal(jnp.array(3.0), jnp.array(1.0)).log_prob(jnp.array(3.0))

    assert jnp.allclose(SummedTerms(model=model, terms=(term,))(), 0.0, atol=1e-6)
    assert jnp.allclose(PriorPenalized(SummedTerms(model=model, terms=(term,)))(), expected, atol=1e-6)

def test_map_problem_covers_hyper_parameters_in_terms(wide_band):
    """A prior on a term's own hyper-parameter is counted alongside the model's."""
    import distreqx.distributions as dist
    from pmrf.parameters import Random, Param
    from pmrf.problems import SummedTerms, PriorPenalized
    from pmrf.terms import AbstractTerm

    class NoisyTerm(AbstractTerm):
        sigma: Param
        def __call__(self, model, **kwargs):
            return jnp.asarray(0.0)

    model = CompositeModel(
        wide=SubModel(val=Random(dist.Normal(jnp.array(3.0), jnp.array(1.0)), value=3.0)),
        narrow=SubModel(val=Fixed(7.0)),
    )
    sigma = Random(dist.Normal(jnp.array(0.1), jnp.array(0.05)), value=0.1)
    model_prior = -dist.Normal(jnp.array(3.0), jnp.array(1.0)).log_prob(jnp.array(3.0))
    noise_prior = -dist.Normal(jnp.array(0.1), jnp.array(0.05)).log_prob(jnp.array(0.1))

    with_hyper = PriorPenalized(SummedTerms(model=model, terms=(NoisyTerm(sigma=sigma),)))

    assert jnp.allclose(with_hyper(), model_prior + noise_prior, atol=1e-5)

def test_map_problem_prior_stays_out_of_the_parameter_set(wide_band):
    """The bound distributions must not be seen as free parameters."""
    import distreqx.distributions as dist
    import equinox as eqx, jax, parax as prx
    from pmrf.parameters import Random
    from pmrf.problems import SummedTerms, PriorPenalized
    from pmrf.terms import BoundEvaluator

    model = CompositeModel(
        wide=SubModel(val=Random(dist.Normal(jnp.array(3.0), jnp.array(1.0)), value=3.0)),
        narrow=SubModel(val=Fixed(7.0)),
    )
    term = BoundEvaluator(lambda m, f: jnp.asarray(0.0), wide_band)
    count = lambda p: len(jax.tree.leaves(prx.unwrap(
        eqx.partition(p, prx.constraints.is_dynamic, is_leaf=prx.constraints.is_leaf)[0],
        only_if=prx.is_constrained)))

    assert count(PriorPenalized(SummedTerms(model=model, terms=(term,)))) == count(SummedTerms(model=model, terms=(term,)))

def test_fit_minimize_bayesian_applies_the_prior(starting_model, wide_band):
    """The regression: a tight prior must move the estimate away from the MLE."""
    import distreqx.distributions as dist
    from pmrf.parameters import Random
    from pmrf.fitting.minimize import fit_minimize

    ntwk = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)) * 5.0, name='wide')
    collection = NetworkCollection([ntwk])

    def fit(prior_std):
        model = CompositeModel(
            wide=SubModel(val=Random(dist.Normal(jnp.array(1.0), jnp.array(prior_std)), value=3.0)),
            narrow=SubModel(val=Fixed(7.0)),
        )
        result = fit_minimize(model, collection, solver=prf.optimize.ScipyMinimize(),
                              inference='bayesian', max_iter=400)
        return float(prf.unwrap(result.model.wide.val))

    flat, tight = fit(100.0), fit(0.01)

    assert jnp.allclose(flat, 5.0, atol=1e-2)      # data wins
    assert tight < 2.0                              # prior wins

def test_fit_minimize_frequentist_ignores_the_prior(starting_model, wide_band):
    import distreqx.distributions as dist
    from pmrf.parameters import Random
    from pmrf.fitting.minimize import fit_minimize

    ntwk = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)) * 5.0, name='wide')
    model = CompositeModel(
        wide=SubModel(val=Random(dist.Normal(jnp.array(1.0), jnp.array(0.01)), value=3.0)),
        narrow=SubModel(val=Fixed(7.0)),
    )
    result = fit_minimize(model, NetworkCollection([ntwk]),
                          solver=prf.optimize.ScipyMinimize(), max_iter=400)

    assert jnp.allclose(float(prf.unwrap(result.model.wide.val)), 5.0, atol=1e-2)


# ---------------------------------------------------------
# Correlated priors attached over a sub-tree
# ---------------------------------------------------------

def _correlated(a=3.0, b=7.0):
    """A model a joint prior can be attached across two of its parameters."""
    return CompositeModel(wide=SubModel(val=prf.Unconstrained(a)),
                          narrow=SubModel(val=prf.Unconstrained(b)))


def test_probabilistic_single_target_prior_is_found(wide_band):
    """A distribution attached after construction must be picked up."""
    import distreqx.distributions as dist
    from pmrf.models import Probabilistic
    from pmrf.parameters import tree_param_distributions, tree_param_log_prob

    base = _correlated()
    model = Probabilistic(model=base, distribution=prf.distributions.Normal(3.0, 1.0),
                          target=lambda m: m.wide.val)

    expected = float(dist.Normal(jnp.array(3.0), jnp.array(1.0)).log_prob(jnp.array(3.0)))
    scored = tree_param_log_prob(tree_param_distributions(model), prf.unwrap(model))

    assert jnp.allclose(scored, expected, atol=1e-6)

def test_correlated_joint_prior_over_a_subtree(wide_band):
    """A joint is scored over the whole sub-tree at once, preserving correlations."""
    import equinox as eqx
    from pmrf.models import Probabilistic
    from pmrf.distributions import Joint
    from pmrf.parameters import tree_param_distributions, tree_param_log_prob

    base = _correlated()
    import equinox as eqx
    sub = base.wide
    dist_tree = eqx.tree_at(lambda m: m.val, sub, prf.distributions.Normal(3.0, 1.0))
    joint = Joint(dist_tree)
    model = Probabilistic(model=base, distribution=joint, target=lambda m: m.wide)

    dists = tree_param_distributions(model)
    found = [d for d in jax.tree.leaves(dists, is_leaf=prx.is_distribution) if prx.is_distribution(d)]

    # One joint covering both parameters, not two independent marginals.
    assert len(found) == 1
    assert jnp.allclose(
        tree_param_log_prob(dists, prf.unwrap(model)),
        joint.log_prob(prf.unwrap(base.wide)),
        atol=1e-6,
    )

def test_map_problem_uses_a_correlated_prior(wide_band):
    """End to end: PriorPenalized must apply an attached joint, not ignore it."""
    import equinox as eqx
    from pmrf.models import Probabilistic
    from pmrf.distributions import Joint
    from pmrf.problems import SummedTerms, PriorPenalized
    from pmrf.terms import BoundEvaluator

    base = _correlated()
    import equinox as eqx
    dist_tree = eqx.tree_at(lambda m: m.val, base.wide, prf.distributions.Normal(3.0, 1.0))
    joint = Joint(dist_tree)
    model = Probabilistic(model=base, distribution=joint, target=lambda m: m.wide)
    term = BoundEvaluator(lambda m, f: jnp.asarray(0.0), wide_band)

    mle = SummedTerms(model=model, terms=(term,))
    mapp = PriorPenalized(SummedTerms(model=model, terms=(term,)))

    assert jnp.allclose(mle(), 0.0, atol=1e-6)
    assert jnp.allclose(mapp(), -joint.log_prob(prf.unwrap(base.wide)), atol=1e-6)
    assert not jnp.allclose(mapp(), mle(), atol=1e-6)

def test_correlated_prior_moves_a_fit(wide_band):
    """A tight joint prior must pull the fit away from the data's answer."""
    import equinox as eqx
    from pmrf.models import Probabilistic
    from pmrf.distributions import Joint
    from pmrf.fitting.minimize import fit_minimize

    ntwk = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)) * 10.0, name='wide')

    def fit(scale):
        base = _correlated(1.0, 1.0)
        import equinox as eqx
        dist_tree = eqx.tree_at(lambda m: m.val, base.wide, prf.distributions.Normal(1.0, scale))
        model = Probabilistic(model=base, distribution=Joint(dist_tree), target=lambda m: m.wide)
        result = fit_minimize(model, NetworkCollection([ntwk]),
                              solver=prf.optimize.ScipyMinimize(),
                              inference='bayesian', max_iter=400)
        m = prf.unwrap(result.model)
        return float(prf.unwrap(m.wide.val))

    assert jnp.allclose(fit(100.0), 10.0, atol=1e-1)   # data wins
    assert fit(0.01) < 5.0                             # prior wins


def test_map_prior_survives_pytree_round_trips(wide_band):
    """
    The captured distributions must not be rebuilt from an already-unwrapped tree.

    They are set in `__post_init__`, which JAX must not re-run when rebuilding the
    problem. If it did they would come back empty and the prior would silently stop
    applying, giving MLE results labelled as MAP.
    """
    import equinox as eqx
    import distreqx.distributions as dist
    from pmrf.parameters import Random
    from pmrf.problems import SummedTerms, PriorPenalized
    from pmrf.terms import BoundEvaluator

    model = CompositeModel(
        wide=SubModel(val=Random(dist.Normal(jnp.array(1.0), jnp.array(0.1)), value=3.0)),
        narrow=SubModel(val=Fixed(7.0)),
    )
    term = BoundEvaluator(lambda m, f: jnp.asarray(0.0), wide_band)
    problem = PriorPenalized(SummedTerms(model=model, terms=(term,)))
    contribution = lambda p: p() - p.inner()

    direct = contribution(problem)
    leaves, treedef = jax.tree_util.tree_flatten(problem)
    rebuilt = contribution(jax.tree_util.tree_unflatten(treedef, leaves))
    dynamic, static = eqx.partition(problem, eqx.is_inexact_array)
    recombined = contribution(eqx.combine(dynamic, static))

    # A zero prior would make the equalities pass while the feature is broken.
    assert not jnp.allclose(direct, 0.0)
    assert jnp.allclose(rebuilt, direct)
    assert jnp.allclose(recombined, direct)


def test_named_params_sees_past_a_probabilistic_wrapper():
    """Parameter traversal must not stop at a wrapper, hiding the parameters beyond it."""
    from pmrf.models import Probabilistic

    base = _correlated()
    wrapped = Probabilistic(model=base, distribution=prf.distributions.Normal(3.0, 1.0),
                            target=lambda m: m.wide.val)

    # wide.val is inside the wrapper and legitimately opaque; the rest must be found.
    assert len(wrapped.named_params()) == len(base.named_params()) - 1


def test_prior_is_finite_for_a_scaled_parameter(wide_band):
    """
    A prior is authored in the parameter's raw space, not its scaled one.

    Evaluating it against the scaled value puts it far outside its support, giving an
    infinite log prior and a MAP objective that never moves. Earlier MAP tests all
    used scale=1, the one case where this is invisible.
    """
    from pmrf.parameters import tree_param_distributions, tree_param_log_prob
    from pmrf.problems import SummedTerms, PriorPenalized
    from pmrf.terms import BoundEvaluator

    # Authored over mm, held in metres: the reported failure.
    scaled = prf.Random(prf.distributions.Uniform(0.0, 100.0), value=75.0, scale=1e-3)
    model = CompositeModel(wide=SubModel(val=scaled), narrow=SubModel(val=Fixed(7.0)))

    log_prior = tree_param_log_prob(tree_param_distributions(model), prf.unwrap(model))
    assert jnp.isfinite(log_prior)

    term = BoundEvaluator(lambda m, f: jnp.asarray(0.0), wide_band)
    assert jnp.isfinite(PriorPenalized(SummedTerms(model=model, terms=(term,)))())

def test_scaled_parameter_still_moves_under_map(wide_band):
    """An infinite prior leaves the optimizer at its starting point."""
    from pmrf.fitting.minimize import fit_minimize

    ntwk = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)) * 5e-3, name='wide')
    scaled = prf.Random(prf.distributions.Uniform(0.0, 100.0), value=75.0, scale=1e-3)
    model = CompositeModel(wide=SubModel(val=scaled), narrow=SubModel(val=Fixed(7.0)))

    result = fit_minimize(model, NetworkCollection([ntwk]),
                          solver=prf.optimize.ScipyMinimize(), inference='bayesian', max_iter=200)

    assert not jnp.allclose(prf.unwrap(result.model.wide.val), prf.unwrap(model.wide.val))


def test_probabilistic_folds_the_scale_of_its_target():
    """
    Attaching a distribution to a scaled parameter must not lose the scale.

    Parax keeps only the physical value and drops the parameter, so a distribution
    authored in the parameter's constrained space would be evaluated against a scaled
    value. For a non-uniform prior that is not a constant offset: it lands far into
    the tail, where the density is nearly flat and exerts almost no gradient.
    """
    from pmrf.models import Probabilistic, Resistor
    from pmrf.parameters import tree_param_distributions, tree_param_log_prob

    prior = lambda: prf.distributions.Normal(75.0, 5.0)
    log_prior = lambda m: tree_param_log_prob(tree_param_distributions(m), prf.unwrap(m))

    scored = {}
    for value in (75.0, 60.0):
        # Authored over mm, held in metres.
        plain = Resistor(prf.Random(prior(), value=value, scale=1e-3))
        wrapped = Probabilistic(model=plain, distribution=prior(), target=lambda m: m.R)
        scored[value] = (log_prior(plain), log_prior(wrapped))
        assert jnp.allclose(prf.unwrap(wrapped).R, value * 1e-3)

    for direct, via_wrapper in scored.values():
        assert jnp.allclose(direct, via_wrapper, atol=1e-4)

    # The prior must still discriminate: at the mean it is higher than 3 sigma away.
    assert scored[75.0][1] > scored[60.0][1] + 1.0
