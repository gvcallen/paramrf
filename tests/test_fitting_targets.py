import pytest
import numpy as np
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
# Shared priors across datasets
# ---------------------------------------------------------

def test_negative_log_prior_penalizes_the_model(wide_band):
    from pmrf.terms import NegativeLogPrior
    from pmrf.distributions import Uniform
    from pmrf.parameters import Random
    from pmrf.utils.tree import log_prob

    model = CompositeModel(
        wide=SubModel(val=Random(Uniform(0.0, 10.0), value=3.0)),
        narrow=SubModel(val=Random(Uniform(0.0, 10.0), value=7.0)),
    )

    assert jnp.allclose(NegativeLogPrior()(model), -log_prob(model), atol=1e-6)

def test_negative_log_prior_takes_no_frequency(wide_band):
    """It is a term, not an evaluator: it is called with the model alone."""
    import inspect
    from pmrf.terms import NegativeLogPrior

    parameters = inspect.signature(NegativeLogPrior.__call__).parameters

    assert 'frequency' not in parameters
    assert list(parameters) == ['self', 'model', 'kwargs']

def test_multi_dataset_map_splits_the_prior_into_its_own_term(starting_model, heterogeneous):
    """Several datasets: a likelihood each, plus exactly one shared prior term."""
    from pmrf.fitting.minimize import fit_minimize
    from pmrf.evaluators import NegativeLogLikelihood, NegativeLogPosterior
    from pmrf.terms import NegativeLogPrior

    result = fit_minimize(
        starting_model, heterogeneous,
        solver=prf.optimize.ScipyMinimize(), inference='bayesian',
    )
    terms = [t.evaluator if hasattr(t, 'evaluator') else t for t in result.solution.objective]

    assert sum(isinstance(t, NegativeLogLikelihood) for t in terms) == 2
    assert sum(isinstance(t, NegativeLogPrior) for t in terms) == 1
    assert not any(isinstance(t, NegativeLogPosterior) for t in terms)

def test_single_dataset_map_keeps_the_posterior_whole(starting_model, wide_band):
    """One dataset needs no split, so the posterior evaluator is used as before."""
    from pmrf.fitting.minimize import fit_minimize
    from pmrf.evaluators import NegativeLogPosterior

    ntwk = skrf.Network(frequency=wide_band.to_skrf(), s=np.ones((21, 1, 1)) * 3.0, name='wide')
    result = fit_minimize(
        starting_model, NetworkCollection([ntwk]),
        solver=prf.optimize.ScipyMinimize(), inference='bayesian',
    )

    assert len(result.solution.objective) == 1
    assert isinstance(result.solution.objective[0].evaluator, NegativeLogPosterior)
