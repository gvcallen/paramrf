import pytest
import jax.numpy as jnp

import pmrf as prf
from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.problem import Problem
from pmrf.terms import AbstractTerm, BoundEvaluator, NegativeLogPrior, as_terms
from pmrf.optimize.minimize import minimize
from pmrf.optimize.solvers.scipy import ScipyMinimize

# ---------------------------------------------------------
# Dummy Concrete Models for Testing
# ---------------------------------------------------------

class BandModel(Model):
    """A 1-port model whose response is flat at `val` across any frequency."""
    val: prf.Param = prf.param(default=1.0, as_free=True)

    def s(self, freq: Frequency) -> jnp.ndarray:
        return jnp.ones((freq.npoints, 1, 1), dtype=complex) * self.val

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def low_band():
    return Frequency(start=50.0, stop=130.0, npoints=9, unit='MHz')

@pytest.fixture
def high_band():
    return Frequency(start=10.0, stop=500.0, npoints=21, unit='MHz')

@pytest.fixture
def model():
    return BandModel(val=1.0)

# ---------------------------------------------------------
# `BoundEvaluator`, `NegativeLogPrior` and `Problem`
# ---------------------------------------------------------

def test_term_binds_its_own_frequency(model, low_band, high_band):
    """A term evaluates over the grid it was constructed with, not a shared one."""
    npoints = lambda m, f: jnp.asarray(float(f.npoints))

    assert BoundEvaluator(npoints, low_band)(model) == 9.0
    assert BoundEvaluator(npoints, high_band)(model) == 21.0

def test_term_weight_scales_contribution(model, low_band):
    ones = lambda m, f: jnp.asarray(2.0)

    assert BoundEvaluator(ones, low_band)(model) == 2.0
    assert BoundEvaluator(ones, low_band, weight=0.5)(model) == 1.0

def test_problem_sums_terms_across_grids(model, low_band, high_band):
    """The defining capability: one parameter set, terms on different grids."""
    npoints = lambda m, f: jnp.asarray(float(f.npoints))
    problem = Problem(model=model, terms=(BoundEvaluator(npoints, low_band), BoundEvaluator(npoints, high_band)))

    assert problem() == 30.0

def test_problem_frequency_free_term(model, low_band):
    """A pure parameter penalty is a term needing no frequency at all."""
    penalty = lambda m: jnp.sum(m.val ** 2)
    problem = Problem(model=model, terms=(BoundEvaluator(lambda m, f: jnp.asarray(1.0), low_band), penalty))

    assert problem() == 2.0

def test_problem_frequency_stays_out_of_the_parameter_set(model, low_band):
    """The bound grid must not be picked up as a free parameter."""
    import equinox as eqx
    import jax
    import parax as prx

    problem = Problem(model=model, terms=(BoundEvaluator(lambda m, f: jnp.sum(m.val), low_band),))
    dynamic, _ = eqx.partition(problem, prx.constraints.is_dynamic, is_leaf=prx.constraints.is_leaf)
    leaves = jax.tree.leaves(prx.unwrap(dynamic, only_if=prx.is_constrained))

    assert len(leaves) == 1

# ---------------------------------------------------------
# `as_terms`
# ---------------------------------------------------------

def test_as_terms_single_objective(low_band):
    terms = as_terms(lambda m, f: jnp.asarray(0.0), low_band)

    assert len(terms) == 1
    assert isinstance(terms[0], BoundEvaluator)

def test_as_terms_list_shares_frequency(low_band):
    terms = as_terms([lambda m, f: jnp.asarray(0.0), lambda m, f: jnp.asarray(0.0)], low_band)

    assert len(terms) == 2
    assert all(t.frequency is not None for t in terms)

def test_as_terms_pair_binds_its_own_frequency(low_band, high_band):
    objective = lambda m, f: jnp.asarray(0.0)
    terms = as_terms([(objective, low_band), (objective, high_band)])

    assert [prf.unwrap(t.frequency).npoints for t in terms] == [9, 21]

def test_as_terms_mixes_bound_and_shared(low_band, high_band):
    objective = lambda m, f: jnp.asarray(0.0)
    terms = as_terms([(objective, high_band), objective], low_band)

    assert [prf.unwrap(t.frequency).npoints for t in terms] == [21, 9]

def test_as_terms_accepts_a_plain_callable_as_a_term(low_band):
    """Subclassing AbstractTerm is a convenience, never a requirement."""
    penalty = lambda model: jnp.sum(model.val ** 2)
    terms = as_terms([penalty])

    assert len(terms) == 1
    assert not isinstance(terms[0], AbstractTerm)

def test_builtin_terms_share_the_abstract_base():
    assert issubclass(BoundEvaluator, AbstractTerm)
    assert issubclass(NegativeLogPrior, AbstractTerm)

def test_as_terms_passes_through_explicit_terms(low_band):
    term = BoundEvaluator(lambda m, f: jnp.asarray(0.0), low_band, weight=3.0)

    assert as_terms([term]) == (term,)

def test_as_terms_rejects_malformed_pair(low_band):
    with pytest.raises(TypeError, match="evaluator, frequency"):
        as_terms([(lambda m, f: 0.0, low_band, 'extra')], low_band)

# ---------------------------------------------------------
# `minimize` integration
# ---------------------------------------------------------

def test_minimize_multi_grid_shares_one_parameter_set(model, low_band, high_band):
    """Both goals pull `val` toward their own target on their own grid."""
    low_goal = lambda m, f: jnp.sum((m.s(f) - 3.0) ** 2).real
    high_goal = lambda m, f: jnp.sum((m.s(f) - 3.0) ** 2).real

    result = minimize([(low_goal, low_band), (high_goal, high_band)], model, solver=ScipyMinimize())

    assert jnp.allclose(result.model.val, 3.0, atol=1e-3)
    assert len(result.objective) == 2

def test_minimize_without_frequency_requires_bound_terms(model):
    with pytest.raises(TypeError):
        minimize(lambda m, f: jnp.asarray(0.0), model)

def test_minimize_weight_biases_the_solution(model, low_band, high_band):
    """A dominant weight pulls the shared parameter toward that term's target."""
    toward = lambda target: (lambda m, f: jnp.sum((m.s(f) - target) ** 2).real)

    result = minimize([
        prf.BoundEvaluator(toward(0.0), low_band, weight=1.0),
        prf.BoundEvaluator(toward(10.0), high_band, weight=1e-6),
    ], model, solver=ScipyMinimize())

    assert prf.unwrap(result.model.val) < 1.0

def test_optimize_result_objective_holds_every_term(model, low_band, high_band):
    objective = lambda m, f: jnp.sum(m.val ** 2)
    result = minimize([(objective, low_band), (objective, high_band)], model, solver=ScipyMinimize())

    assert len(result.objective) == 2
    assert all(isinstance(t, BoundEvaluator) for t in result.objective)

def test_optimize_result_objective_for_single_term(model, low_band):
    goal = prf.evaluators.Goal('s11_db', '<', -20)
    result = minimize(goal, model, low_band, solver=ScipyMinimize())

    assert len(result.objective) == 1
    assert isinstance(result.objective[0].evaluator, prf.evaluators.Goal)
