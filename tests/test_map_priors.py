# tests/test_map_priors.py

import numpy as np
import pytest
import jax.numpy as jnp

import pmrf as prf
from pmrf.models import Resistor, Capacitor
from pmrf.distributions import Normal, Uniform
from pmrf.problems import SummedTerms, PriorPenalized
from pmrf.parameters import tree_param_distributions, tree_param_log_prob
from pmrf.terms import as_terms
from pmrf.utils import unwrap
from tests._dependency_checks import requires_distreqx_transpose


DATA = 5.0 + np.random.default_rng(0).normal(0.0, 0.5, 32)
VARIANCE = 0.25


def analytic_map(y, variance, prior_mean, prior_sd):
    """MAP for y_i ~ N(theta, variance) with theta ~ N(prior_mean, prior_sd^2)."""
    precision_data, precision_prior = len(y) / variance, 1.0 / prior_sd ** 2
    return (precision_data * y.mean() + precision_prior * prior_mean) / \
        (precision_data + precision_prior)


def linear_features(model, frequency):
    """A prediction linear in the parameter, so the MAP has a closed form."""
    return jnp.broadcast_to(jnp.asarray(model.R), (len(frequency.f),))


def fit(model, **kwargs):
    frequency = prf.Frequency(1.0, 2.0, len(DATA), "GHz")
    return prf.fitting.fit_minimize(
        model, jnp.asarray(DATA), frequency, features=linear_features,
        inference="bayesian", noise=VARIANCE,
        solver=prf.optimize.ScipyMinimize(method="trust-constr"),
        max_iter=4000, **kwargs)


def fitted_value(result):
    return float(np.asarray(list(result.model.named_params().values())[0]).ravel()[0])


def log_prior(model):
    return tree_param_log_prob(unwrap(tree_param_distributions(model)), unwrap(model))


# ==========================================
# 1. The prior's weight against a closed form
# ==========================================

@pytest.mark.parametrize("n_points", [4, 32, 256])
@pytest.mark.parametrize("prior_sd", [0.3, 1.0])
@requires_distreqx_transpose
def test_map_matches_closed_form(n_points, prior_sd):
    y = 5.0 + np.random.default_rng(0).normal(0.0, 0.5, n_points)
    frequency = prf.Frequency(1.0, 2.0, n_points, "GHz")
    model = Resistor(prf.Random(Normal(1.0, prior_sd), value=3.0))

    result = prf.fitting.fit_minimize(
        model, jnp.asarray(y), frequency, features=linear_features,
        inference="bayesian", noise=VARIANCE,
        solver=prf.optimize.ScipyMinimize(method="trust-constr"), max_iter=4000)

    expected = analytic_map(y, VARIANCE, 1.0, prior_sd)
    assert fitted_value(result) == pytest.approx(expected, rel=1e-4)


# ==========================================
# 2. Invariances
# ==========================================

@requires_distreqx_transpose
def test_map_is_invariant_to_parameter_scale():
    """The physical answer must not depend on the units a parameter is stored in."""
    values = []
    for scale in (1.0, 1e-3, 1e-6):
        model = Resistor(prf.Random(Normal(1.0 / scale, 1.0 / scale),
                                    value=3.0 / scale, scale=scale))
        values.append(fitted_value(fit(model)))

    expected = analytic_map(DATA, VARIANCE, 1.0, 1.0)
    assert values == pytest.approx([expected] * 3, rel=1e-4)


@requires_distreqx_transpose
def test_bounded_and_unconstrained_paths_agree():
    """Both solver paths optimize the same objective and must reach the same optimum."""
    bounded = fitted_value(fit(Resistor(prf.Random(Normal(1.0, 1.0), value=3.0))))
    unbounded = fitted_value(
        fit(Resistor(prf.Random(Normal(1.0, 1.0), value=3.0)), use_bounds=False))

    assert bounded == pytest.approx(unbounded, rel=1e-4)
    assert bounded == pytest.approx(analytic_map(DATA, VARIANCE, 1.0, 1.0), rel=1e-4)


@requires_distreqx_transpose
def test_flat_prior_reproduces_the_mle():
    """A Uniform is flat, so inside its support the MAP is the MLE."""
    model = Resistor(prf.Random(Uniform(0.0, 20.0), value=3.0))
    assert fitted_value(fit(model)) == pytest.approx(DATA.mean(), rel=1e-3)


@requires_distreqx_transpose
def test_flat_prior_excluding_the_mle_lands_on_the_bound():
    model = Resistor(prf.Random(Uniform(0.0, 4.0), value=2.0))
    assert fitted_value(fit(model)) == pytest.approx(4.0, abs=1e-2)


# ==========================================
# 3. Array-valued parameters
# ==========================================

@pytest.mark.parametrize("shape", [(1,), (3,), (2, 2)])
def test_log_prior_is_scalar_for_array_parameters(shape):
    """
    An array parameter scores one density per element; they must be summed.

    Left unreduced the objective becomes a vector and `jax.grad` refuses it, so MAP
    fails for any parameter not of shape ().
    """
    model = Resistor(prf.Random(Normal(1.0, 1.0), value=jnp.full(shape, 3.0)))
    assert jnp.asarray(log_prior(model)).shape == ()


def test_array_parameter_prior_scales_with_element_count():
    scalar = Resistor(prf.Random(Normal(1.0, 1.0), value=jnp.array([3.0])))
    vector = Resistor(prf.Random(Normal(1.0, 1.0), value=jnp.full((3,), 3.0)))
    assert float(log_prior(vector)) == pytest.approx(3 * float(log_prior(scalar)), rel=1e-6)


def test_map_objective_is_scalar_for_array_parameters():
    frequency = prf.Frequency(1.0, 2.0, 8, "GHz")
    model = Resistor(prf.Random(Normal(1.0, 1.0), value=jnp.full((3,), 3.0)))
    terms = as_terms([(lambda m, f, **kw: jnp.sum(jnp.asarray(m.R) ** 2), frequency)])

    assert jnp.asarray(PriorPenalized(SummedTerms(model=model, terms=terms))()).shape == ()


@requires_distreqx_transpose
def test_array_parameter_can_be_fitted():
    model = Resistor(prf.Random(Normal(1.0, 1.0), value=jnp.full((3,), 3.0)))
    frequency = prf.Frequency(1.0, 2.0, len(DATA), "GHz")
    result = prf.fitting.fit_minimize(
        model, jnp.asarray(DATA), frequency,
        features=lambda m, f: jnp.broadcast_to(jnp.mean(jnp.asarray(m.R)), (len(f.f),)),
        inference="bayesian", noise=VARIANCE,
        solver=prf.optimize.ScipyMinimize(method="trust-constr"), max_iter=2000)

    assert np.all(np.isfinite(np.asarray(list(result.model.named_params().values())[0])))


# ==========================================
# 4. Structural cases
# ==========================================

def test_prior_is_independent_of_term_count():
    """The prior belongs to the parameters, not to the terms evaluating them."""
    model = Resistor(prf.Random(Normal(1.0, 1.0), value=3.0))
    frequency = prf.Frequency(1.0, 2.0, 8, "GHz")
    objective = lambda m, f, **kw: jnp.sum(jnp.abs(jnp.asarray(m.R)) ** 2)

    contributions = []
    for n_terms in (1, 2, 5):
        summed = SummedTerms(model=model, terms=as_terms([(objective, frequency)] * n_terms))
        contributions.append(float(PriorPenalized(summed)()) - float(summed()))

    assert contributions == pytest.approx([contributions[0]] * 3, rel=1e-9)


def test_double_penalisation_is_rejected():
    model = Resistor(prf.Random(Normal(1.0, 1.0), value=3.0))
    frequency = prf.Frequency(1.0, 2.0, 8, "GHz")
    terms = as_terms([(lambda m, f, **kw: jnp.sum(jnp.asarray(m.R) ** 2), frequency)])
    penalised = PriorPenalized(SummedTerms(model=model, terms=terms))

    with pytest.raises(ValueError, match="already prior-penalized"):
        PriorPenalized(penalised)


def test_fixed_parameter_contributes_no_prior():
    fixed_random = Resistor(prf.Random(Normal(1.0, 1.0), value=3.0)) ** \
        Capacitor(prf.Random(Normal(1.0, 1.0), value=2.0, fixed=True))
    plain_fixed = Resistor(prf.Random(Normal(1.0, 1.0), value=3.0)) ** \
        Capacitor(prf.Fixed(2.0))

    assert float(log_prior(fixed_random)) == pytest.approx(float(log_prior(plain_fixed)))


@requires_distreqx_transpose
def test_no_free_parameters_reports_why():
    """A tree with nothing to fit should say so, not surface an internal pytree error."""
    model = Resistor(prf.Random(Normal(1.0, 1.0), value=3.0, fixed=True))
    with pytest.raises(ValueError, match="no free parameters"):
        fit(model)
