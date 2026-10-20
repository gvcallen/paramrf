"""`ScipyLeastSquares` minimises half the squared norm of the whitened residual over box
space (ADR-0009, #313)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.fitting import fit_minimize
from pmrf.frequency import Frequency
from pmrf.materials import BulkConductor, ConstantDielectric
from pmrf.models.components.lines.microstrip import MicrostripLine
from pmrf.objectives.evaluators import (
    Feature, GibbsMarginalLogLikelihood, MarginalLogLikelihood, Negated, TargetLoss, whitened_residual,
)
from pmrf.objectives.losses import MSELoss, RMSELoss
from pmrf.objectives.problems import PriorPenalized, SummedTerms
from pmrf.objectives.terms import BoundEvaluator, as_terms
from pmrf.optimize import ScipyLeastSquares, ScipyMinimize, minimize
from pmrf.optimize import base as optimize_base
from pmrf.optimize.solvers import scipy as scipy_solver
from pmrf.stats.likelihoods import GaussianLikelihood


class Pair(prf.Model):
    u: prf.Param = prf.param()
    v: prf.Param = prf.param()

    def s(self, freq):
        f = freq.f_scaled[:, None, None]
        return (self.u + 1j * self.v * f) * jnp.ones((freq.npoints, 1, 1))


@pytest.fixture
def freq():
    return Frequency(1.0, 3.0, 5, "GHz")


def _quiet():
    return ScipyLeastSquares(show_progress=False)


def _mse(target, predictor=None):
    return TargetLoss(predictor=predictor or Feature("s"), target=target, loss=MSELoss())


# ---- Whitened residual ----------------------------------------------------------------


def test_half_squared_norm_of_residual_is_the_mse_objective(freq):
    target = Pair(u=1.0, v=2.0).s(freq)
    real_target = jnp.real(target[:, 0, 0]) + 0.5
    model = Pair(u=prf.Bounded(0.0, 5.0, value=1.7), v=prf.Unconstrained(0.4))
    problem = SummedTerms(
        model=model,
        terms=(
            BoundEvaluator(_mse(target), freq),
            BoundEvaluator(_mse(real_target, lambda m, f: jnp.real(m.s(f)[:, 0, 0])), freq, weight=2.5),
        ),
    )

    r = whitened_residual(problem)

    assert r.ndim == 1 and not jnp.iscomplexobj(r)
    assert 0.5 * jnp.sum(r**2) == pytest.approx(float(problem()), rel=1e-12)


def test_an_unsupported_term_raises_before_scipy_is_called(monkeypatch, freq):
    def never(*args, **kwargs):
        raise AssertionError("SciPy was called")

    monkeypatch.setattr(scipy_solver, "scipy_least_squares", never)
    model = Pair(u=prf.Bounded(0.0, 5.0, value=1.0), v=prf.Unconstrained(0.0))
    target = Pair(u=2.0, v=1.0).s(freq)
    mll = MarginalLogLikelihood(predictor=Feature("s"), observed=target, likelihood=GaussianLikelihood(1e-3))
    gibbs = GibbsMarginalLogLikelihood(predictor=Feature("s"), observed=target, loss=MSELoss())
    unsupported = [
        TargetLoss(predictor=Feature("s"), target=target, loss=RMSELoss()),
        TargetLoss(predictor=Feature("s"), target=target, loss=MSELoss(multioutput="raw_values")),
        Negated(mll),
        Negated(gibbs),
        lambda m, f: jnp.sum(jnp.abs(m.s(f)) ** 2),
    ]
    for term in unsupported:
        with pytest.raises(NotImplementedError, match="not yet supported with a least-squares minimiser"):
            minimize((term, freq), model, solver=_quiet())

    plain = SummedTerms(model=model, terms=(BoundEvaluator(_mse(target), freq), lambda m: m.u**2))
    with pytest.raises(NotImplementedError, match="not yet supported"):
        minimize(plain, solver=_quiet())
    with pytest.raises(NotImplementedError, match="not yet supported"):
        minimize(PriorPenalized(SummedTerms(model=model, terms=as_terms(_mse(target), freq))), solver=_quiet())


# ---- Box space ------------------------------------------------------------------------


def test_iterates_stay_inside_the_box_from_a_start_on_a_closed_bound(freq):
    seen = []
    record = lambda u, v: seen.append((float(u), float(v)))

    def predictor(m, f):
        jax.debug.callback(record, m.u, m.v)
        return m.s(f)

    # The optimum of u is past its upper bound, so the fit presses against it.
    target = Pair(u=12.0, v=1.5).s(freq)
    model = Pair(u=prf.Bounded(0.0, 10.0, value=0.0), v=prf.Bounded(-2.0, 2.0, value=-2.0))
    result = minimize((_mse(target, predictor), freq), model, solver=_quiet())

    assert result.success
    assert seen and all(0.0 <= u <= 10.0 and -2.0 <= v <= 2.0 for u, v in seen)
    values = prf.values(result.model)
    assert values["u"] == pytest.approx(10.0, rel=1e-6)
    assert values["v"] == pytest.approx(1.5, rel=1e-6)


def test_run_minimizer_hands_a_least_squares_minimiser_the_box():
    class Recording(optimize_base.AbstractLeastSquaresMinimizer):
        def run(self, residual_fn, y0, args, bounds=None, max_iter=1024, **kwargs):
            Recording.seen = (y0, bounds, residual_fn(y0, args))
            return optimize_base.MinimizeResult(y=y0)

    model = Pair(u=prf.Bounded(0.0, 10.0, value=2.5), v=prf.Fixed(0.0))
    optimize_base.run_minimizer(
        lambda m, _: 0.5 * (m.u - 1.0) ** 2, model, Recording(),
        residual_fn=lambda m, _: jnp.atleast_1d(m.u - 1.0),
    )

    y0, (lower, upper), r = Recording.seen
    assert y0["u"] == pytest.approx(0.25) and lower["u"] == 0.0 and upper["u"] == 1.0
    assert r == pytest.approx(1.5)

    with pytest.raises(ValueError, match="needs a residual function"):
        optimize_base.run_minimizer(lambda m, _: m.u, model, Recording())


def test_a_start_on_a_one_sided_closed_bound_moves():
    result = _quiet().run(
        lambda values, _: jnp.stack([values["a"] - 1.0, values["b"] - 7.0]),
        {"a": jnp.array(5.0), "b": jnp.array(0.0)},
        bounds=({"a": jnp.array(-jnp.inf), "b": jnp.array(0.0)}, {"a": jnp.array(5.0), "b": jnp.array(jnp.inf)}),
    )

    assert result.success
    assert result.y["a"] == pytest.approx(1.0) and result.y["b"] == pytest.approx(7.0)


# ---- Nonfinite diagnostics ------------------------------------------------------------


def test_nonfinite_residual_raises_with_evaluation_context(monkeypatch):
    def fake_least_squares(fun, x, **kwargs):
        fun(x)
        fun(np.array([1.0]))

    monkeypatch.setattr(scipy_solver, "scipy_least_squares", fake_least_squares)
    with pytest.raises(FloatingPointError, match=r"residual evaluation 2.*nonfinite residual.*'x'"):
        _quiet().run(
            lambda values, _: jnp.where(values["x"] > 0.5, jnp.nan, values["x"]),
            {"x": jnp.array([0.0])},
        )


def test_nonfinite_jacobian_names_the_parameter(monkeypatch):
    def fake_least_squares(fun, x, jac, **kwargs):
        jac(x)

    monkeypatch.setattr(scipy_solver, "scipy_least_squares", fake_least_squares)
    with pytest.raises(FloatingPointError, match=r"Jacobian evaluation 1.*nonfinite Jacobian.*'b'"):
        _quiet().run(
            lambda values, _: jnp.stack([values["a"], jnp.sqrt(values["b"])]),
            {"a": jnp.array(1.0), "b": jnp.array(0.0)},
        )


def test_nonfinite_attempted_vector_raises(monkeypatch):
    def fake_least_squares(fun, x, **kwargs):
        fun(np.array([np.inf]))

    monkeypatch.setattr(scipy_solver, "scipy_least_squares", fake_least_squares)
    with pytest.raises(FloatingPointError, match=r"nonfinite attempted optimizer vector.*'x'"):
        _quiet().run(lambda values, _: values["x"], {"x": jnp.array([0.0])})


# ---- A degenerate line fit ------------------------------------------------------------


_TRUE = dict(w=3e-3, h=1.6e-3, ep_r=4.3, tand=0.02, sigma=5.8e7, length=0.1)
_START = dict(w=3.15e-3, h=1.55e-3, ep_r=4.4, tand=0.0, sigma=4e7, length=0.098)
_NAMES = dict(
    w="w", h="substrate.h", ep_r="substrate.dielectric.ep_r", tand="substrate.dielectric.tand",
    sigma="substrate.conductor.sigma", length="length",
)


def _line(v, bounded):
    b = (lambda lo, hi, x: prf.Bounded(lo, hi, value=x)) if bounded else (lambda lo, hi, x: x)
    return MicrostripLine(
        w=b(1e-3, 5e-3, v["w"]),
        h=b(0.8e-3, 3.2e-3, v["h"]),
        dielectric=ConstantDielectric(ep_r=b(2.0, 6.0, v["ep_r"]), tand=b(0.0, 0.05, v["tand"])),
        conductor=BulkConductor(sigma=b(1e7, 1e8, v["sigma"])),
        length=b(0.05, 0.2, v["length"]),
    )


def test_degenerate_microstrip_fit_recovers_the_truth():
    """Six jointly near-degenerate parameters (Jacobian condition number about 6e5 in box
    space), with `tand` starting on its closed lower bound (ADR-0009)."""
    freq = Frequency(10, 500, 101, "MHz")
    data = np.asarray(_line(_TRUE, bounded=False).s(freq))
    start = _line(_START, bounded=True)

    # The MSE is tiny near the solution, and SciPy's gtol is absolute, so the
    # tolerances are tightened to reach the truth.
    solver = ScipyLeastSquares(ftol=1e-15, xtol=1e-15, gtol=1e-15, show_progress=False)
    result = fit_minimize(start, data, freq, solver=solver, max_iter=2000)

    assert result.solution.success
    fitted = prf.values(result.model)
    for key, name in _NAMES.items():
        assert fitted[name] == pytest.approx(_TRUE[key], rel=1e-9), key

    # L-BFGS-B reports success on the same problem, at a wrong point: sigma is off by
    # about a third.
    lbfgsb = fit_minimize(start, data, freq, solver=ScipyMinimize(show_progress=False))
    assert lbfgsb.solution.success
    assert prf.values(lbfgsb.model)["substrate.conductor.sigma"] == pytest.approx(3.9e7, rel=0.05)
