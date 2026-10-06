"""A minimiser that honours bounds searches box space, the box of the parameters' bounds
(ADR-0007, #226). One that does not moves through raw space."""
import jax
import jax.numpy as jnp
import numpy as np
import parax.bijectors as db
import parax.distributions as dd
import pytest

import pmrf as prf
from pmrf._solver_view import SolverView
from pmrf.constraints import Interval, Positive
from pmrf.distributions import Normal, Uniform
from pmrf.optimize import base as optimize_base
from pmrf.optimize.solvers.jaxopt import LBFGSB
from pmrf.optimize.solvers.optimistix import BFGS
from pmrf.optimize.solvers.scipy import ScipyMinimize


class Pair(prf.Model):
    u: prf.Param = prf.param()
    v: prf.Param = prf.param()

    def s(self, freq):
        return jnp.ones((freq.npoints, 1, 1)) * (self.u + self.v)


def _bounded_solvers():
    return [ScipyMinimize(method="L-BFGS-B", show_progress=False), ScipyMinimize(show_progress=False), LBFGSB()]


_BOUNDED_IDS = ["scipy-lbfgsb", "scipy-default", "jaxopt-lbfgsb"]


class _RecordingBoundedMinimizer(optimize_base.AbstractBoundedMinimizer):
    """Records what it is given and returns `y0` unchanged."""

    def run(self, fn, y0, args, bounds=None, max_iter=1024, **kwargs):
        _RecordingBoundedMinimizer.seen = (y0, bounds, fn(y0, args), jax.grad(fn)(y0, args))
        return optimize_base.MinimizeResult(y=y0)


# ---- Capability -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, honours",
    [
        (None, True), ("L-BFGS-B", True), ("TNC", True), ("SLSQP", True), ("trust-constr", True),
        ("Powell", True), ("Nelder-Mead", True), ("COBYLA", True),
        ("BFGS", False), ("CG", False), ("Newton-CG", False),
    ],
)
def test_scipy_minimize_honours_bounds_by_method(method, honours):
    assert ScipyMinimize(method=method).honours_bounds is honours


def test_minimiser_interface_declares_whether_it_honours_bounds():
    assert LBFGSB().honours_bounds
    assert not BFGS().honours_bounds


# ---- Box space ------------------------------------------------------------------------


def test_box_is_the_unit_box_between_two_finite_bounds():
    model = Pair(u=prf.Bounded(0.0, 10.0, value=0.0), v=prf.Bounded(-2.0, 2.0, value=1.0))
    y0, (lower, upper) = SolverView(model, "optimize").box()

    assert y0["u"] == 0.0 and y0["v"] == pytest.approx(0.75)
    assert lower["u"] == 0.0 and upper["u"] == 1.0


def test_open_edges_are_inset():
    model = Pair(
        u=prf.Constrained(Interval(0.0, 10.0, closed=False), value=5.0),
        v=prf.Constrained(Positive(), value=3.0),
    )
    y0, (lower, upper) = SolverView(model, "optimize").box()

    eps = jnp.finfo(y0["u"].dtype).eps
    assert lower["u"] == pytest.approx(eps / 2) and upper["u"] == pytest.approx(1 - eps)
    # Without two finite bounds, the box is declared space.
    assert y0["v"] == 3.0 and lower["v"] == pytest.approx(3 * eps) and upper["v"] == jnp.inf


def test_a_normal_prior_gets_an_unbounded_box():
    """The box is built from bounds, not the prior: no CDF box for a normal prior."""
    model = Pair(u=prf.Random(Normal(1.0, 2.0), value=4.0), v=prf.Random(Uniform(0.0, 50.0), value=10.0))
    y0, (lower, upper) = SolverView(model, "optimize").box()

    assert y0["u"] == 4.0 and lower["u"] == -jnp.inf and upper["u"] == jnp.inf
    # A uniform prior's support is its bounds.
    assert y0["v"] == pytest.approx(0.2) and lower["v"] == 0.0 and upper["v"] == 1.0


def test_joint_prior_block_gets_the_box_of_its_parameters_bounds():
    """The parameters under a joint prior (ADR-0005) are searched over their own bounds,
    not the prior's whitened space."""
    parts = {
        "a": Pair(u=prf.Bounded(0.0, 10.0, value=2.0), v=prf.Bounded(0.0, 4.0, value=1.0), name="a"),
    }
    base = dd.Independent(dd.Normal(jnp.zeros(2), jnp.ones(2)))
    model = prf.prior(parts, ("a.u", "a.v"), dd.Transformed(base, db.Shift(jnp.zeros(2))), space="raw")
    view = SolverView(model, "optimize")
    y0, (lower, upper) = view.box()

    assert y0["a.u"] == pytest.approx(0.2) and y0["a.v"] == pytest.approx(0.25)
    assert lower["a.u"] == 0.0 and upper["a.v"] == 1.0
    fitted = view.updated({"a.u": jnp.asarray(1.0), "a.v": jnp.asarray(0.5)}, "box")
    assert prf.values(fitted) == pytest.approx({"a.u": 10.0, "a.v": 2.0})
    objective = view.objective(lambda m, args: m["a"].u + m["a"].v, "box")
    assert objective({"a.u": jnp.asarray(1.0), "a.v": jnp.asarray(0.5)}, None) == pytest.approx(12.0)


def test_box_objective_is_the_model_objective_with_a_finite_gradient_on_a_bound():
    """The objective sees physical values, and has a finite, non-zero gradient at a start
    on a closed bound, where the raw value is infinite."""
    model = Pair(u=prf.Bounded(0.0, 10.0, value=0.0, scale=1e-3), v=prf.Unconstrained(2.0))
    fn = lambda m, args: (m.u - 3e-3) ** 2 * 1e6 + m.v ** 2
    optimize_base.run_minimizer(fn, model, _RecordingBoundedMinimizer())
    y0, bounds, value, grad = _RecordingBoundedMinimizer.seen

    assert y0 == {"u": 0.0, "v": 2.0}
    assert bounds[0]["u"] == 0.0 and bounds[1]["u"] == 1.0
    assert bounds[0]["v"] == -jnp.inf and bounds[1]["v"] == jnp.inf
    assert value == pytest.approx(fn(prf.unwrap(model), None))
    # d/du of (10 u - 3)^2 at u = 0.
    assert grad["u"] == pytest.approx(-60.0) and grad["v"] == pytest.approx(4.0)


def test_box_objective_follows_ties():
    model = prf.tie(Pair(u=prf.Bounded(0.0, 10.0, value=1.0), v=prf.Bounded(0.0, 10.0, value=1.0)), "v", "u", fn=lambda u: 2 * u)
    view = SolverView(model, "optimize")
    y0, _ = view.box()

    assert set(y0) == {"u"}
    assert view.objective(lambda m, args: m.wrapped.v, "box")({"u": jnp.asarray(0.3)}, None) == pytest.approx(6.0)


# ---- Bounded minimisers ---------------------------------------------------------------


@pytest.mark.parametrize("solver", _bounded_solvers(), ids=_BOUNDED_IDS)
def test_bounded_minimiser_starts_on_a_closed_bound_and_moves(solver):
    model = Pair(u=prf.Bounded(0.0, 10.0, value=0.0), v=prf.Fixed(0.0))
    fitted, _ = optimize_base.run_minimizer(lambda m, args: (m.u - 3.0) ** 2, model, solver, max_iter=200)

    assert prf.values(fitted)["u"] == pytest.approx(3.0, rel=1e-4)


@pytest.mark.parametrize("solver", _bounded_solvers(), ids=_BOUNDED_IDS)
@pytest.mark.parametrize("target, bound", [(-5.0, 0.0), (15.0, 10.0)], ids=["lower", "upper"])
def test_bounded_minimiser_ends_exactly_on_a_bound(solver, target, bound):
    model = Pair(u=prf.Bounded(0.0, 10.0, value=4.0), v=prf.Fixed(0.0))
    fitted, _ = optimize_base.run_minimizer(lambda m, args: (m.u - target) ** 2, model, solver, max_iter=200)

    assert prf.values(fitted)["u"] == bound


@pytest.mark.parametrize("solver", _bounded_solvers(), ids=_BOUNDED_IDS)
def test_open_edge_is_never_evaluated(solver):
    """With an optimum past an open bound, the minimiser stops on the inset edge without
    evaluating the objective on the bound."""
    seen = []
    record = lambda u: seen.append(float(u))

    def fn(m, args):
        jax.debug.callback(record, m.u)
        return m.u + m.v

    model = Pair(u=prf.Constrained(Interval(0.0, 10.0, closed=False), value=5.0), v=prf.Constrained(Positive(), value=1.0))
    fitted, _ = optimize_base.run_minimizer(fn, model, solver, max_iter=200)

    assert seen and min(seen) > 0.0
    values = prf.values(fitted)
    eps = jnp.finfo(values["u"].dtype).eps
    assert values["u"] == pytest.approx(5 * eps, rel=1e-6)
    assert values["v"] == pytest.approx(eps, rel=1e-6)


# ---- Minimisers that do not honour bounds ---------------------------------------------


def test_scipy_bfgs_moves_in_raw_space_and_never_leaves_the_bounds():
    seen = []
    record = lambda u: seen.append(float(u))

    def fn(m, args):
        jax.debug.callback(record, m.u)
        return (m.u - 15.0) ** 2

    model = Pair(u=prf.Bounded(0.0, 10.0, value=0.0), v=prf.Fixed(0.0))
    fitted, _ = optimize_base.run_minimizer(fn, model, ScipyMinimize(method="BFGS", show_progress=False), max_iter=200)

    # The start on the bound is nudged inside it, as for any raw-space minimiser.
    assert seen[0] == pytest.approx(1e-5, rel=1e-6)
    assert all(0.0 <= u <= 10.0 for u in seen)
    assert 9.9 < prf.values(fitted)["u"] <= 10.0
