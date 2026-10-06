# tests/test_optimize/test_optimize.py
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds

import pmrf as prf
from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.parameters import Bounded
from pmrf.optimize.minimize import minimize
from pmrf.optimize.solvers.scipy import ScipyMinimize
from pmrf.optimize.solvers import scipy as scipy_solver

# ---------------------------------------------------------
# Dummy Concrete Models for Testing
# ---------------------------------------------------------

class DummyOptModel(Model):
    """A simple 1-port model with one free parameter for optimization."""
    val: prf.Param = prf.param(default=1.0, as_free=True)

    def s(self, freq: Frequency) -> jnp.ndarray:
        nf = freq.npoints
        return jnp.ones((nf, 1, 1), dtype=complex) * self.val

# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

@pytest.fixture
def model():
    return DummyOptModel(val=1.0)

# ---------------------------------------------------------
# `minimize` Tests
# ---------------------------------------------------------

def test_minimize_scipy_unbounded(model, basic_freq):
    """Test standard unconstrained optimization using the default Scipy backend."""
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    result = minimize(obj_fn, model, basic_freq, solver=ScipyMinimize())
    
    assert isinstance(result.model, DummyOptModel)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-3)

def test_minimize_scipy_bounded(basic_freq):
    """Test that parameter boundaries are successfully intercepted and enforced."""
    # Initialize parameter at 1.0, trying to reach 5.0, but capped at 3.0
    bounded_param = Bounded(0.0, 3.0, value=1.0)
    bounded_model = DummyOptModel(val=bounded_param)
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    result = minimize(obj_fn, bounded_model, basic_freq, solver=ScipyMinimize())
    
    # The optimizer should hit the upper bound and stop
    assert jnp.allclose(result.model.val, 3.0, atol=1e-3)


def test_trust_constr_uses_feasible_affine_coordinates_and_preserves_scipy_metrics(monkeypatch):
    x0 = np.array([2.0, 3.0, 4.0, 0.0, 5.0])
    lower = np.array([0.0, 1.0, -np.inf, -np.inf, 5.0])
    upper = np.array([4.0, np.inf, 10.0, np.inf, np.inf])
    captured = {}

    def fake_minimize(fun, x, *, jac, method, bounds, **kwargs):
        captured.update(x=x, jac=jac, method=method, bounds=bounds)
        captured["loss"], captured["grad"] = fun(np.zeros_like(x))
        result = type("Result", (), {})()
        result.x = np.array([0.1, -0.2, 0.3, 0.4, 0.5])
        result.success = True
        return result

    monkeypatch.setattr(scipy_solver, "scipy_minimize", fake_minimize)
    result = ScipyMinimize(method="TRUST-CONSTR", show_progress=False).run(
        lambda values, _: jnp.sum(values["x"] ** 2),
        {"x": jnp.asarray(x0)},
        bounds=({"x": jnp.asarray(lower)}, {"x": jnp.asarray(upper)}),
    )

    scale = np.array([4.0, 2.0, 6.0, 1.0, 1.0])
    np.testing.assert_array_equal(captured["x"], np.zeros(5))
    assert captured["method"].lower() == "trust-constr"
    assert isinstance(captured["bounds"], Bounds)
    assert captured["bounds"].keep_feasible.all()
    np.testing.assert_allclose(captured["bounds"].lb, (lower - x0) / scale)
    np.testing.assert_allclose(captured["bounds"].ub, (upper - x0) / scale)
    assert captured["loss"] == pytest.approx(np.sum(x0**2))
    np.testing.assert_allclose(captured["grad"], 2.0 * x0 * scale)
    np.testing.assert_allclose(result.y["x"], x0 + scale * np.array([0.1, -0.2, 0.3, 0.4, 0.5]))
    np.testing.assert_array_equal(result.metrics.box_origin, x0)
    np.testing.assert_array_equal(result.metrics.box_scale, scale)
    np.testing.assert_array_equal(result.metrics.x, [0.1, -0.2, 0.3, 0.4, 0.5])


def test_trust_constr_affine_coordinates_support_gradient_free_calls(monkeypatch):
    captured = {}

    def fake_minimize(fun, x0, *, jac, method, bounds, **kwargs):
        captured.update(x0=x0, jac=jac, bounds=bounds)
        captured["value"] = fun(x0)
        result = type("Result", (), {})()
        result.x = np.array([0.25])
        result.success = True
        return result

    monkeypatch.setattr(scipy_solver, "scipy_minimize", fake_minimize)
    result = ScipyMinimize(method="trust-constr", use_grad=False, show_progress=False).run(
        lambda values, _: jnp.sum(values["x"] ** 2),
        {"x": jnp.array([2.0])},
        bounds=({"x": jnp.array([0.0])}, {"x": jnp.array([4.0])}),
    )

    np.testing.assert_array_equal(captured["x0"], [0.0])
    assert captured["jac"] is False
    assert isinstance(captured["bounds"], Bounds)
    assert captured["value"] == pytest.approx(4.0)
    np.testing.assert_allclose(result.y["x"], [3.0])


def test_scipy_minimizer_raises_on_first_nonfinite_loss_with_eval_context(monkeypatch):
    calls = []

    def fake_minimize(fun, x, **kwargs):
        fun(x)
        fun(np.array([1.0]))

    monkeypatch.setattr(scipy_solver, "scipy_minimize", fake_minimize)
    solver = ScipyMinimize(method="BFGS", show_progress=False)
    with pytest.raises(
        FloatingPointError,
        match=r"SciPy BFGS objective evaluation 2.*nonfinite loss.*'x'",
    ):
        solver.run(
            lambda values, _: jnp.where(values["x"][0] > 0.5, jnp.inf, values["x"][0] ** 2),
            {"x": jnp.array([0.0])},
        )


def test_scipy_minimizer_names_nonfinite_gradient_parameter(monkeypatch):
    @jax.custom_jvp
    def finite_loss_with_bad_gradient(x):
        return jnp.asarray(1.0)

    @finite_loss_with_bad_gradient.defjvp
    def bad_gradient_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        tangent = jax.lax.cond(
            x > 0.5,
            lambda _: dx,
            lambda _: jnp.asarray(jnp.nan) * dx,
            operand=None,
        )
        return finite_loss_with_bad_gradient(x), tangent

    def fake_minimize(fun, x, **kwargs):
        fun(x)
        fun(np.array([0.0]))

    monkeypatch.setattr(scipy_solver, "scipy_minimize", fake_minimize)
    solver = ScipyMinimize(method="BFGS", show_progress=False)
    with pytest.raises(
        FloatingPointError,
        match=r"SciPy BFGS objective evaluation 2.*nonfinite gradient.*'x'",
    ):
        solver.run(lambda values, _: finite_loss_with_bad_gradient(values["x"][0]), {"x": jnp.array([1.0])})


def test_scipy_gradient_free_path_checks_nonfinite_loss_and_attempted_coordinates(monkeypatch):
    def fake_minimize(fun, x, **kwargs):
        fun(x)
        fun(np.array([np.inf]))

    monkeypatch.setattr(scipy_solver, "scipy_minimize", fake_minimize)
    solver = ScipyMinimize(method="Nelder-Mead", show_progress=False)
    with pytest.raises(
        FloatingPointError,
        match=r"SciPy Nelder-Mead objective evaluation 2.*nonfinite attempted optimizer vector.*'x'",
    ):
        solver.run(lambda values, _: jnp.sum(values["x"] ** 2), {"x": jnp.array([1.0])})


def test_scipy_gradient_free_path_checks_nonfinite_loss(monkeypatch):
    def fake_minimize(fun, x, **kwargs):
        assert np.isfinite(fun(x))
        fun(np.array([1.0]))

    monkeypatch.setattr(scipy_solver, "scipy_minimize", fake_minimize)
    with pytest.raises(FloatingPointError, match=r"evaluation 2.*nonfinite loss"):
        ScipyMinimize(method="Nelder-Mead", show_progress=False).run(
            lambda values, _: jnp.where(values["x"][0] > 0.5, jnp.nan, values["x"][0] ** 2),
            {"x": jnp.array([0.0])},
        )


def test_scipy_closes_progress_bar_after_numerical_failure(monkeypatch):
    progress = []

    class RecordingProgress:
        def __init__(self, **kwargs):
            self.closed = False
            progress.append(self)

        def update(self, _):
            pass

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(scipy_solver, "tqdm", RecordingProgress)
    monkeypatch.setattr(scipy_solver, "scipy_minimize", lambda fun, x, **kwargs: fun(x))
    with pytest.raises(FloatingPointError):
        ScipyMinimize(method="BFGS", show_progress=True).run(
            lambda values, _: jnp.asarray(jnp.inf), {"x": jnp.array([1.0])}
        )
    assert progress and progress[0].closed


@pytest.mark.parametrize(
    "method,bounds",
    [
        ("L-BFGS-B", True),
        ("trust-constr", False),
    ],
)
def test_affine_coordinates_are_limited_to_bounded_trust_constr(monkeypatch, method, bounds):
    captured = {}

    def fake_minimize(fun, x, **kwargs):
        captured.update(x=x, **kwargs)
        if kwargs["jac"]:
            fun(x)
        else:
            fun(x)
        result = type("Result", (), {})()
        result.x = np.asarray(x)
        result.success = True
        return result

    monkeypatch.setattr(scipy_solver, "scipy_minimize", fake_minimize)
    box = ({"x": jnp.array([-2.0])}, {"x": jnp.array([2.0])}) if bounds else None
    result = ScipyMinimize(method=method, show_progress=False).run(
        lambda values, _: jnp.sum(values["x"] ** 2), {"x": jnp.array([0.5])}, bounds=box
    )

    np.testing.assert_array_equal(captured["x"], [0.5])
    if bounds:
        assert isinstance(captured["bounds"], list)
        assert not hasattr(result.metrics, "box_origin")
    else:
        assert captured["bounds"] is None
        assert not hasattr(result.metrics, "box_scale")


def test_public_minimize_propagates_nonfinite_objective(monkeypatch):
    def fake_minimize(fun, x, **kwargs):
        return fun(x)

    monkeypatch.setattr(scipy_solver, "scipy_minimize", fake_minimize)
    with pytest.raises(FloatingPointError, match="nonfinite loss"):
        minimize(
            lambda model, _: jnp.asarray(jnp.nan),
            DummyOptModel(val=1.0),
            Frequency(1.0, 2.0, 2, "GHz"),
            solver=ScipyMinimize(method="BFGS", show_progress=False),
        )


@pytest.mark.parametrize("start", [0.0, 0.4], ids=["closed-boundary", "interior"])
def test_trust_constr_never_evaluates_objective_outside_closed_box(monkeypatch, start):
    model = DummyOptModel(val=Bounded(0.0, 1.0, value=start))
    evaluations = []
    scipy_minimize = scipy_solver.scipy_minimize

    def recording_minimize(fun, x0, *args, **kwargs):
        def record_box_evaluation(z, *fn_args):
            box_value = start + float(np.asarray(z)[0])
            evaluations.append(box_value)
            return fun(z, *fn_args)

        return scipy_minimize(record_box_evaluation, x0, *args, **kwargs)

    monkeypatch.setattr(scipy_solver, "scipy_minimize", recording_minimize)

    def objective(candidate, _):
        value = jnp.asarray(candidate.val)
        return jnp.where((value < 0.0) | (value > 1.0), jnp.nan, (value - 0.8) ** 2)

    result = prf.optimize.minimize(
        objective,
        model,
        Frequency(1.0, 2.0, 2, "GHz"),
        solver=ScipyMinimize(
            method="trust-constr",
            show_progress=False,
            options={"gtol": 1e-10},
        ),
        max_iter=1000,
    )

    assert result.success
    assert evaluations
    assert np.all(np.asarray(evaluations) >= 0.0)
    assert np.all(np.asarray(evaluations) <= 1.0)
    assert np.isfinite(result.model.val.value)
    assert 0.0 <= result.model.val.value <= 1.0

def test_minimize_nelder(model, basic_freq):
    """Test the Nelder-Mead solver"""
    optx = pytest.importorskip("optimistix")
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    # Use a gradient-free JAX solver
    solver = prf.optimize.NelderMead(xrtol=1e-5, xatol=1e-5)
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iter=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_bfgs(model, basic_freq):
    """Test the BFGS solver"""
    optx = pytest.importorskip("optimistix")
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    solver = prf.optimize.BFGS(step_rtol=1e-5, step_atol=1e-5)
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iter=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_lbfgsb(model, basic_freq):
    """Test the LBFGS-B solver"""
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    solver = prf.optimize.LBFGSB()
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iter=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_optimistix(model, basic_freq):
    """Test the optimistix wrapper"""
    optx = pytest.importorskip("optimistix")
    
    def obj_fn(m, f):
        return jnp.sum(jnp.abs(m.val - 5.0)**2)
    
    solver = prf.optimize.OptimistixMinimise(solver=optx.BFGS(rtol=1e-5, atol=1e-5))
    
    result = minimize(obj_fn, model, basic_freq, solver=solver, max_iter=500)
    assert jnp.allclose(result.model.val, 5.0, atol=1e-2)

def test_minimize_list_of_objectives(model, basic_freq):
    """Ensure that passing a list of callables automatically sums them."""
    # The minimum of (x-2)^2 + (x-4)^2 is exactly x=3
    obj1 = lambda m, f: jnp.sum((m.val - 2.0)**2)
    obj2 = lambda m, f: jnp.sum((m.val - 4.0)**2)
    
    result = minimize([obj1, obj2], model, basic_freq)
    assert jnp.allclose(result.model.val, 3.0, atol=1e-3)
