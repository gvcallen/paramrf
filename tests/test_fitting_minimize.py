# tests/test_optimize/test_fit.py
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from pmrf.frequency import Frequency
from pmrf.models import CoaxialLine, Model, RLGCLine
from pmrf.materials import BulkConductor, ConstantDielectric
from pmrf.fitting import fit_minimize
from pmrf.parameters import Fixed, Bounded, Param, Random, update, values
from pmrf.losses import MSELoss, RMSELoss
from pmrf.optimize import ScipyMinimize
from pmrf.optimize.solvers import scipy as scipy_solver
from pmrf.distributions import LogNormal, Normal
from pmrf.covariance_kernels import Matern52Kernel
from pmrf.discrepancy_models import GaussianProcess
from pmrf.likelihoods import GaussianLikelihood

@pytest.fixture
def fit_freq():
    return Frequency(start=1.0, stop=5.0, npoints=21, unit='GHz')

@pytest.fixture
def truth_model():
    return CoaxialLine(
        d_in=1.12e-3, 
        d_out=3.2e-3, 
        dielectric=ConstantDielectric(ep_r=1.384, tand=0.001),
        conductor=BulkConductor(sigma=1 / 1.6e-8),
        length=0.1,  # This is the target length we want to find (10 cm)
    )

@pytest.fixture
def target_network(fit_freq, truth_model):
    skrf = pytest.importorskip("skrf")
    s_target = np.array(truth_model.s(fit_freq))
    freq_skrf = fit_freq.to_skrf()
    
    return skrf.Network(frequency=freq_skrf, s=s_target, z0=50)

@pytest.fixture
def starting_model():
    """
    The model we will actually optimize. 
    """
    return CoaxialLine(
        d_in=Fixed(1.12e-3),
        d_out=Fixed(3.2e-3),
        dielectric=ConstantDielectric(ep_r=Fixed(1.384), tand=Fixed(0.001)),
        conductor=BulkConductor(sigma=Fixed(1 / 1.6e-8)),
        # Start at 9.5 cm to stay within a fraction of a wavelength of 10 cm
        length=Bounded(0.05, 0.15, value=0.095)
    )

def test_fit_skrf_synthetic_data(starting_model, target_network):
    # Note that this also tests full two-port fitting (all S-params).
    # We test only one feature (s21) below
    results = fit_minimize(starting_model, target_network)
    fitted_model = results.model

    assert jnp.allclose(fitted_model.length.value, 0.1, atol=1e-3)

    target_freq = Frequency.from_skrf(target_network.frequency)
    residuals = target_network.s - fitted_model.s(target_freq)
    # Add a tiny epsilon to avoid log10(0) if the fit is mathematically perfect
    max_residual_db = np.max(20 * np.log10(np.abs(residuals) + 1e-15))
    
    assert max_residual_db < -30

def test_fit_raw_ndarray(truth_model, starting_model, fit_freq):
    target_s = np.array(truth_model.s(fit_freq))
    results = fit_minimize(starting_model, target_s, frequency=fit_freq)

    assert jnp.allclose(results.model.length.value, 0.1, atol=1e-3)

def test_fit_missing_freq_error(starting_model, fit_freq):
    dummy_s = jnp.zeros((fit_freq.npoints, 2, 2), dtype=complex)
    with pytest.raises(Exception, match="Frequency must be passed if Network data is not provided"):
        fit_minimize(starting_model, dummy_s, frequency=None)

def test_fit_specific_feature(truth_model, starting_model, fit_freq):
    from pmrf.evaluators import Feature
    s21_mag_target = Feature('s21_mag')(truth_model, fit_freq)
    # |S21| barely moves with length, so the MSE starts near 5e-7. SciPy's default
    # gtol, and its ftol (absolute below an objective of 1), stop L-BFGS-B there.
    results = fit_minimize(
        starting_model, 
        s21_mag_target, 
        frequency=fit_freq, 
        features='s21_mag',
        solver=ScipyMinimize(options={'gtol': 1e-12, 'ftol': 1e-15}),
    )
    
    assert jnp.allclose(results.model.length.value, 0.1, atol=1e-3)

# ---------------------------------------------------------
# Default frequentist loss
# ---------------------------------------------------------

class ConstantModel(Model):
    val: Param

    def s(self, freq: Frequency):
        return jnp.ones((freq.npoints, 1, 1), dtype=complex) * self.val


def _constant_outputs(model, freq, n_outputs):
    return jnp.full((freq.npoints, n_outputs), model.val)


def _alternating(npoints, spread):
    return spread * (-1.0) ** np.arange(npoints)


def test_mse_and_rmse_reach_the_same_single_output_optimum(fit_freq):
    # Residuals are non-zero at the optimum, so the RMSE has no kink there.
    target = (2.0 + _alternating(fit_freq.npoints, 0.5))[:, None]
    features = lambda m, f: _constant_outputs(m, f, 1)
    model = ConstantModel(val=Bounded(0.0, 5.0, value=1.0))

    mse = fit_minimize(model, target, frequency=fit_freq, features=features, loss=MSELoss())
    rmse = fit_minimize(model, target, frequency=fit_freq, features=features, loss=RMSELoss())

    # Both are minimised by the target mean.
    assert jnp.allclose(mse.model.val.value, np.mean(target), atol=1e-4)
    assert jnp.allclose(rmse.model.val.value, np.mean(target), atol=1e-4)

def test_default_loss_pools_residuals_across_outputs(fit_freq):
    # Output 0 is centred on 0 with a small spread; output 1 on 1 with a large one.
    # Pooled MSE is (a² + s0²) + ((a - 1)² + s1²), minimised at a = 0.5. The mean of
    # per-output RMSEs, sqrt(a² + s0²) + sqrt((a - 1)² + s1²), favours the output
    # that fits well, and is minimised near a = 0.1.
    s0, s1 = 0.1, 1.0
    target = np.stack([
        _alternating(fit_freq.npoints, s0),
        1.0 + _alternating(fit_freq.npoints, s1),
    ], axis=-1)
    features = lambda m, f: _constant_outputs(m, f, 2)
    model = ConstantModel(val=Bounded(-1.0, 2.0, value=0.8))

    # Locate both optima on a grid: the alternating spread is not zero-mean over an
    # odd number of points, which shifts them slightly from the closed forms above.
    a = np.linspace(-1.0, 2.0, 300001)[:, None]
    per_output_mse = np.stack([np.mean((target[:, k] - a) ** 2, axis=1) for k in range(2)])
    a_mse = a[np.argmin(per_output_mse.mean(axis=0)), 0]
    a_rmse = a[np.argmin(np.sqrt(per_output_mse).mean(axis=0)), 0]
    assert abs(a_rmse - a_mse) > 0.2

    default = fit_minimize(model, target, frequency=fit_freq, features=features)
    rmse = fit_minimize(model, target, frequency=fit_freq, features=features, loss=RMSELoss())

    assert jnp.allclose(default.model.val.value, a_mse, atol=1e-4)
    assert jnp.allclose(rmse.model.val.value, a_rmse, atol=1e-3)


def test_bayesian_fit_propagates_nonfinite_objective():
    model = ConstantModel(val=Random(Normal(0.0, 1.0), value=1.0))
    frequency = Frequency(10.0, 20.0, 3, "MHz")
    data = jnp.zeros((3,))
    with pytest.raises(FloatingPointError, match=r"trust-constr objective evaluation 1.*nonfinite loss"):
        fit_minimize(
            model,
            data,
            frequency=frequency,
            features=lambda m, f: jnp.full((f.npoints,), jnp.nan),
            inference="bayesian",
            likelihood=GaussianLikelihood(1e-8),
            solver=ScipyMinimize(method="trust-constr", show_progress=False),
        )


def test_bounded_trust_constr_fits_lognormal_gp_hyperparameters_from_tiny_starts(monkeypatch):
    frequency = Frequency(10.0, 500.0, 20, "MHz")
    line = RLGCLine(
        length=Fixed(10.0),
        R=Fixed(0.5),
        L=Fixed(250e-9),
        C=Fixed(100e-12),
        G=Fixed(1e-6),
    )
    predictor = lambda model, freq: jnp.real(model.s(freq)[:, 0, 1])
    observed = predictor(line, frequency) + 1e-3 * jnp.sin(frequency.f_scaled / 80.0)
    scipy_minimize = scipy_solver.scipy_minimize
    observed_evaluations = []

    def recording_minimize(fun, x0, *args, **kwargs):
        def finite_fun(x, *fn_args):
            result = fun(x, *fn_args)
            if isinstance(result, tuple):
                loss, grad = result
                observed_evaluations.append((float(np.asarray(loss)), np.asarray(grad)))
                assert np.isfinite(np.asarray(loss)).all()
                assert np.isfinite(np.asarray(grad)).all()
            else:
                observed_evaluations.append((float(np.asarray(result)), None))
                assert np.isfinite(np.asarray(result)).all()
            return result

        return scipy_minimize(finite_fun, x0, *args, **kwargs)

    monkeypatch.setattr(scipy_solver, "scipy_minimize", recording_minimize)
    outcomes = []
    for variance_start, variance_loc in ((1e-5, 1e-5), (1e-7, 1e-7)):
        discrepancy = GaussianProcess(
            Matern52Kernel(
                Random(LogNormal(jnp.log(30.0), 1.0), value=30.0)
            )
            * Random(LogNormal(jnp.log(variance_loc), 1.0), value=variance_start),
            jitter=1e-12,
        )
        result = fit_minimize(
            line,
            observed,
            frequency=frequency,
            features=predictor,
            inference="bayesian",
            likelihood=GaussianLikelihood(1e-8),
            discrepancy=discrepancy,
            solver=ScipyMinimize(method="trust-constr", show_progress=False),
            max_iter=1000,
        )
        problem = result.solution.problem
        free = values(problem, free_only=True, space="declared")
        names = tuple(name for name in free if name.endswith(("lengthscale", "variance")))
        assert result.solution.success
        assert len(names) == 2
        fitted = jnp.asarray([free[name] for name in names])
        assert np.isfinite(fitted).all() and np.all(np.asarray(fitted) > 0.0)
        expected_start = np.array([
            30.0 if name.endswith("lengthscale") else variance_start for name in names
        ])
        np.testing.assert_allclose(
            result.solution.metrics.box_origin, expected_start, rtol=2e-15, atol=0.0
        )
        recovered = (
            np.asarray(result.solution.metrics.box_origin)
            + np.asarray(result.solution.metrics.box_scale) * np.asarray(result.solution.metrics.x)
        )
        np.testing.assert_allclose(recovered, np.asarray(fitted), rtol=1e-12, atol=0.0)

        def log_coordinate_loss(log_values):
            moved = update(
                problem,
                {name: jnp.exp(log_values[i]) for i, name in enumerate(names)},
                space="declared",
            )
            return moved()

        log_gradient = jax.grad(log_coordinate_loss)(jnp.log(fitted))
        assert np.max(np.abs(np.asarray(log_gradient))) < 1e-3
        initial = dict(zip(names, expected_start))
        initial_loss = update(problem, initial, space="declared")()
        assert np.isfinite(initial_loss)
        assert log_coordinate_loss(jnp.log(fitted)) < initial_loss
        variance_index = next(i for i, name in enumerate(names) if name.endswith("variance"))
        if variance_start == 1e-7:
            assert float(fitted[variance_index]) < 1e-6
        outcomes.append(result)

    assert observed_evaluations
