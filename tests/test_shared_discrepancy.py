"""Public shared-discrepancy behavior against explicit dense Gaussian conditioning."""

import numpy as np
import scipy.linalg
from scipy.optimize import minimize
import pytest

import jax
import jax.numpy as jnp
import equinox as eqx
import parax.bijectors as bij

from pmrf.frequency import Frequency
from pmrf.objectives import MarginalLogLikelihood, SharedMarginalLogLikelihood, SummedTerms
from pmrf.stats import (
    ComplexPowerMixing, GaussianLikelihood, GaussianProcess, IdentityMixing,
    ConstantKernel, Matern32Kernel, RBFKernel, SharedLatent, gram,
)


def _predict(model, frequency):
    return model


def _dense_log_prob(residual, covariance):
    chol = np.linalg.cholesky(covariance)
    solved = scipy.linalg.cho_solve((chol, True), residual)
    return -0.5 * (residual @ solved + 2 * np.log(np.diag(chol)).sum()
                   + len(residual) * np.log(2 * np.pi))


class TwoRowKernel(eqx.Module):
    base: RBFKernel

    def __call__(self, x1, x2):
        value = self.base(x1, x2)
        return jnp.stack((value, value))


@pytest.mark.parametrize('noise', [0.07, jnp.array([0.07, 0.09, 0.06, 0.08])])
def test_identity_reduces_to_single_dataset_gp(noise):
    frequency = Frequency(1, 4, 4, 'GHz')
    model = jnp.array([0.1, 0.2, 0.3, 0.4])
    observed = model + jnp.array([0.04, -0.08, 0.02, 0.1])
    kernel = RBFKernel(lengthscale=0.9)
    evaluator = MarginalLogLikelihood(_predict, observed, GaussianLikelihood(noise))
    gp = MarginalLogLikelihood(_predict, observed, GaussianLikelihood(noise),
                               GaussianProcess(kernel, jitter=1e-10))
    shared = SharedMarginalLogLikelihood(
        ((evaluator, frequency),), (SharedLatent(kernel, IdentityMixing()),)
    )
    np.testing.assert_allclose(shared(model), gp(model, frequency), rtol=1e-10)
    np.testing.assert_allclose(shared.predict_latents(model, frequency).mean(),
                               gp.predict_discrepancy(model, frequency, frequency).mean(), rtol=1e-10)
    np.testing.assert_allclose(shared.predict_latents(model, frequency).covariance(),
                               gp.predict_discrepancy(model, frequency, frequency).covariance(), rtol=1e-10)
    np.testing.assert_allclose(SummedTerms(model, (shared,))(), shared(model), rtol=1e-10)


def test_coupled_blocks_mixed_grids_and_transfer_match_dense_reference():
    first = Frequency.from_f([1., 2., 3.], 'GHz')
    second = Frequency.from_f([2., 3.5], 'GHz')
    model = (jnp.array([0.2, 0.3, 0.5]), jnp.array([0.1, -0.3]))
    observed = (jnp.array([[0.3, 0.1], [0.2, 0.35], [0.65, 0.4]]),
                jnp.array([[0.15, 0.2], [-0.2, -0.25]]))
    rho = 0.6

    def predictor(index):
        return lambda candidate, frequency: jnp.broadcast_to(candidate[index][:, None], (len(frequency), 2))

    def mixing(prediction, x):
        return jnp.broadcast_to(jnp.array([[1.], [rho]]), (len(x), 2, 1))

    datasets = tuple((MarginalLogLikelihood(predictor(i), observed[i], GaussianLikelihood(0.04)), f)
                     for i, f in enumerate((first, second)))
    shared = SharedMarginalLogLikelihood(datasets, (SharedLatent(RBFKernel(0.8), mixing),))
    union = np.array([1., 2., 3., 3.5])
    K = np.exp(-0.5 * ((union[:, None] - union[None, :]) / 0.8) ** 2) + np.eye(4) * 1e-10
    rows = []
    residual = []
    for i, f in enumerate((first, second)):
        for block in range(2):
            for point, value in zip(np.asarray(f.f_scaled), np.asarray(observed[i][:, block] - model[i])):
                basis = np.zeros(4)
                basis[np.flatnonzero(union == point)[0]] = [1., rho][block]
                rows.append(basis)
                residual.append(value)
    Phi = np.asarray(rows)
    residual = np.asarray(residual)
    covariance = Phi @ K @ Phi.T + 0.04 * np.eye(len(residual))
    np.testing.assert_allclose(shared(model), _dense_log_prob(residual, covariance), rtol=1e-10)

    future = Frequency.from_f([1.5, 2.5], 'GHz')
    new_x = np.asarray(future.f_scaled)
    K_cross = np.exp(-0.5 * ((new_x[:, None] - union[None, :]) / 0.8) ** 2)
    K_new = np.exp(-0.5 * ((new_x[:, None] - new_x[None, :]) / 0.8) ** 2) + np.eye(2) * 1e-10
    alpha = np.linalg.solve(covariance, residual)
    expected_mean = K_cross @ Phi.T @ alpha
    expected_cov = K_new - K_cross @ Phi.T @ np.linalg.solve(covariance, Phi @ K_cross.T)
    latent = shared.predict_latents(model, future)
    np.testing.assert_allclose(latent.mean(), expected_mean, atol=1e-10)
    np.testing.assert_allclose(latent.covariance(), expected_cov, atol=1e-10)
    transfer = shared.predict_discrepancy(model, future, prediction_event=jnp.zeros((2, 2)))
    expected_map = np.kron(np.array([[1.], [rho]]), np.eye(2))
    np.testing.assert_allclose(transfer.mean(), expected_map @ expected_mean, atol=1e-10)
    np.testing.assert_allclose(transfer.covariance(), expected_map @ expected_cov @ expected_map.T, atol=1e-10)


def test_one_two_block_dataset_has_correlated_block_covariance():
    frequency = Frequency(1, 4, 4, 'GHz')
    model = jnp.array([0.2, 0.3, 0.4, 0.5])
    observed = jnp.stack((model + 0.1, model - 0.07), axis=-1)
    rho = 0.45
    evaluator = MarginalLogLikelihood(
        lambda candidate, f: jnp.stack((candidate, candidate), axis=-1),
        observed, GaussianLikelihood(0.03),
    )
    mixing = lambda prediction, x: jnp.broadcast_to(jnp.array([[1.], [rho]]), (len(x), 2, 1))
    shared = SharedMarginalLogLikelihood(((evaluator, frequency),),
                                         (SharedLatent(RBFKernel(0.8), mixing),))
    K = np.asarray(gram(RBFKernel(0.8), frequency.f_scaled, jitter=1e-10))
    B = np.array([[1., rho], [rho, rho ** 2]])
    covariance = np.kron(B, K) + 0.03 * np.eye(observed.size)
    residual = np.asarray(observed - jnp.stack((model, model), axis=-1)).T.reshape(-1)
    np.testing.assert_allclose(shared(model), _dense_log_prob(residual, covariance), rtol=1e-8)


def test_complex_power_mixing_and_compilation():
    frequency = Frequency(1, 3, 3, 'GHz')
    model = jnp.array([0.1 + 0.2j, -0.2 + 0.1j, 0.4 - 0.3j])
    observed = model + jnp.array([0.01 + 0.02j, 0.03 - 0.02j, -0.01 + 0.03j])
    evaluator = MarginalLogLikelihood(_predict, observed, GaussianLikelihood(0.01))
    shared = SharedMarginalLogLikelihood(
        ((evaluator, frequency),),
        (SharedLatent(TwoRowKernel(RBFKernel(0.7)), ComplexPowerMixing(1)),),
    )
    value = jax.jit(lambda m: shared(m))(model)
    assert np.isfinite(value)
    prediction = shared.predict_latents(model, frequency)
    discrepancy = shared.predict_discrepancy(model, frequency, dataset=0)
    assert prediction.mean().shape == (6,)
    assert discrepancy.mean().shape == (6,)
    np.testing.assert_allclose(jax.jit(lambda m: shared.predict_discrepancy(m, frequency, dataset=0).mean())(model),
                               discrepancy.mean())
    gradient = jax.grad(lambda real: shared(model + real))(0.)
    assert np.isfinite(gradient)
    latent_gradient = jax.grad(
        lambda real: jnp.sum(shared.predict_latents(model + real, frequency).mean())
    )(0.)
    discrepancy_gradient = jax.grad(
        lambda real: jnp.sum(shared.predict_discrepancy(model + real, frequency, dataset=0).mean())
    )(0.)
    assert np.isfinite(latent_gradient)
    assert np.isfinite(discrepancy_gradient)


def test_two_complex_latents_four_devices_match_dense_and_gradients():
    frequency = Frequency(1, 2, 6, 'GHz')
    x = np.asarray(frequency.f_scaled)
    rng = np.random.default_rng(311)
    truth = np.array([0.08, 0.3, 0.6, 0.9])
    phase = np.exp(0.3j * x)
    e00 = rng.normal(0, 0.02, 6) + 1j * rng.normal(0, 0.02, 6)
    et = rng.normal(0, 0.01, 6) + 1j * rng.normal(0, 0.01, 6)
    observations = tuple(jnp.asarray(t * phase + e00 + et * t * phase
                                     + rng.normal(0, 0.01, 6) + 1j * rng.normal(0, 0.01, 6))
                         for t in truth)
    model = jnp.asarray(truth + np.array([0.01, -0.02, 0.03, -0.04]))

    def build(lengthscale, noise):
        datasets = tuple((MarginalLogLikelihood(
            lambda candidate, f, i=i: candidate[i] * jnp.exp(0.3j * f.f_scaled),
            observations[i], GaussianLikelihood(noise)), frequency)
            for i in range(4))
        latents = (SharedLatent(TwoRowKernel(RBFKernel(lengthscale)), ComplexPowerMixing(0)),
                   SharedLatent(TwoRowKernel(RBFKernel(0.4)), ComplexPowerMixing(1)))
        return SharedMarginalLogLikelihood(datasets, latents)

    shared = build(0.5, 0.02)
    kernel = scipy.linalg.block_diag(*([np.asarray(gram(RBFKernel(0.5), jnp.asarray(x), jitter=1e-10))] * 2
                                         + [np.asarray(gram(RBFKernel(0.4), jnp.asarray(x), jitter=1e-10))] * 2))
    design = []
    residual = []
    for i in range(4):
        prediction = np.asarray(model[i] * phase)
        event_prediction = jnp.stack((jnp.real(prediction), jnp.imag(prediction)))
        mixing = np.concatenate([np.asarray(latent.mixing(event_prediction, jnp.asarray(x)))
                                 for latent in shared.latents], axis=-1)
        block_residual = observations[i] - prediction
        for block, values in enumerate((np.real(block_residual), np.imag(block_residual))):
            for point, value in enumerate(values):
                row = np.zeros(4 * len(x))
                for latent_row in range(4):
                    row[latent_row * len(x) + point] = mixing[point, block, latent_row]
                design.append(row)
                residual.append(value)
    design = np.asarray(design)
    residual = np.asarray(residual)
    dense = design @ kernel @ design.T + 0.02 * np.eye(len(residual))
    np.testing.assert_allclose(shared(model), _dense_log_prob(residual, dense), rtol=1e-8)

    def score(theta, lengthscale, noise):
        return build(lengthscale, noise)(theta)

    analytic = jax.grad(score, argnums=(0, 1, 2))(model, 0.5, 0.02)
    step = 1e-5
    for i in range(4):
        direction = jnp.eye(4)[i] * step
        finite = (score(model + direction, 0.5, 0.02) - score(model - direction, 0.5, 0.02)) / (2 * step)
        np.testing.assert_allclose(analytic[0][i], finite, rtol=1e-5)
    for arg, offset, derivative in ((0.5, 1, analytic[1]), (0.02, 2, analytic[2])):
        args = [model, 0.5, 0.02]
        args[offset] = arg + step
        upper = score(*args)
        args[offset] = arg - step
        lower = score(*args)
        np.testing.assert_allclose(derivative, (upper - lower) / (2 * step), rtol=1e-5)
    np.testing.assert_allclose(jax.jit(lambda theta: shared.predict_latents(theta, frequency).mean())(model),
                               shared.predict_latents(model, frequency).mean())


def _vna_case(n=200, seed=311):
    """Draw a reproducible four-device VNA calibration case."""
    frequency = Frequency(10, 500, n, 'MHz')
    x = np.asarray(frequency.f_scaled)
    theta = jnp.array([0.01, 0.2, 0.55, 0.9])
    phase = np.exp(0.4j * x / 500)
    rng = np.random.default_rng(seed)
    noise = 1e-4

    def kernel(lengthscale, variance):
        return Matern32Kernel(lengthscale) * ConstantKernel(jnp.array([variance, variance]))

    kernels = (kernel(80., 0.04 ** 2), kernel(130., 0.03 ** 2))
    row_covariances = [np.asarray(matrix) for latent_kernel in kernels
                       for matrix in gram(latent_kernel, jnp.asarray(x), jitter=1e-10)]
    draws = [np.linalg.cholesky(covariance) @ rng.standard_normal(n)
             for covariance in row_covariances]
    e00 = draws[0] + 1j * draws[1]
    et = draws[2] + 1j * draws[3]
    observations = tuple(jnp.asarray(t * phase + e00 + et * t * phase
                                     + rng.normal(0, np.sqrt(noise), n)
                                     + 1j * rng.normal(0, np.sqrt(noise), n))
                         for t in np.asarray(theta))

    def build(lengthscale=80., noise_variance=noise, directivity_variance=0.04 ** 2,
              tracking_lengthscale=130., tracking_variance=0.03 ** 2):
        datasets = tuple((MarginalLogLikelihood(
            lambda model, f, i=i: model[i] * jnp.exp(0.4j * f.f_scaled / 500),
            observations[i], GaussianLikelihood(noise_variance)), frequency)
            for i in range(4))
        latents = (SharedLatent(kernel(lengthscale, directivity_variance), ComplexPowerMixing(0)),
                   SharedLatent(kernel(tracking_lengthscale, tracking_variance), ComplexPowerMixing(1)))
        return SharedMarginalLogLikelihood(datasets, latents)

    return frequency, theta, observations, kernels, draws, noise, phase, build


def test_synthetic_vna_latents_and_likelihood_match_dense_conditioning():
    """Four VNA devices on 10–500 MHz share Matérn directivity and tracking error."""
    frequency, theta, observations, kernels, draws, noise, phase, build = _vna_case()
    x = np.asarray(frequency.f_scaled)
    row_covariances = [np.asarray(matrix) for latent_kernel in kernels
                       for matrix in gram(latent_kernel, jnp.asarray(x), jitter=1e-10)]

    shared = build()
    design = []
    residual = []
    for i, t in enumerate(np.asarray(theta)):
        prediction = t * phase
        difference = np.asarray(observations[i]) - prediction
        for block, values in enumerate((difference.real, difference.imag)):
            for point, value in enumerate(values):
                row = np.zeros(4 * len(x))
                row[block * len(x) + point] = 1.
                row[2 * len(x) + point] = prediction.real[point] if block == 0 else prediction.imag[point]
                row[3 * len(x) + point] = -prediction.imag[point] if block == 0 else prediction.real[point]
                design.append(row)
                residual.append(value)
    design, residual = np.asarray(design), np.asarray(residual)
    prior = scipy.linalg.block_diag(*row_covariances)
    dense = design @ prior @ design.T + noise * np.eye(len(residual))
    # Jitter is a diagonal stabilization at each grid, not cross-grid covariance.
    cross_prior = prior - 1e-10 * np.eye(len(prior))
    alpha = np.linalg.solve(dense, residual)
    expected_mean = cross_prior @ design.T @ alpha
    expected_covariance = prior - cross_prior @ design.T @ np.linalg.solve(dense, design @ cross_prior)
    np.testing.assert_allclose(shared(theta), _dense_log_prob(residual, dense), rtol=1e-8)
    posterior = shared.predict_latents(theta, frequency)
    np.testing.assert_allclose(posterior.mean(), expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(posterior.covariance(), expected_covariance, rtol=1e-10, atol=1e-10)
    drawn = np.concatenate(draws)
    assert np.sqrt(np.mean((np.asarray(posterior.mean()) - drawn) ** 2)) < (
        2 * np.sqrt(np.mean(np.diag(expected_covariance))))

    def score(model, lengthscale, noise_variance):
        return build(lengthscale, noise_variance)(model)

    derivatives = jax.grad(score, argnums=(0, 1, 2))(theta, 80., noise)
    finite_model = (score(theta.at[1].add(1e-5), 80., noise)
                    - score(theta.at[1].add(-1e-5), 80., noise)) / 2e-5
    finite_lengthscale = (score(theta, 80.01, noise) - score(theta, 79.99, noise)) / .02
    finite_noise = (score(theta, 80., noise + 1e-8) - score(theta, 80., noise - 1e-8)) / 2e-8
    np.testing.assert_allclose(derivatives[0][1], finite_model, rtol=1e-5)
    np.testing.assert_allclose(derivatives[1], finite_lengthscale, rtol=1e-5)
    np.testing.assert_allclose(derivatives[2], finite_noise, rtol=1e-5)


def test_synthetic_vna_on_partly_overlapping_grids_matches_dense():
    frequency, theta, observations, kernels, _, noise, _, _ = _vna_case(n=80)
    x = np.asarray(frequency.f_scaled)
    selections = (np.arange(0, 80, 2), np.arange(20, 80, 3))
    frequencies = tuple(Frequency.from_f(x[index], 'MHz') for index in selections)
    datasets = tuple((MarginalLogLikelihood(
        lambda model, f, i=i: model[i] * jnp.exp(0.4j * f.f_scaled / 500),
        observations[i][index], GaussianLikelihood(noise)), frequencies[i])
        for i, index in enumerate(selections))
    shared = SharedMarginalLogLikelihood(datasets, tuple(
        SharedLatent(kernel, ComplexPowerMixing(power))
        for kernel, power in zip(kernels, (0, 1))
    ))
    union = np.union1d(x[selections[0]], x[selections[1]])
    row_covariances = [np.asarray(matrix) for latent_kernel in kernels
                       for matrix in gram(latent_kernel, jnp.asarray(union), jitter=1e-10)]
    design, residual = [], []
    for i, selection in enumerate(selections):
        prediction = np.asarray(theta[i]) * np.exp(0.4j * x[selection] / 500)
        difference = np.asarray(observations[i][selection]) - prediction
        for block, values in enumerate((difference.real, difference.imag)):
            for point, value in enumerate(values):
                row = np.zeros(4 * len(union))
                union_point = np.searchsorted(union, x[selection[point]])
                row[block * len(union) + union_point] = 1.
                row[2 * len(union) + union_point] = prediction.real[point] if block == 0 else prediction.imag[point]
                row[3 * len(union) + union_point] = -prediction.imag[point] if block == 0 else prediction.real[point]
                design.append(row)
                residual.append(value)
    design, residual = np.asarray(design), np.asarray(residual)
    covariance = design @ scipy.linalg.block_diag(*row_covariances) @ design.T
    covariance = covariance + noise * np.eye(len(residual))
    np.testing.assert_allclose(shared(theta), _dense_log_prob(residual, covariance), rtol=1e-8)


def _minimize_jax(objective, start, bounds):
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def score(values):
        value, gradient = value_and_grad(jnp.asarray(values))
        return float(value), np.asarray(gradient, dtype=float)

    return minimize(score, start, jac=True, bounds=bounds, method='L-BFGS-B',
                    options={'maxiter': 100})


def test_synthetic_vna_type_ii_ml_recovers_hyperparameters():
    _, theta, _, _, _, noise, _, build = _vna_case()
    known = np.array([80., noise, 0.04 ** 2, 130., 0.03 ** 2])
    result = _minimize_jax(
        lambda log_values: -build(*jnp.exp(log_values))(theta),
        np.log(known),
        [(np.log(value / 3), np.log(value * 3)) for value in known],
    )
    assert result.success
    ratio = np.exp(result.x) / known
    # A single realization leaves the two latent amplitudes less identified
    # than the noise and correlation lengths; these bounds cover their spread.
    assert np.all((ratio > np.array([0.75, 0.8, 0.45, 0.75, 0.45]))
                  & (ratio < np.array([1.5, 1.2, 2., 1.5, 2.])))


def test_shared_vna_fit_improves_device_parameter_error():
    _, truth, _, _, _, _, _, build = _vna_case(n=80)
    shared = build()
    independent = tuple(SharedMarginalLogLikelihood((pair,), shared.latents)
                        for pair in shared.datasets)
    start = np.full(4, 0.4)
    bounds = [(0.001, 1.)] * 4
    def prior_penalty(theta):
        return 0.5 * jnp.sum(((theta - 0.45) / 0.5) ** 2)

    shared_fit = _minimize_jax(lambda theta: -shared(theta) + prior_penalty(theta),
                               start, bounds)
    independent_fit = _minimize_jax(
        lambda theta: -sum(term(theta) for term in independent) + prior_penalty(theta),
        start, bounds,
    )
    assert shared_fit.success and independent_fit.success
    shared_error = np.linalg.norm(shared_fit.x - np.asarray(truth))
    independent_error = np.linalg.norm(independent_fit.x - np.asarray(truth))
    assert shared_error <= independent_error


def test_twelve_dataset_scale_evaluates_value_and_gradient():
    frequency = Frequency(10, 500, 990, 'MHz')
    theta = jnp.linspace(0.01, 0.9, 12)
    datasets = []
    for i in range(12):
        predictor = lambda model, f, i=i: model[i] * jnp.exp(0.4j * f.f_scaled / 500)
        observed = np.asarray(predictor(theta, frequency)) + (
            0.003 + 0.002j) * np.sin(np.asarray(frequency.f_scaled) / 50 + i)
        datasets.append((MarginalLogLikelihood(
            predictor, observed, GaussianLikelihood(1e-4)), frequency))
    latents = (
        SharedLatent(Matern32Kernel(80.) * ConstantKernel(jnp.array([0.04 ** 2] * 2)),
                     ComplexPowerMixing(0)),
        SharedLatent(Matern32Kernel(130.) * ConstantKernel(jnp.array([0.03 ** 2] * 2)),
                     ComplexPowerMixing(1)),
    )
    shared = SharedMarginalLogLikelihood(tuple(datasets), latents)
    value, gradient = jax.jit(jax.value_and_grad(lambda model: shared(model)))(theta)
    value, gradient = jax.block_until_ready((value, gradient))
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))


def test_rejects_invalid_datasets_and_mixing():
    frequency = Frequency(1, 2, 2, 'GHz')
    evaluator = MarginalLogLikelihood(_predict, jnp.zeros(2), GaussianLikelihood(0.1))
    latent = SharedLatent(RBFKernel(1.), IdentityMixing())
    with pytest.raises(ValueError, match='local or orthogonal'):
        with_discrepancy = MarginalLogLikelihood(_predict, jnp.zeros(2), GaussianLikelihood(0.1),
                                                GaussianProcess(RBFKernel(1.)))
        SharedMarginalLogLikelihood(((with_discrepancy, frequency),), (latent,))
    invalid = SharedLatent(RBFKernel(1.), lambda prediction, x: jnp.zeros((len(x), 2, 2)))
    with pytest.raises(ValueError, match='Mixing returned shape'):
        SharedMarginalLogLikelihood(((evaluator, frequency),), (invalid,))(jnp.zeros(2))


def test_equinox_mixing_parameter_contributes_to_gradient():
    class GainMixing(eqx.Module):
        gain: jnp.ndarray

        def __call__(self, prediction_event, x):
            return jnp.broadcast_to(self.gain, (len(x), 1, 1))

    frequency = Frequency(1, 3, 3, 'GHz')
    evaluator = MarginalLogLikelihood(_predict, jnp.array([0.1, 0.2, 0.3]),
                                      GaussianLikelihood(0.05))

    def score(gain):
        shared = SharedMarginalLogLikelihood(
            ((evaluator, frequency),),
            (SharedLatent(RBFKernel(0.8), GainMixing(gain)),),
        )
        return shared(jnp.zeros(3))

    derivative = jax.grad(score)(jnp.array(1.2))
    finite = (score(1.20001) - score(1.19999)) / 2e-5
    np.testing.assert_allclose(derivative, finite, rtol=1e-5)


def test_conditional_transform_jacobian_is_added_once():
    frequency = Frequency(1, 3, 3, 'GHz')
    model = jnp.array([0.1, 0.2, 0.3])
    observed = jnp.array([0.2, 0.1, 0.4])
    affine = bij.ScalarAffine(shift=np.zeros(3), scale=np.full(3, 2.))
    static = MarginalLogLikelihood(_predict, observed, GaussianLikelihood(0.03),
                                   event_transform=affine)
    conditional = MarginalLogLikelihood(_predict, observed, GaussianLikelihood(0.03),
                                        event_transform=lambda prediction: affine)
    latent = SharedLatent(RBFKernel(0.7), IdentityMixing())
    baseline = SharedMarginalLogLikelihood(((static, frequency),), (latent,))
    transformed = SharedMarginalLogLikelihood(((conditional, frequency),), (latent,))
    np.testing.assert_allclose(transformed(model) - baseline(model), 3 * np.log(2.), atol=1e-12)
