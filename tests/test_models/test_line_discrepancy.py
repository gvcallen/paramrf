import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

import pmrf as prf
from pmrf.models import (BasisLineDiscrepancy, GridLineDiscrepancy, LineCorrected,
                         RLGCLine, line_internal_features, reference_line_features)
from pmrf.objectives.evaluators import MarginalLogLikelihood
from pmrf.stats.likelihoods import GaussianLikelihood
from pmrf.stats.discrepancy_models import GaussianProcess
from pmrf.stats.covariance_kernels import RBFKernel
from pmrf.models import Circuit, CoaxialLine, MicrostripLine, Port, PortCorrected, GridPortDiscrepancy
from pmrf.materials import BulkConductor
from pmrf.stats.distributions import Normal
from pmrf.stats.linearization import posterior_covariance
from pmrf.optimize import minimize, ScipyMinimize
from pmrf.models import project_line_basis_joint


FREQUENCY = prf.Frequency(1, 2, 5, 'GHz')


def test_zero_values_reproduce_uniform_line_and_keep_parameter_names():
    line = RLGCLine(length=0.3, R=1.0, name='line')
    corrected = LineCorrected(line, GridLineDiscrepancy.zeros(FREQUENCY))
    np.testing.assert_array_equal(corrected.s(FREQUENCY), line.s(FREQUENCY))
    np.testing.assert_allclose(corrected.y(FREQUENCY), line.y(FREQUENCY), rtol=1e-15, atol=1e-15)
    assert 'length' in prf.values(corrected)
    assert 'discrepancy.values' in prf.values(corrected)
    updated = prf.update(corrected, {'discrepancy.values': jnp.ones((2, 2, 5)) * 0.01})
    assert not np.allclose(updated.s(FREQUENCY), line.s(FREQUENCY))


@pytest.mark.parametrize('line', [CoaxialLine(length=0.3, name='line'),
                                  MicrostripLine(length=0.3, name='line')])
def test_zero_values_reproduce_physical_lines(line):
    corrected = LineCorrected(line, GridLineDiscrepancy.zeros(FREQUENCY))
    np.testing.assert_allclose(corrected.s(FREQUENCY), line.s(FREQUENCY), rtol=1e-14, atol=1e-14)


def test_log_correction_scales_attenuation_and_phase_separately():
    line = RLGCLine(length=0.3, R=1.0)
    values = np.zeros((2, 2, 5))
    values[0, 0] = np.log(1.02)
    values[1, 0] = np.log(1.1)
    values[1, 1] = np.log(0.9)
    corrected = LineCorrected(line, GridLineDiscrepancy(values, FREQUENCY)).build()
    zc, gamma_length = line.zc_and_gammaL(FREQUENCY)
    got_zc, got_gamma_length = corrected.zc_and_gammaL(FREQUENCY)
    np.testing.assert_allclose(got_zc, 1.02 * zc, rtol=1e-14)
    np.testing.assert_allclose(got_gamma_length.real, 1.1 * gamma_length.real, rtol=1e-14)
    np.testing.assert_allclose(got_gamma_length.imag, 0.9 * gamma_length.imag, rtol=1e-14)


@pytest.mark.parametrize('compiled', [False, True])
def test_different_grid_raises(compiled):
    discrepancy = GridLineDiscrepancy.zeros(FREQUENCY)
    other = prf.Frequency(1, 3, 5, 'GHz')
    evaluate = eqx.filter_jit(lambda f: discrepancy(f)) if compiled else discrepancy
    with pytest.raises(Exception, match='Line discrepancy requires its own frequency grid'):
        jax.block_until_ready(evaluate(other))


def test_circuit_outer_port_correction_jit_derivative_and_names():
    line = LineCorrected(RLGCLine(length=0.3, R=1.0, name='line'),
                         GridLineDiscrepancy.zeros(FREQUENCY))
    left, right = Port(name='left'), Port(name='right')
    circuit = Circuit([[(left, 0), (line, 0)], [(line, 1), (right, 0)]])
    assert 'line.length' in prf.values(circuit)
    assert 'line.discrepancy.values' in prf.values(circuit)
    port = PortCorrected(circuit, GridPortDiscrepancy.zeros(FREQUENCY, ('s21',)))
    np.testing.assert_allclose(jax.jit(lambda: port.s(FREQUENCY))(), port.s(FREQUENCY), rtol=1e-12, atol=1e-12)
    gradient, = prf.derivative(lambda m: jnp.sum(jnp.abs(m.s(FREQUENCY))**2), port)
    assert np.all(np.isfinite(prf.values(gradient)['line.discrepancy.values']))


def test_internal_correction_transfers_between_lengths_where_port_correction_does_not():
    frequency = prf.Frequency(1, 2, 7, 'GHz')
    # This intentionally lossy conductor makes the doubled port attenuation
    # at half length larger than the original 10% resistance error.
    sigma = 1e3
    truth_ref = CoaxialLine(length=0.30, conductor=BulkConductor(sigma=sigma))
    model_ref = CoaxialLine(length=0.30, conductor=BulkConductor(sigma=sigma / 1.1**2))
    truth_new = CoaxialLine(length=0.15, conductor=BulkConductor(sigma=sigma))
    model_new = CoaxialLine(length=0.15, conductor=BulkConductor(sigma=sigma / 1.1**2))
    z_truth, g_truth = truth_ref.zc_and_gammaL(frequency)
    z_model, g_model = model_ref.zc_and_gammaL(frequency)
    z_ratio = np.log(np.asarray(z_truth / z_model))
    values = np.stack((np.stack((z_ratio.real, z_ratio.imag)),
                       np.stack((np.log(np.asarray(g_truth.real / g_model.real)),
                                 np.log(np.asarray(g_truth.imag / g_model.imag))))))
    corrected = LineCorrected(model_new, GridLineDiscrepancy(values, frequency))
    np.testing.assert_allclose(corrected.s(frequency), truth_new.s(frequency), rtol=1e-12, atol=1e-12)

    s_truth, s_model = np.asarray(truth_ref.s(frequency)), np.asarray(model_ref.s(frequency))
    port_values = np.stack((np.stack(((s_truth[:, 0, 0] - s_model[:, 0, 0]).real,
                                      (s_truth[:, 0, 0] - s_model[:, 0, 0]).imag)),
                            np.stack(((s_truth[:, 1, 1] - s_model[:, 1, 1]).real,
                                      (s_truth[:, 1, 1] - s_model[:, 1, 1]).imag)),
                            np.stack((np.log(s_truth[:, 1, 0] / s_model[:, 1, 0]).real,
                                      np.log(s_truth[:, 1, 0] / s_model[:, 1, 0]).imag))))
    port_corrected = PortCorrected(model_new, GridPortDiscrepancy(
        port_values, frequency, ('11', '22', 's21')))
    target = np.abs(np.asarray(truth_new.s(frequency))[:, 1, 0])
    uncorrected_error = np.mean(np.abs(np.abs(np.asarray(model_new.s(frequency))[:, 1, 0]) - target))
    port_error = np.mean(np.abs(np.abs(np.asarray(port_corrected.s(frequency))[:, 1, 0]) - target))
    assert port_error > uncorrected_error


def test_reference_features_and_prediction_map_to_carrier():
    frequency = prf.Frequency(1, 2, 7, 'GHz')
    line = RLGCLine(length=0.3, R=1.0)
    zc, gamma_length = line.zc_and_gammaL(frequency)
    np.testing.assert_allclose(
        line_internal_features(line, frequency),
        reference_line_features(zc, gamma_length / 0.3), rtol=1e-14,
    )
    injected = np.zeros((2, 2, 7))
    injected[0, 0] = 0.005
    injected[1, 0] = -0.01
    injected[1, 1] = 0.003
    truth = LineCorrected(line, GridLineDiscrepancy(injected, frequency)).build()
    observed_zc, observed_gamma_length = truth.zc_and_gammaL(frequency)
    observed = reference_line_features(observed_zc, observed_gamma_length / 0.3)
    mll = MarginalLogLikelihood(
        predictor=line_internal_features, observed=observed,
        likelihood=GaussianLikelihood(1e-10),
        discrepancy=GaussianProcess(RBFKernel(1e9) * 1e-3, jitter=1e-12),
    )
    assert np.isfinite(mll(line, frequency))
    prediction = mll.predict_discrepancy(line, frequency, frequency)
    carrier = GridLineDiscrepancy.from_prediction(frequency, prediction)
    np.testing.assert_array_equal(prf.values(carrier)['values'], prediction.mean())
    assert prf.values(carrier)['values'].shape == (2, 2, 7)


def test_reference_joint_prediction_attaches_to_transfer_line_by_name():
    frequency = prf.Frequency(1, 2, 5, 'GHz')
    line = RLGCLine(length=0.3, R=prf.Random(Normal(1.0, 0.1), value=1.0))
    injected = np.zeros((2, 2, frequency.npoints))
    injected[0, 0] = 0.003
    injected[1, 0] = -0.005
    injected[1, 1] = 0.002
    zc, gamma_length = LineCorrected(
        line, GridLineDiscrepancy(injected, frequency)).build().zc_and_gammaL(frequency)
    observed = reference_line_features(zc, gamma_length / 0.3)
    mll = MarginalLogLikelihood(
        predictor=line_internal_features, observed=observed,
        likelihood=GaussianLikelihood(1e-8),
        discrepancy=GaussianProcess(RBFKernel(1e9) * 1e-4, jitter=1e-10),
    )
    names, joint = mll.predict_joint(line, frequency, frequency)
    assert names == ('R',)
    assert np.isfinite(jax.grad(lambda resistance: mll(
        prf.update(line, 'R', value=resistance), frequency))(1.0))
    mean = np.asarray(joint.mean())[1:].reshape(injected.shape)
    sigma = np.sqrt(np.diag(np.asarray(joint.covariance()))[1:].reshape(injected.shape))
    assert np.all(np.abs(mean - injected) <= 2 * sigma)
    carrier = GridLineDiscrepancy.from_prediction(frequency, joint, joint=True)
    transfer = LineCorrected(RLGCLine(length=0.15, R=prf.Unconstrained(1.0)), carrier)
    transfer = prf.prior(transfer, [*names, 'discrepancy.values'], joint,
                         truncate='unnormalised')
    assert np.isfinite(prf.log_prior(transfer))
    truth_transfer = LineCorrected(RLGCLine(length=0.15, R=1.0),
                                   GridLineDiscrepancy(injected, frequency))
    assert np.max(np.abs(np.asarray(transfer.s(frequency) - truth_transfer.s(frequency)))) < 10 * np.max(sigma)


def _basis(frequency):
    lengthscale = 0.3e9
    spectrum = lambda omega: jnp.sqrt(2 * jnp.pi) * lengthscale * jnp.exp(-0.5 * (lengthscale * omega)**2)
    return BasisLineDiscrepancy.from_kernel(
        RBFKernel(lengthscale), spectrum, (0.0, 3e9), frequency,
        variance_tolerance=0.05, max_rank=24,
    )


def test_basis_variance_domain_and_dense_carrier():
    frequency = prf.Frequency(1, 2, 7, 'GHz')
    basis = _basis(frequency)
    variance = np.sum(np.asarray(basis.basis(frequency))**2, axis=-1)
    assert np.max(np.abs(variance - 1)) <= 0.05
    coefficients = np.zeros(prf.values(basis)['coefficients'].shape)
    coefficients[0, 0, 0] = 0.2
    basis = prf.update(basis, 'coefficients', value=coefficients)
    dense = basis.materialize(frequency)
    np.testing.assert_array_equal(dense(frequency), basis(frequency))
    with pytest.raises(Exception, match='outside the line discrepancy basis domain'):
        basis(prf.Frequency(2.9, 3.1, 3, 'GHz'))


def test_s_only_map_and_laplace_projection_transfer():
    frequency = prf.Frequency(1, 2, 7, 'GHz')
    basis = _basis(frequency)
    coefficients = np.zeros(prf.values(basis)['coefficients'].shape)
    coefficients[:, :, 0] = [[0.006, -0.004], [-0.02, 0.008]]
    truth_basis = prf.update(basis, 'coefficients', value=coefficients)

    def circuit(discrepancy, length):
        line = LineCorrected(
            RLGCLine(length=prf.Fixed(length), R=prf.Fixed(3.0),
                     L=prf.Fixed(2.8e-7), G=prf.Fixed(0.0),
                     C=prf.Fixed(9e-11), name='line'), discrepancy)
        left, right = Port(name='left'), Port(name='right')
        return Circuit([[(left, 0), (line, 0)], [(line, 1), (right, 0)]])

    truth = circuit(truth_basis, 0.30)
    initial = circuit(basis, 0.30)
    observed = truth.s(frequency)
    mll = MarginalLogLikelihood(
        predictor=lambda model, f: model.s(f), observed=observed,
        likelihood=GaussianLikelihood(1e-6),
    )
    objective = lambda model: -mll(model, frequency) - prf.log_prior(model)
    assert np.isfinite(objective(initial))
    assert np.all(np.isfinite(prf.values(prf.derivative(objective, initial)[0])['line.discrepancy.coefficients']))
    fitted = minimize(objective, initial, solver=ScipyMinimize(method='L-BFGS-B'), max_iter=1000).model
    assert np.isfinite(objective(fitted))
    assert objective(fitted) < objective(initial)

    fitted_basis = prf.update(
        basis, 'coefficients', value=prf.values(fitted)['line.discrepancy.coefficients'])
    linearization = mll.linearize(fitted, frequency)
    covariance = posterior_covariance([linearization], fitted)
    names, joint = project_line_basis_joint(fitted, fitted_basis, linearization, covariance,
                                            frequency, discrepancy_name='line.discrepancy.coefficients')
    assert names == ()
    mean = np.asarray(joint.mean()).reshape(2, 2, frequency.npoints)
    sigma = np.sqrt(np.diag(np.asarray(joint.covariance())).reshape(2, 2, frequency.npoints))
    injected = np.asarray(truth_basis(frequency))
    assert np.mean(np.abs(injected - mean) <= 2 * sigma) >= 0.95
    assert np.all(np.isfinite(sigma)) and np.all(sigma >= 0)

    transfer_frequency = frequency
    carrier = GridLineDiscrepancy.from_prediction(transfer_frequency, joint, joint=True)
    map_carrier = fitted_basis.materialize(transfer_frequency)
    np.testing.assert_allclose(prf.values(carrier)['values'], prf.values(map_carrier)['values'], atol=1e-12)
    dense_map = circuit(map_carrier, 0.30)
    np.testing.assert_allclose(dense_map.s(frequency), fitted.s(frequency), rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(np.asarray(fitted.s(frequency) - observed)) < np.linalg.norm(
        np.asarray(initial.s(frequency) - observed))
    truth_transfer = circuit(truth_basis, 0.15)
    raw_transfer = circuit(basis, 0.15)
    corrected_transfer = circuit(carrier, 0.15)
    target = np.asarray(truth_transfer.s(frequency))[:, 1, 0]
    assert np.mean(np.abs(np.asarray(corrected_transfer.s(frequency))[:, 1, 0] - target)) < np.mean(
        np.abs(np.asarray(raw_transfer.s(frequency))[:, 1, 0] - target))


def test_basis_joint_projection_keeps_free_line_parameter_covariance():
    frequency = prf.Frequency(1, 2, 5, 'GHz')
    basis = _basis(frequency)
    line = RLGCLine(length=prf.Fixed(0.30), R=prf.Random(Normal(3.0, 0.5), value=3.0),
                    L=prf.Fixed(2.8e-7), G=prf.Fixed(0.0), C=prf.Fixed(9e-11))
    model = LineCorrected(line, basis)
    mll = MarginalLogLikelihood(
        predictor=lambda m, f: m.s(f), observed=model.s(frequency),
        likelihood=GaussianLikelihood(1e-8),
    )
    linearization = mll.linearize(model, frequency)
    covariance = posterior_covariance([linearization], model)
    names, joint = project_line_basis_joint(model, basis, linearization, covariance,
                                            frequency)
    assert names == ('R',)
    assert np.all(np.isfinite(joint.covariance()))
    assert np.any(np.abs(np.asarray(joint.covariance())[0, 1:]) > 0)
    carrier = GridLineDiscrepancy.from_prediction(frequency, joint, joint=True)
    transfer = LineCorrected(RLGCLine(length=prf.Fixed(0.15), R=prf.Unconstrained(3.0),
                                      L=prf.Fixed(2.8e-7), G=prf.Fixed(0.0),
                                      C=prf.Fixed(9e-11)), carrier)
    transfer = prf.prior(transfer, [*names, 'discrepancy.values'], joint,
                         truncate='unnormalised')
    assert np.isfinite(prf.log_prior(transfer))
