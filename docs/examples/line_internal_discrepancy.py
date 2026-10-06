"""Transfer a line's internal discrepancy between physical lengths."""

import jax.numpy as jnp
import numpy as np

import pmrf as prf
from pmrf.covariance_kernels import RBFKernel
from pmrf.discrepancy_models import GaussianProcess
from pmrf.distributions import Normal
from pmrf.evaluators import MarginalLogLikelihood
from pmrf.likelihoods import GaussianLikelihood
from pmrf.linearization import posterior_covariance
from pmrf.materials import BulkConductor, ConstantDielectric
from pmrf.models import (
    BasisLineDiscrepancy, CoaxialLine, GridLineDiscrepancy,
    GridPortDiscrepancy, LineCorrected, PortCorrected,
    TescheCoaxialFormulation, line_internal_features, project_line_basis_joint,
    reference_line_features,
)
from pmrf.optimize import ScipyMinimize, minimize


TRUTH_SIGMA = 1e3
MODEL_SIGMA = TRUTH_SIGMA / 1.1**2


def frequency():
    """The small positive-frequency band used by all three recipes."""
    return prf.Frequency(1, 2, 7, 'GHz')


def line(length, sigma, *, free_sigma=False):
    """A lossy coax whose conductor is 10% too resistive when modelled."""
    sigma_parameter = (prf.Random(Normal(float(sigma), 1.0), value=sigma)
                       if free_sigma else prf.Fixed(sigma))
    return CoaxialLine(
        length=prf.Fixed(length), d_in=prf.Fixed(1.12e-3),
        d_out=prf.Fixed(3.2e-3),
        dielectric=ConstantDielectric(ep_r=prf.Fixed(1.5), tand=prf.Fixed(0.0),
                                      sigma=prf.Fixed(0.0), mu_r=prf.Fixed(1.0)),
        conductor=BulkConductor(sigma=sigma_parameter, mu_r=prf.Fixed(1.0)),
        formulation=TescheCoaxialFormulation(),
    )


def exact_values(f, reference_length=0.30):
    """Reference log ratios of characteristic impedance, loss and phase."""
    z_truth, g_truth = line(reference_length, TRUTH_SIGMA).zc_and_gammaL(f)
    z_model, g_model = line(reference_length, MODEL_SIGMA).zc_and_gammaL(f)
    z = jnp.log(z_truth / z_model)
    return jnp.stack((jnp.stack((z.real, z.imag)),
                      jnp.stack((jnp.log(g_truth.real / g_model.real),
                                 jnp.log(g_truth.imag / g_model.imag)))))


def _rms_s21(model, truth, f):
    return float(np.sqrt(np.mean(np.abs(np.asarray(model.s(f) - truth.s(f))[:, 1, 0])**2)))


def _eta(model, f):
    s = np.asarray(model.s(f))
    return np.abs(s[:, 1, 0])**2 / (1 - np.abs(s[:, 0, 0])**2)


def compare_transfer():
    """Compare uncorrected, transferred port and transferred internal errors."""
    f = frequency()
    truth_ref, model_ref = line(0.30, TRUTH_SIGMA), line(0.30, MODEL_SIGMA)
    truth_new, model_new = line(0.15, TRUTH_SIGMA), line(0.15, MODEL_SIGMA)
    s_truth, s_model = np.asarray(truth_ref.s(f)), np.asarray(model_ref.s(f))
    reflection = s_truth[:, 0, 0] - s_model[:, 0, 0]
    transmission = np.log(s_truth[:, 1, 0] / s_model[:, 1, 0])
    port_values = np.stack((np.stack((reflection.real, reflection.imag)),
                            np.stack((reflection.real, reflection.imag)),
                            np.stack((transmission.real, transmission.imag))))
    port = PortCorrected(model_new, GridPortDiscrepancy(
        port_values, f, ('11', '22', 's21')))
    internal = LineCorrected(model_new, GridLineDiscrepancy(exact_values(f), f))
    models = {'uncorrected': model_new, 'port': port, 'internal': internal}
    return {
        name: {'rms_s21': _rms_s21(candidate, truth_new, f),
               'eta_bias': float(np.mean(_eta(candidate, f) - _eta(truth_new, f)))}
        for name, candidate in models.items()
    }


def reference_route():
    r"""Learn from reference $Z_c$ and $\gamma$ and transfer a joint prediction."""
    f = frequency()
    reference = line(0.30, TRUTH_SIGMA)
    zc, gamma_length = reference.zc_and_gammaL(f)
    observed = reference_line_features(zc, gamma_length / 0.30)
    nominal = line(0.30, MODEL_SIGMA, free_sigma=True)
    mll = MarginalLogLikelihood(
        predictor=line_internal_features, observed=observed,
        likelihood=GaussianLikelihood(1e-10),
        discrepancy=GaussianProcess(RBFKernel(0.5) * 0.1**2, jitter=1e-12),
    )
    names, joint = mll.predict_joint(nominal, f, f)
    carrier = GridLineDiscrepancy.from_prediction(f, joint, joint=True)
    transfer = LineCorrected(line(0.15, MODEL_SIGMA, free_sigma=True), carrier)
    transfer = prf.prior(transfer, [*names, 'discrepancy.values'], joint,
                         truncate='unnormalised')
    return {'rms_s21': _rms_s21(transfer, line(0.15, TRUTH_SIGMA), f),
            'log_prior': float(prf.log_prior(transfer)), 'carrier': carrier}


def _basis(f):
    lengthscale = 0.3e9
    amplitude = 0.1
    spectral_density = lambda omega: (amplitude**2 * jnp.sqrt(2 * jnp.pi)
                                      * lengthscale * jnp.exp(-0.5 * (lengthscale * omega)**2))
    return BasisLineDiscrepancy.from_kernel(
        RBFKernel(lengthscale) * amplitude**2, spectral_density,
        (0.0, 3e9), f, variance_tolerance=0.05, max_rank=24,
    )


def s_only_route():
    """Fit complex S, report 2σ coverage and transfer the Laplace mean."""
    f = frequency()
    basis = _basis(f)
    nominal = LineCorrected(line(0.30, MODEL_SIGMA), basis)
    observed = line(0.30, TRUTH_SIGMA).s(f)
    mll = MarginalLogLikelihood(
        predictor=lambda model, grid: model.s(grid), observed=observed,
        likelihood=GaussianLikelihood(1e-8),
    )
    objective = lambda model: -mll(model, f) - prf.log_prior(model)
    fitted = minimize(objective, nominal, solver=ScipyMinimize(method='BFGS'),
                      max_iter=400).model
    fitted_basis = prf.update(
        basis, 'coefficients', value=prf.values(fitted)['discrepancy.coefficients'])
    linearization = mll.linearize(fitted, f)
    covariance = posterior_covariance([linearization], fitted)
    names, joint = project_line_basis_joint(fitted, fitted_basis, linearization,
                                            covariance, f)
    carrier = GridLineDiscrepancy.from_prediction(f, joint, joint=True)
    mean = np.asarray(joint.mean()).reshape(2, 2, f.npoints)
    sigma = np.sqrt(np.diag(np.asarray(joint.covariance())).reshape(2, 2, f.npoints))
    coverage = float(np.mean(np.abs(np.asarray(exact_values(f)) - mean) <= 2 * sigma))
    transfer = LineCorrected(line(0.15, MODEL_SIGMA), carrier)
    return {'rms_s21': _rms_s21(transfer, line(0.15, TRUTH_SIGMA), f),
            'coverage': coverage, 'mean': mean, 'sigma': sigma,
            'injected': np.asarray(exact_values(f)), 'names': names, 'carrier': carrier}


if __name__ == '__main__':
    print('Comparison:', compare_transfer())
    print('Reference route:', reference_route()['rms_s21'])
    result = s_only_route()
    print('S-only route:', result['rms_s21'], 'coverage:', result['coverage'])
