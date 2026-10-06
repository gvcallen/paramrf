"""Port discrepancy from a reference two-port to a reflection-only transfer fit.

Run with ``python docs/examples/port_discrepancy.py --n-frequency 1000``.
"""
import argparse
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from distreqx.bijectors import Lambda
from distreqx.distributions import Normal
from scipy.optimize import minimize

import pmrf as prf
from pmrf.covariance_kernels import Matern52Kernel
from pmrf.discrepancy_models import GaussianProcess
from pmrf.evaluators import MarginalLogLikelihood
from pmrf.likelihoods import GaussianLikelihood
from pmrf.linearization import posterior_covariance
from pmrf.models import GridPortDiscrepancy, Load, PortCorrected, RLGCLine, SModel

jax.config.update('jax_enable_x64', True)


def line(resistance):
    return RLGCLine(length=10.0, R=resistance, L=250e-9, C=100e-12, G=1e-6)


@prf.derived
def fitted_line(base, *, log_R):
    return prf.replace(base, R=jnp.exp(log_R))


def truth(frequency):
    f = frequency.f_scaled
    # Skin-effect R and a small inductive dispersion absent from the fit model.
    dispersive = RLGCLine(length=10.0, R=0.5 * jnp.sqrt(f / 250),
                          L=250e-9 * (1 + 0.0005 * jnp.log(f / 250)), C=100e-12, G=1e-6)
    return SModel(dispersive.s(frequency), frequency, 50.0)


def port_event_transform(predicted):
    """Invertible full two-port transform; frequency is the last event axis."""
    anchor = jnp.unwrap(jnp.angle(predicted), axis=0)

    def forward(s):
        phase = anchor + jnp.unwrap(jnp.angle(s / predicted), axis=0)
        logs = jnp.log(jnp.abs(s)) + 1j * phase
        blocks = jnp.stack((s[:, 0, 0], s[:, 1, 1],
                            (logs[:, 1, 0] + logs[:, 0, 1]) / 2,
                            (logs[:, 1, 0] - logs[:, 0, 1]) / 2))
        return jnp.stack((blocks.real, blocks.imag), axis=1)

    def inverse(event):
        blocks = event[:, 0] + 1j * event[:, 1]
        return jnp.stack((jnp.stack((blocks[0], jnp.exp(blocks[2] - blocks[3])), axis=-1),
                          jnp.stack((jnp.exp(blocks[2] + blocks[3]), blocks[1]), axis=-1)), axis=-2)

    def logdet(s):
        # The mean/half-difference rotation has determinant 1/2 for each Re/Im pair.
        return -2 * jnp.log(jnp.abs(s[:, 0, 1])) - 2 * jnp.log(jnp.abs(s[:, 1, 0])) - 2 * jnp.log(2.0)

    return Lambda(forward=forward, inverse=inverse,
                  forward_log_det_jacobian=logdet,
                  inverse_log_det_jacobian=lambda e: -logdet(inverse(e)),
                  is_constant_jacobian=False)


def reciprocal_event_transform(predicted):
    """Transform the retained reflections and symmetric transmission only."""
    anchor = jnp.unwrap(jnp.angle(predicted[:, 2]))

    def forward(s):
        log_t = jnp.log(jnp.abs(s[:, 2])) + 1j * (
            anchor + jnp.unwrap(jnp.angle(s[:, 2] / predicted[:, 2])))
        blocks = jnp.stack((s[:, 0], s[:, 1], log_t))
        return jnp.stack((blocks.real, blocks.imag), axis=1)

    def inverse(event):
        blocks = event[:, 0] + 1j * event[:, 1]
        return jnp.stack((blocks[0], blocks[1], jnp.exp(blocks[2])), axis=-1)

    logdet = lambda s: -2 * jnp.log(jnp.abs(s[:, 2]))
    return Lambda(forward=forward, inverse=inverse,
                  forward_log_det_jacobian=logdet,
                  inverse_log_det_jacobian=lambda e: -logdet(inverse(e)),
                  is_constant_jacobian=False)


def reciprocal_predictor(model, frequency):
    s = model.s(frequency)
    return jnp.stack((s[:, 0, 0], s[:, 1, 1], s[:, 1, 0]), axis=-1)


def reference_fit(mll, model, frequency):
    """Fit the reference line's one log-resistance coordinate."""
    def loss(vector):
        candidate = prf.update(model, {'log_R': vector[0]})
        return -mll(candidate, frequency) - prf.log_prior(candidate)

    value_gradient = jax.jit(jax.value_and_grad(loss))
    result = minimize(lambda x: tuple(np.asarray(v) for v in value_gradient(jnp.asarray(x))),
                      np.array([prf.values(model)['log_R']]), jac=True, method='L-BFGS-B',
                      bounds=[(np.log(0.05), np.log(2.0))],
                      options={'gtol': 1e-6, 'ftol': 1e-14, 'maxiter': 200})
    if not np.isfinite(result.fun) or np.linalg.norm(result.jac, ord=np.inf) > 1e-3:
        raise RuntimeError(f'MAP fit failed: {result.message}; gradient {result.jac}')
    return prf.update(model, {'log_R': jnp.asarray(result.x[0])})


def run(n_frequency=1000, n_transfer=None, mc_samples=10000):
    """Run the complete recipe, returning diagnostics and synchronized timings."""
    n_transfer = n_frequency if n_transfer is None else n_transfer
    timings = {}
    before = perf_counter()

    def finish(name, value):
        nonlocal before
        jax.block_until_ready(value)
        now = perf_counter()
        timings[name] = now - before
        before = now
        print(f'{name}: {timings[name]:.3f} s', flush=True)

    frequency = prf.Frequency(10, 500, n_frequency, 'MHz')
    transfer_frequency = prf.Frequency(10.1, 499.9, n_transfer, 'MHz')
    true_s = truth(frequency).s(frequency)
    rng = np.random.default_rng(265)
    sigma = 0.002
    observed = true_s + sigma * (rng.normal(size=true_s.shape) + 1j * rng.normal(size=true_s.shape))
    initial = fitted_line(line(0.5), log_R=prf.Unconstrained(jnp.log(0.5)))
    initial = prf.prior(initial, 'log_R', Normal(jnp.log(0.5), 0.02))
    full_event = port_event_transform(initial.s(frequency)).forward(observed)
    # Discard only the antisymmetric block, then encode the symmetric log in S space.
    retained = full_event[:3, 0] + 1j * full_event[:3, 1]
    reciprocal_observed = jnp.stack((retained[0], retained[1], jnp.exp(retained[2])), axis=-1)
    variance = jnp.ones((3, 2, n_frequency)) * sigma**2
    variance = variance.at[2].set(sigma**2 / (2 * jnp.abs(reciprocal_observed[:, 2])**2))
    reference = MarginalLogLikelihood(
        predictor=reciprocal_predictor, observed=reciprocal_observed,
        likelihood=GaussianLikelihood(variance),
        discrepancy=GaussianProcess(Matern52Kernel(100.0) * 0.05**2, jitter=1e-8),
        event_transform=reciprocal_event_transform,
    )
    finish('synthetic data and event transform', reciprocal_observed)
    fitted = reference_fit(reference, initial, frequency)
    finish('reference MAP', fitted.s(frequency))
    event = reciprocal_event_transform(reciprocal_predictor(fitted, frequency))
    log_residual = event.forward(reciprocal_observed)[2] - event.forward(reciprocal_predictor(fitted, frequency))[2]
    additive_residual = true_s[:, 1, 0] - fitted.s(frequency)[:, 1, 0]
    names, joint = reference.predict_joint(fitted, frequency, transfer_frequency)
    finish('joint prediction', joint.covariance())
    corrected = PortCorrected(fitted, GridPortDiscrepancy(
        prf.Unconstrained(joint.mean()[1:].reshape(3, 2, n_transfer)),
        transfer_frequency, ('11', '22', 's21')))
    corrected = prf.prior(corrected, [*names, 'discrepancy.values'], joint)
    finish('joint prior attachment', prf.log_prior(corrected))
    # Keep the cable as the base: its parameter names and joint prior survive.
    @prf.derived
    def terminated(cable, *, load):
        return cable.terminated(Load(gamma=(load - 50) / (load + 50)))

    transfer = terminated(corrected, load=prf.Unconstrained(75.0))
    true_reflection = truth(transfer_frequency).terminated(Load(gamma=(76.0 - 50) / (76.0 + 50))).s(transfer_frequency)[:, 0, 0]
    transfer_observed = true_reflection + sigma * (
        rng.normal(size=n_transfer) + 1j * rng.normal(size=n_transfer))
    transfer_mll = MarginalLogLikelihood(
        predictor=lambda m, f: m.s(f)[:, 0, 0], observed=transfer_observed,
        likelihood=GaussianLikelihood(sigma**2),
    )
    # Whiten the cable's joint prior, leaving the new load in ohms.
    mean = joint.mean()
    factor = jnp.linalg.cholesky(joint.covariance())

    def from_white(x):
        cable_vector = mean + factor @ x[:-1]
        return prf.update(transfer, {'log_R': cable_vector[0],
                                    'discrepancy.values': cable_vector[1:].reshape(3, 2, n_transfer),
                                    'load': x[-1]})

    def loss_white(x):
        return -transfer_mll(from_white(x), transfer_frequency) + jnp.sum(x[:-1]**2) / 2

    value_gradient = jax.jit(jax.value_and_grad(loss_white))
    solution = minimize(lambda x: tuple(np.asarray(v) for v in value_gradient(jnp.asarray(x))),
                        np.r_[np.zeros(len(mean)), 75.0], jac=True, method='L-BFGS-B',
                        options={'maxiter': 300, 'gtol': 1e-5, 'ftol': 1e-12})
    if not solution.success:
        raise RuntimeError(f'Transfer fit failed: {solution.message}')
    transfer_map = from_white(jnp.asarray(solution.x))
    finish('transfer MAP', transfer_map.s(transfer_frequency))
    linearization = transfer_mll.linearize(transfer_map, transfer_frequency)
    covariance = posterior_covariance([linearization], transfer_map)
    finish('transfer linearisation and posterior', covariance)

    # QoI belongs to the unterminated corrected cable; the fitted load does not enter it.
    def qoi(model):
        cable = model.operands[0]
        return jnp.mean(jnp.abs(cable.s(transfer_frequency)[:, 1, 0])**2)

    (gradient,) = prf.derivative(qoi, transfer_map)
    gradient_values = prf.values(gradient)
    J_q = jnp.concatenate([jnp.ravel(gradient_values[name]) for name in linearization.names])
    q_variance = J_q @ covariance @ J_q
    J = linearization.J.reshape(-1, covariance.shape[0])
    # Independent real/imaginary reflection noise makes Sigma_D diagonal here.
    G = covariance @ J.T / sigma**2
    R = G @ J
    P_q = J_q @ (jnp.eye(len(J_q)) - R)
    G_q = J_q @ G
    finish('QoI and gain diagnostics', (q_variance, P_q, G_q))
    parameter_mean = jnp.concatenate([jnp.ravel(prf.values(transfer_map)[name]) for name in linearization.names])
    posterior_factor = jnp.linalg.cholesky(covariance)
    evaluate = jax.jit(jax.vmap(lambda x: qoi(prf.update(transfer_map, linearization.unflatten(x)))))
    # Batch both sampling and evaluation: 10^4 by 6002 need not stay resident.
    q_samples = []
    for start in range(0, mc_samples, 100):
        count = min(100, mc_samples - start)
        draws = parameter_mean + jnp.asarray(rng.normal(size=(count, len(parameter_mean)))) @ posterior_factor.T
        q_samples.append(np.asarray(evaluate(draws)))
    mc_std = np.std(np.concatenate(q_samples), ddof=1)
    delta_std = float(jnp.sqrt(q_variance))
    finish('Monte Carlo push-forward', q_variance)
    relative_std_error = abs(mc_std / delta_std - 1)
    print(f'QoI {float(qoi(transfer_map)):.6f}; linear std {delta_std:.6g}; '
          f'MC std {mc_std:.6g}; relative std error {relative_std_error:.2%}', flush=True)
    return dict(timings=timings, relative_std_error=relative_std_error,
                frequency=frequency, additive_residual=additive_residual,
                log_residual=log_residual, P_q=P_q, G_q=G_q,
                fitted_load=float(prf.values(transfer_map)['load']),
                delta_std=delta_std, mc_std=mc_std)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-frequency', type=int, default=1000)
    args = parser.parse_args()
    run(args.n_frequency)
