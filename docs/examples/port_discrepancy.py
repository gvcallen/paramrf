"""Transfer a constrained joint Gaussian prior to a reflection-only fit.

Run with ``python docs/examples/port_discrepancy.py``.
"""
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from distreqx.bijectors import AbstractBijector
from distreqx.distributions import Normal
from scipy.optimize import minimize

import pmrf as prf
from pmrf.stats.covariance_kernels import Matern52Kernel
from pmrf.stats.discrepancy_models import GaussianProcess
from pmrf.objectives.evaluators import MarginalLogLikelihood
from pmrf.stats.likelihoods import GaussianLikelihood
from pmrf.stats.linearization import posterior_covariance
from pmrf.models import CoaxialLine, GridPortDiscrepancy, Load, PortCorrected, RLGCLine, SModel
from pmrf.materials import BulkConductor, ConstantDielectric

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


class AbstractPortEventTransform(AbstractBijector, strict=True):
    """Example-local event transform conditioned on a predicted response."""

    #: Prediction selecting the local unwrapped logarithm branch.
    predicted: jax.Array
    #: Log-transmission derivatives depend on the response.
    _is_constant_jacobian: bool = eqx.field(static=True, init=False, default=False)
    #: The log determinant depends on transmission magnitude.
    _is_constant_log_det: bool = eqx.field(static=True, init=False, default=False)

    def inverse_log_det_jacobian(self, event):
        return -self.forward_log_det_jacobian(self.inverse(event))

    def forward_and_log_det(self, s):
        return self.forward(s), self.forward_log_det_jacobian(s)

    def inverse_and_log_det(self, event):
        return self.inverse(event), self.inverse_log_det_jacobian(event)

    def same_as(self, other):
        return self is other


class PortEventTransform(AbstractPortEventTransform, strict=True):
    r"""Invertible full two-port transform, with frequency last in event space.

    **Mathematical Formulation**

    $$h(\widetilde S; S)=\ln|\widetilde S| + i[\operatorname{unwrap}(\arg S)
    + \operatorname{unwrap}(\arg(\widetilde S/S))].$$

    Reflections remain additive. Directional transmission logs become their
    mean and half-difference. Each complex log contributes
    $$-2\ln|\widetilde S|$$ to the forward log determinant, and the rotation
    contributes $$-2\ln 2$$ per frequency.
    """

    def forward(self, s):
        anchor = jnp.unwrap(jnp.angle(self.predicted), axis=0)
        phase = anchor + jnp.unwrap(jnp.angle(s / self.predicted), axis=0)
        logs = jnp.log(jnp.abs(s)) + 1j * phase
        blocks = jnp.stack((s[:, 0, 0], s[:, 1, 1],
                            (logs[:, 1, 0] + logs[:, 0, 1]) / 2,
                            (logs[:, 1, 0] - logs[:, 0, 1]) / 2))
        return jnp.stack((blocks.real, blocks.imag), axis=1)

    def inverse(self, event):
        blocks = event[:, 0] + 1j * event[:, 1]
        return jnp.stack((jnp.stack((blocks[0], jnp.exp(blocks[2] - blocks[3])), axis=-1),
                          jnp.stack((jnp.exp(blocks[2] + blocks[3]), blocks[1]), axis=-1)), axis=-2)

    def forward_log_det_jacobian(self, s):
        # The mean/half-difference rotation has determinant 1/2 for each Re/Im pair.
        return -2 * jnp.log(jnp.abs(s[:, 0, 1])) - 2 * jnp.log(jnp.abs(s[:, 1, 0])) - 2 * jnp.log(2.0)


class ReciprocalEventTransform(AbstractPortEventTransform, strict=True):
    r"""Transform the retained reflections and symmetric transmission only.

    **Mathematical Formulation**

    The symmetric transmission uses the anchored logarithm of
    :class:`PortEventTransform`, with forward log determinant
    $$-2\ln|\widetilde S^s|.$$ Reflections remain additive.
    """

    def forward(self, s):
        anchor = jnp.unwrap(jnp.angle(self.predicted[:, 2]))
        log_t = jnp.log(jnp.abs(s[:, 2])) + 1j * (
            anchor + jnp.unwrap(jnp.angle(s[:, 2] / self.predicted[:, 2])))
        blocks = jnp.stack((s[:, 0], s[:, 1], log_t))
        return jnp.stack((blocks.real, blocks.imag), axis=1)

    def inverse(self, event):
        blocks = event[:, 0] + 1j * event[:, 1]
        return jnp.stack((blocks[0], blocks[1], jnp.exp(blocks[2])), axis=-1)

    def forward_log_det_jacobian(self, s):
        return -2 * jnp.log(jnp.abs(s[:, 2]))


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
    full_event = PortEventTransform(initial.s(frequency)).forward(observed)
    # Discard only the antisymmetric block, then encode the symmetric log in S space.
    retained = full_event[:3, 0] + 1j * full_event[:3, 1]
    reciprocal_observed = jnp.stack((retained[0], retained[1], jnp.exp(retained[2])), axis=-1)
    variance = jnp.ones((3, 2, n_frequency)) * sigma**2
    variance = variance.at[2].set(sigma**2 / (2 * jnp.abs(reciprocal_observed[:, 2])**2))
    reference = MarginalLogLikelihood(
        predictor=reciprocal_predictor, observed=reciprocal_observed,
        likelihood=GaussianLikelihood(variance),
        discrepancy=GaussianProcess(Matern52Kernel(100.0) * 0.05**2, jitter=1e-8),
        event_transform=ReciprocalEventTransform,
    )
    finish('synthetic data and event transform', reciprocal_observed)
    fitted = reference_fit(reference, initial, frequency)
    finish('reference MAP', fitted.s(frequency))
    event = ReciprocalEventTransform(reciprocal_predictor(fitted, frequency))
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


def run_constrained_transfer():
    """Run the constrained CoaxialLine reference-to-transfer prior recipe."""
    def cable(length):
        return CoaxialLine(
            length=length,
            d_in=prf.Fixed(1.12e-3),
            d_out=prf.Fixed(3.2e-3),
            dielectric=ConstantDielectric(ep_r=prf.Fixed(1.384), tand=prf.Fixed(0.001)),
            conductor=BulkConductor(sigma=prf.Fixed(1 / 1.6e-8)),
        )

    reference_frequency = prf.Frequency(100.0, 500.0, 12, 'MHz')
    transfer_frequency = prf.Frequency(120.0, 480.0, 6, 'MHz')
    truth_model = cable(prf.Fixed(0.1))
    reference_model = cable(prf.Bounded(0.05, 0.15, value=0.095))
    reference_observed = reciprocal_predictor(truth_model, reference_frequency)
    reference = MarginalLogLikelihood(
        predictor=reciprocal_predictor,
        observed=reference_observed,
        likelihood=GaussianLikelihood(jnp.full((3, 2, reference_frequency.npoints), 1e-8)),
        discrepancy=GaussianProcess(Matern52Kernel(80.0) * 1e-6, jitter=1e-12),
        event_transform=ReciprocalEventTransform,
    )

    def reference_loss(length):
        candidate = prf.update(reference_model, {'length': length})
        return -reference(candidate, reference_frequency)

    value_gradient = jax.jit(jax.value_and_grad(reference_loss))
    reference_fit_result = minimize(
        lambda x: tuple(np.asarray(value) for value in value_gradient(jnp.asarray(x))),
        np.array([0.095]),
        jac=True,
        method='L-BFGS-B',
        bounds=[(0.05, 0.15)],
        options={'maxiter': 1000, 'gtol': 1e-10, 'ftol': 1e-14},
    )
    if not reference_fit_result.success or not np.isfinite(reference_fit_result.fun):
        raise RuntimeError(f'Reference MAP fit failed: {reference_fit_result.message}')
    fitted = prf.update(reference_model, {'length': jnp.asarray(reference_fit_result.x[0])})
    names, joint = reference.predict_joint(fitted, reference_frequency, transfer_frequency)
    mean, covariance = joint.mean(), joint.covariance()
    delta_mean = mean[1:].reshape((3, 2, transfer_frequency.npoints), order='C')
    corrected = PortCorrected(
        fitted,
        GridPortDiscrepancy(
            prf.Unconstrained(delta_mean), transfer_frequency, ('11', '22', 's21')
        ),
    )
    attached = prf.prior(
        corrected,
        [*names, 'discrepancy.values'],
        joint,
        truncate='unnormalised',
    )
    transfer = attached.terminated(Load(gamma=(75.0 - 50.0) / (75.0 + 50.0), z0=50.0))
    transfer_observed = truth_model.terminated(
        Load(gamma=(75.0 - 50.0) / (75.0 + 50.0), z0=50.0)
    ).s(transfer_frequency, z0=50.0)[:, 0, 0]
    transfer_likelihood = MarginalLogLikelihood(
        predictor=lambda model, frequency: model.s(frequency, z0=50.0)[:, 0, 0],
        observed=transfer_observed,
        likelihood=GaussianLikelihood(1e-6),
    )
    linearization = transfer_likelihood.linearize(transfer, transfer_frequency, space='declared')
    posterior = posterior_covariance([linearization], transfer, space='declared')

    prior_names = (*names, 'discrepancy.values')
    prior_shapes = ((), (3, 2, transfer_frequency.npoints))
    prior_offsets = {}
    offset = 0
    for name, shape in zip(prior_names, prior_shapes):
        prior_offsets[name] = offset
        offset += int(np.prod(shape))
    linearization_values = prf.values(transfer, free_only=True, space='declared')
    covariance_indices = []
    theta = []
    for name, shape in zip(linearization.names, linearization.shapes):
        size = int(np.prod(shape))
        matches = [
            prior_name for prior_name in prior_names
            if name == prior_name or name.endswith('.' + prior_name)
        ]
        if len(matches) != 1:
            raise ValueError(f'Cannot map transfer parameter {name!r} into the joint prior.')
        start = prior_offsets[matches[0]]
        covariance_indices.extend(range(start, start + size))
        theta.extend(np.ravel(np.asarray(linearization_values[name])))
    covariance = np.asarray(covariance)[np.ix_(covariance_indices, covariance_indices)]
    jacobian = np.asarray(linearization.J).reshape((-1, len(theta)))
    measurement = np.eye(jacobian.shape[0]) * 1e-6
    expected_posterior = covariance - covariance @ jacobian.T @ np.linalg.solve(
        jacobian @ covariance @ jacobian.T + measurement,
        jacobian @ covariance,
    )
    relative_diagonal_error = np.abs(
        np.diag(np.asarray(posterior) - expected_posterior) / np.diag(expected_posterior)
    )

    theta = jnp.asarray(theta)
    sizes = [int(np.prod(shape)) for shape in linearization.shapes]
    offsets = np.cumsum([0, *sizes])

    def values_from_vector(vector):
        return {
            name: vector[offsets[i]:offsets[i + 1]].reshape(shape)
            for i, (name, shape) in enumerate(zip(linearization.names, linearization.shapes))
        }

    prior_precision = -jax.hessian(
        lambda vector: prf.log_prior(
            prf.update(transfer, values_from_vector(vector), space='declared'),
            space='declared',
        )
    )(theta)
    expected_prior_precision = np.linalg.inv(covariance)
    prior_precision_relative_error = np.linalg.norm(
        np.asarray(prior_precision) - expected_prior_precision, ord='fro'
    ) / np.linalg.norm(expected_prior_precision, ord='fro')
    log_prior = prf.log_prior(transfer, space='declared')
    jitted_log_prior = eqx.filter_jit(lambda model: prf.log_prior(model, space='declared'))(transfer)
    return dict(
        names=names,
        joint=joint,
        attached=attached,
        reference_success=reference_fit_result.success,
        reference_objective=reference_fit_result.fun,
        relative_diagonal_error=relative_diagonal_error,
        expected_covariance_diagonal=np.diag(expected_posterior),
        prior_precision=np.asarray(prior_precision),
        expected_prior_precision=expected_prior_precision,
        prior_precision_relative_error=prior_precision_relative_error,
        log_prior=log_prior,
        jitted_log_prior=jitted_log_prior,
    )


if __name__ == '__main__':
    result = run_constrained_transfer()
    print(f"Reference MAP success: {result['reference_success']}")
    print(f"Maximum relative Kalman covariance error: {np.max(result['relative_diagonal_error']):.3g}")
