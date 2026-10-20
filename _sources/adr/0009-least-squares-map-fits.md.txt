# ADR-0009: MAP fits are bounded least squares, alternating with a hyperparameter step

Status: accepted (2026-10)

## Context

Physical line models often have parameters that are each identifiable but jointly
near-degenerate: the loss is a long, narrow valley. On a synthetic `MicrostripLine` fit
(10-500 MHz, `w`, `h`, `ep_r`, `tand`, `sigma` and `length` free, noise 1e-5, Jacobian
condition number about 4e5 in box space), starting from a perturbed point:

| Method | Outcome |
|---|---|
| `ScipyMinimize` L-BFGS-B (default) | Reports success at a wrong point |
| `ScipyMinimize` trust-constr (BFGS) | Not converged in 3000 iterations |
| trust-constr, exact JAX Hessian | Not converged in 3000 iterations, about 20x slower |
| trust-constr, Gauss-Newton Hessian | Not converged in 3000 iterations |
| trust-exact, Levenberg-Marquardt (unbounded) | Leave the box and fail |
| Optimistix Levenberg-Marquardt, Dogleg, Gauss-Newton (raw space) | Stall on the bounds, where the raw Jacobian vanishes |
| SciPy `least_squares`, `'trf'`, box space | Recovers all six true values |

The exact Hessian adds the residual-weighted second derivative of the model, the term a
linearisation drops. Far from the solution it hurts rather than helps.

Under a Gaussian likelihood with fixed noise and discrepancy hyperparameters, the negative
log posterior is half the squared norm of a residual vector, up to a constant, whenever
each prior is Gaussian, truncated Gaussian or uniform. Noise and kernel hyperparameters
are usually fitted too, and they enter through $\tfrac{1}{2}\ln|\Sigma_D|$, which is not a
square.

## Decision

### A least-squares minimiser over box space

`pmrf.optimize.ScipyLeastSquares` wraps `scipy.optimize.least_squares` with `'trf'`. Like
any minimiser that honours bounds, it searches box space and is given the box's edges
(ADR-0007). It minimises half the squared norm of a residual vector, not a scalar.

### The residual vector

The vector stacks the **whitened residual** of each term and the **prior residual** of each
free parameter.

- **Whitened residual.** For a `MarginalLogLikelihood`, $L^{-1} r$ per event block, with
  $L$ the lower Cholesky factor of $\Sigma_D = K + \Sigma_n$: the same factor
  `linearize` uses. For a `TargetLoss` with `MSELoss`, the prediction error scaled so
  that half its squared norm is the loss.
- **Prior residual.** Computed from the distribution, in the space it is over. It is not
  the raw value, since truncation makes the raw map nonlinear.

| Prior | Prior residual | Bounds |
|---|---|---|
| `Normal` | $(\theta - \mu)/\sigma$ | none |
| `TruncatedNormal`, including a `Normal` truncated to a validity | $(\theta - \mu)/\sigma$ | the truncation |
| Joint multivariate normal, including `truncate='unnormalised'` | $L_0^{-1}(\theta - \mu)$, with $\Sigma_0 = L_0 L_0^\top$ | each parameter's validity |
| `Uniform`, or no prior | none | the parameter's bounds |

A truncation normaliser is constant in $\theta$, so it does not move the MAP.

### Alternating with a hyperparameter step

The level above the minimiser splits the free parameters. The model parameters are the
free parameters of `problem.model`. The **hyperparameters** are the free parameters the
terms own, such as white noise and kernel hyperparameters. `run_minimizer` hands the
minimiser both groups: a residual function of the model parameters given the
hyperparameters, and the full scalar objective of the hyperparameters given the model
parameters. The minimiser knows nothing about noise or kernels.

Each round is a least-squares solve over the model parameters, then a **hyperparameter
step**: a scalar minimisation of the full negative log posterior over the hyperparameters
with the model parameters held, by `hyper_minimizer` (default `ScipyMinimize()`). The fit
stops when a round no longer decreases the full objective. With no free hyperparameters
it is a single least-squares solve.

### Not yet supported

Each of these raises an error saying it is not yet supported:

- noise given as a callable of the prediction, since $\Sigma_D$ then depends on the
  model parameters;
- a mixing that depends on the prediction, for the same reason;
- orthogonal discrepancy without a fixed reference basis;
- priors other than those in the table, such as a prior over raw space carried to
  declared space with a change-of-variables term;
- `GibbsMarginalLogLikelihood`, losses other than `MSELoss`, and any term without a
  whitened residual.

## Considered options

- **Restore the exact Hessian in `ScipyMinimize`.** Rejected as the fix: it did not
  converge on the degenerate fit and is slow to compile. It may still be added for other
  reasons.
- **Optimistix least-squares solvers.** They keep the fit in JAX, but take no bounds, and
  in raw space they stall on the bounds. Rejected.
- **One least-squares solve over model parameters and hyperparameters together.** The
  log-determinant is not a square; writing it as one gives a residual with a singular
  derivative. Rejected.

## Consequences

- At the MAP, the stacked whitened Jacobian and prior square root of the last
  least-squares solve are the $A$ of `posterior_covariance_factor`. The fit and its
  linearisation share one approximation.
- `ScipyMinimize`'s commented-out Hessian path is removed.
- Shared discrepancy is supported once its whitened residual covers the joint covariance
  across datasets, for mixings that do not depend on the prediction.
