## What to build

A bounded LogNormal GP-plus-line MAP fit preserves tiny valid starts, uses feasible scale-aware SciPy search coordinates, and reaches a stationary solution. Any remaining nonfinite objective or derivative evaluation raises a clear diagnostic rather than silently returning a failed start point.

## Reviewer policy gate

- [ ] Before implementation, the reviewer amends ADR-0007 in the reviewer's own commit to approve the scale-relative open-edge algorithm and fixed affine SciPy scaling below. SolverView continues to expose box space; this introduces no public value space. Raw-start nudging and closed-edge semantics remain unchanged. The reviewer also ratifies the internal-coordinate metrics contract and fail-fast numerical diagnostics below. The implementation agent does not edit ADRs or domain context or choose another scaling policy.

## Implementation boundary

Work in SolverView box-edge/start construction, SciPy bound and private-coordinate conversion, host-side numerical diagnostics, and their optimization/fitting tests and docstrings. Preserve box space: the unit box for two finite bounds, declared space otherwise. Use feasible Bounds and affine scaling for trust-constr, case-insensitively, only when bounds are supplied; other methods and calls without supplied bounds retain their current representation and coordinates. Keep LogNormal and its bijectors unchanged. Every change stays in ParamRF; escalate a required dependency-repository change before making it.

## Chosen open-edge algorithm

For each finite open edge `b`, let `x0` be the parameter's original box-space start, `eps` the floating dtype's machine epsilon, and `tiny` its smallest positive normal number. Compute an inward distance `max(tiny, eps * max(abs(b), abs(x0 - b), tiny))`. Add it at a lower edge and subtract it at an upper edge. If rounding leaves the result on the excluded edge, move one representable value inward. If the proposed inset would exclude an already valid interior start, use that start as the search edge instead. Keep infinite edges infinite. A start exactly on an open edge is moved to the resulting interior edge; existing construction still rejects deliberately declared invalid starts. The reviewer ratifies these choices in this ticket before implementation.


## Acceptance criteria: preserving valid starts and domain edges

- [ ] Implement the chosen algorithm with dtype-preserving operations. Closed box edges remain exactly equal to their existing endpoints. All returned finite open edges are strictly inside the original domain; all original infinite edges retain sign and infinity.
- [ ] For `Random(LogNormal(log(v0), 1.0), value=v0)` with `v0` in `{1e-12, 1e-7, 1e-5, 2, 30, 130}`, `SolverView.box()` preserves the original representable start exactly and includes it in its bounds. In x64 the zero lower edge's inset equals `max(tiny, eps * v0)`; it is not an absolute `1e-6`.
- [ ] Cover float32 and float64, positive lower bounds, negative upper bounds, a two-sided open interval, an interval with one closed edge, and array-valued bounds/starts. For large nonzero bounds where addition rounds back to the edge, exercise the representable-value fallback.
- [ ] A valid start arbitrarily close to an open edge remains inside the returned search bounds. Include a start equal to the nearest representable interior value. The fallback must not move it outward or exclude it.
- [ ] A `Positive` parameter near zero obtains a positive normal-number lower search edge, avoiding a subnormal zero that JAX may flush. A finite one-sided bound retains an infinite opposite search edge; no accidental maximum-float replacement is allowed.
- [ ] Starts on closed edges remain unchanged in box space, and the existing raw-space closed-start nudging tests continue to pass. Existing closed interval endpoint objective values and derivatives remain valid.
- [ ] A recording bounded minimizer receives the original `1e-7` start, a strictly positive lower bound less than that start, and an infinite upper bound. Its initial loss and gradient match evaluating the intended declared start, rather than an implicitly moved value.
- [ ] Update box-space docstrings to describe the chosen edge policy. Run box-space and raw-space solver-view regressions. Keep all changes in ParamRF; ADR changes are completed by the reviewer at this ticket's policy gate.

## Chosen internal SciPy scaling

For flattened original box start `x0` and bounds `lower, upper`, select each scale `s`: two finite bounds use `upper - lower`; only a finite lower bound uses `abs(x0 - lower)`; only a finite upper bound uses `abs(upper - x0)`; neither finite uses `abs(x0)`. Replace any zero or nonfinite candidate scale with `1`. Scales are computed once and held fixed. SciPy starts at the zero vector `z0`, sees bounds `(lower - x0) / s` and `(upper - x0) / s`, and evaluates the original objective at `x = x0 + s * z`. Differentiate through that affine map, so a supplied gradient is multiplied by `s`. Convert the returned `z` to box `x` before unravelling the fitted parameters.

The equation `x = x0 + s * z` and scale selection were exercised in the temporary diagnosis prototype. Keep raw SciPy metrics in their actual internal coordinates and add `box_origin=x0` and `box_scale=s`; document that these recover box values from `metrics.x`. Do not partially rewrite derivative or multiplier fields into different units. For methods/calls without this scaling, metrics remain as before. Existing unsupported/equal-bound cases retain their existing handling; this ticket does not add a fixed-coordinate solver.

## Fixed LogNormal convergence fixture

Use x64, 20 linearly spaced frequencies from 10 to 500 MHz, and a fixed RLGCLine with length `10`, R `0.5`, L `250e-9`, C `100e-12`, and G `1e-6`. Predictor is real S21. Observations are the predictor plus `1e-3 * sin(f_MHz / 80)`. Use GaussianLikelihood noise variance `1e-8`; GP is `Matern52Kernel(lengthscale) * variance` with jitter `1e-12`. Lengthscale prior is `LogNormal(log(30), 1)` started at 30; variance prior is `LogNormal(log(1e-5), 1)` started at `1e-5`. These are the only free parameters. Fit with Bayesian prior penalty, `trust-constr`, default solver tolerances and at most 1,000 iterations.

The local diagnosis ran this fixture through the actual MAP problem, SolverView and SciPy adapter. Current behavior did not converge after 1,000 objective evaluations. Corrected edges plus feasible Bounds alone reported success but had a log-lengthscale gradient around `-14.6`: that result is not acceptable. Adding the affine scaling reached success in 25 iterations, with both log-coordinate gradient magnitudes below `7e-6`. With prior location and variance start changed to `1e-7`, it reached success in 32 iterations with gradient magnitudes below `6e-8`. These are measured sanity references, not required iteration counts or frozen fitted outputs.


## Acceptance criteria: feasible search and genuine convergence

- [ ] A bound-conversion test records the SciPy call and verifies a `Bounds` object with feasibility enabled for trust-constr. Bound order and infinities are preserved, and finite bounds use the specified affine conversion. Other methods and calls without bounds retain their current representation/behavior.
- [ ] Test scale selection for two-sided, lower-only, upper-only and unbounded coordinates, including a closed-edge start and a zero start. Test the transformed loss and gradient against the original objective and affine chain rule. Verify both gradient-based and gradient-free calls, and that returned parameters equal `box_origin + box_scale * metrics.x` in flattened order.
- [ ] A separate actual-SciPy regression uses an objective defined only inside its bounds and records every host-side evaluation. Every recorded point satisfies the bounds; no clipping inside the objective masks an outside trial. Exercise a boundary start as well as an interior start.
- [ ] The fixed public Bayesian fitting fixture returns `success=True` within 1,000 iterations, finite positive lengthscale and variance, and a finite loss lower than the initial loss.
- [ ] At the fitted point, both hyperparameters are interior and the gradient with respect to each hyperparameter's log value has magnitude less than `1e-3`. This verifies stationarity in meaningful coordinates rather than accepting SciPy's success flag alone.
- [ ] Record objective evaluations in the regression and assert all losses and gradients are finite and every evaluated variance/lengthscale is positive. The test must synchronize JAX evaluations. Do not loosen convergence tolerances, increase the iteration limit, replace LogNormal priors, or assert a particular evaluation count to pass.
- [ ] Run the same fixture through the public fit API with both the variance prior location and variance start changed to `1e-7` as a second case. Require convergence, finite results and the same log-gradient stationarity threshold; preserve its requested start. In this case require fitted variance less than `1e-6`, demonstrating that the old artificial lower edge no longer gates the solution.
- [ ] Keep the regression small and deterministic. The external 990-points-per-block, three-kernel NaN was not reproduced by the diagnosis and is not claimed fixed by this test. The numerical diagnostics group below ensures any remaining nonfinite evaluation fails clearly.
- [ ] Run SciPy optimizer, MAP-prior and bounded fitting regressions. Keep all changes in ParamRF.


## Acceptance criteria: numerical failure diagnostics

- [ ] Before returning a value to SciPy, check loss and every requested gradient entry with `isfinite`. Both NaN and either infinity count as failure. Apply the loss check to gradient-free methods too. If a Hessian callback is actually passed to SciPy, apply the same rule to that callback; do not introduce Hessian support merely for this ticket.
- [ ] Raise `FloatingPointError` on the first nonfinite evaluation. The message includes the SciPy method, the one-based objective evaluation number, whether `loss` or `gradient` was nonfinite, and the affected free parameter names for a nonfinite gradient. A nonfinite loss reports the available free parameter names as context without claiming any specific parameter caused it.
- [ ] If the attempted parameter vector itself is nonfinite, identify that separately in the same exception. Diagnostics refer to the evaluated original box or raw coordinates when invoked through SolverView, before the private SciPy affine conversion; they must not label the internal vector as box values. Direct low-level solver calls may use positional coordinate indices instead of unavailable names.
- [ ] Parameter values, full model representations and distributions are absent from exception messages and logs. Method, names, coordinate space and evaluation index are sufficient; no debug files are written.
- [ ] A finite initial loss/gradient followed by a NaN-loss trial raises on that trial, not after max iterations. A separate fixture with finite loss and a NaN gradient raises with the gradient's parameter name. Add initial-evaluation failure and infinite-loss/gradient cases.
- [ ] Test gradient-free loss failure and a JIT-computed nonfinite result. Synchronize the value before checking it on the host; no runtime exception escapes without the specified diagnostic merely because execution was deferred.
- [ ] The public minimize and Bayesian fitting APIs propagate this exception and do not construct a successful result or return the start point as a fallback. Existing finite nonconvergence still returns `success=False` with its current warning; it is not reclassified as a numerical failure.
- [ ] A finite converging objective still returns the existing fitted-tree shape; metrics follow the internal-coordinate contract specified above for scaled trust-constr, and remain unchanged otherwise. An unrelated user exception retains its type and message. Any progress bar opened by the adapter is closed on numerical failure.
- [ ] Documentation states that this adapter fails on nonfinite objective/derivative evaluations. It makes no claim that an unconfirmed external GP NaN was caused by raw whitening, nor that replacing LogNormal priors is required.
- [ ] Run SciPy optimizer and Bayesian fitting regressions. Keep all changes in ParamRF.

## Blocked by

None (can start immediately).
