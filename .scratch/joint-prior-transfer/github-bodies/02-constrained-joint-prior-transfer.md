## What to build

A CoaxialLine reference fit passes its declared parameter/discrepancy Gaussian to a PortCorrected transfer model with `prior(..., truncate='unnormalised')`. Interior density and Hessian are preserved, the terminated one-port posterior matches a NumPy Kalman update, and sampling rejects the unnormalised prior.

## Reviewer policy gate

- [ ] Before implementation, the reviewer amends ADR-0005 in the reviewer's own commit to approve explicit unnormalised joint priors for MAP, scoring, differentiation and posterior covariance, plus the static normalization flag and blanket sampling rejection specified below. The implementation agent does not edit ADRs or domain context.

The source is the ParamRF 0.36.5 joint-prior handoff dated 2026-10-06 and the subsequent diagnosis. Retain the declared Gaussian because its interior Hessian is the precision required by the Kalman-update test. A Gaussian approximation in raw space generally changes that Hessian. The design is explicit opt-in truncation without renormalisation; the implementation agent does not choose among raw-space approximation, tail-mass tolerance or truncation policies.

## Implementation boundary

Work in prior attachment/scoring, the existing `Probabilistic` wrapper, the distributions mirror used by `PriorPenalized`, sampling dispatch validation, the `predict_joint` documentation/example, and their unit/integration tests. Retain one joint-prior mechanism, the existing constant whitening log-determinant optimization, and declared-space joint prediction. Explain any necessary prediction-algorithm change against an independent covariance calculation. Keep every change in ParamRF; escalate any required dependency-repository change before making it.


## Acceptance criteria: attachment and normalization metadata

- [ ] Add keyword-only `truncate='normalised'` to `prior`. Unknown modes raise `ValueError`. `'unnormalised'` with a scalar-event distribution or `space='raw'` raises `ValueError` naming the unsupported combination. Declared and physical joint distributions support the opt-in.
- [ ] Default attachment still rejects a multivariate normal whose support leaves any selected field's validity, including one whose mean is many standard deviations from its bound. There is no automatic negligible-tail tolerance or support-check monkeypatch.
- [ ] Opt-in attachment accepts that distribution, drops the selected parameters' own priors and user ranges as existing declared/physical attachment does, and retains their field validity, declared values, scales, names, shapes and metadata. Parameters outside the joint prior keep their existing priors.
- [ ] Store static `normalised=False` on the joint wrapper for opt-in attachments and `True` for default attachments. The flag survives `resolve`, `update`, wrapping the model in a larger tree and supported serialization/deserialization. It is not an optimizable array.

## Acceptance criteria: scoring, validity and curvature

- [ ] At finite interior values, declared-space `log_prior` equals the supplied multivariate Gaussian log density plus the independent priors outside the joint block, with `rtol=1e-12, atol=1e-12` in x64. The joint block's gradient and negative Hessian equal the analytic Gaussian gradient and inverse covariance to `rtol=1e-10, atol=1e-10`.
- [ ] Physical-space scoring applies the existing scale Jacobian exactly once. Raw-space scoring applies the joint whitening Jacobian exactly once. The new mode adds no normalization constant and no parameter-dependent term inside validity.
- [ ] An invalid joint member, including NaN produced by #276, makes that joint block score `-inf`. The mask handles excluded open endpoints and included closed endpoints correctly, and reduces over the member's declared shape while retaining leading batch axes. Tests exercise these masks through both public `log_prior` and the distributions-mirror scoring used by `PriorPenalized`.
- [ ] For an interior trial, `PriorPenalized` equals its underlying loss minus declared `log_prior`, and its Hessian contains the Gaussian precision exactly once. The test uses at least two correlated scalar parameters plus one array-valued member and one independent parameter outside the joint prior.
- [ ] Existing overlap, fixing, tying, event-size and free-parameter checks remain active in opt-in mode. Existing exact-normalisation and raw-prior tests still pass.

## Acceptance criteria: sampling restrictions

- [ ] Every public sampling path, including direct sampling and `fit_sample`, rejects an active opt-in prior before `solver.run`. Use a recording fake sampler for each joint, split and hypercube interface. The `ValueError` must contain `unnormalised joint prior`, the affected parameter names, and `MAP and linearisation`; it must explain that sampling requires a normalised prior. Default priors remain sampleable.

## Acceptance criteria: prior documentation and regressions

- [ ] Docstrings describe zero density outside validity, absent renormalisation, the supported inference operations and sampling rejection. Do not implement a mass tolerance, raw-space prediction feature, evidence approximation or new sampling algorithm in this ticket.
- [ ] Run joint-prior, prior-validity, MAP-prior, serialization and sampling-dispatch regressions. Keep all changes in ParamRF.

## Fixed transfer fixture

Use x64 and a reciprocal CoaxialLine with length `0.1 m`, inner diameter `1.12e-3 m`, outer diameter `3.2e-3 m`, ConstantDielectric with relative permittivity `1.384` and loss tangent `0.001`, and BulkConductor with conductivity `1 / 1.6e-8 S/m`. Keep all parameters fixed except length. Give length `Bounded(0.05, 0.15, value=0.095)` so its field validity is still positive. Use 12 linearly spaced reference frequencies from 100 to 500 MHz, and six transfer frequencies from 120 to 480 MHz. The line's true length is `0.1 m`.

Use reciprocal event blocks in order `('11', '22', 's21')`, each with Re/Im axes. Reflection events are additive; transmission events use the local continuous complex logarithm of S21. The existing example's reciprocal event transform provides this recipe. Use fixed Matern52 lengthscale `80 MHz`, fixed discrepancy variance `1e-6`, GP jitter `1e-12`, and measurement variance `1e-8` per real event coordinate. Use noise-free synthetic observations of the true line; the likelihood still has the specified nonzero variance.


## Acceptance criteria: constrained reference-to-transfer workflow

- [ ] Run a reference MAP fit using a bounded minimizer, with fixed GP hyperparameters and at most 1,000 iterations. Assert success and finite fitted values. Only model length is free in the model passed to `predict_joint`.
- [ ] Call `predict_joint` on the transfer grid. Verify the parameter names are exactly `('length',)` and the joint event size is `1 + 3 * 2 * 6 = 37`. Its discrepancy mean is reshaped in C order to `(3, 2, 6)` without permuting event blocks or Re/Im axes.
- [ ] Construct `GridPortDiscrepancy` on that grid, wrap the fitted model with `PortCorrected`, update discrepancy values to the predicted mean, and attach the joint with `prior(corrected, [*names, 'discrepancy.values'], joint, truncate='unnormalised')`. No monkeypatching or private support-check bypass is used. `log_prior` is finite at that joint mean eagerly and under JIT.
- [ ] The corrected component is terminated at its second port in a fixed 75-ohm load and observed from its first port at 50-ohm reference impedance. Flatten S11 real/imaginary events in the same order used by the transfer likelihood. Transfer noise variance is `1e-6` per real coordinate.
- [ ] Build the transfer linearisation in declared space at the attached joint mean. Let `C` be the predicted joint covariance in the attachment's order and `J` the transfer observation Jacobian reordered into that same order. Assemble the permutation explicitly from free parameter names and declared shapes; do not assume resolver order equals attachment order.
- [ ] Independently compute in NumPy `C - C @ J.T @ solve(J @ C @ J.T + R, J @ C)`, with `R = 1e-6 * I`. Use a NumPy solve, not ParamRF posterior covariance, for the reference. Reorder the library covariance to the same order and require maximum relative diagonal error at most `1e-6`; all reference diagonals must be positive. Do not add a tolerance-sized absolute floor or extra covariance jitter to pass this check.
- [ ] Check the interior prior negative Hessian against `inverse(C)` separately to `rtol=1e-8, atol=1e-8`. This distinguishes attachment/scoring defects from transfer Jacobian defects. Include the length/discrepancy cross-covariance; a block-diagonal replacement is a failing implementation.

## Acceptance criteria: executable recipe and integration regressions

- [ ] Replace the broken docstring attachment example with the complete constrained recipe, using `truncate='unnormalised'`. The example names must match the tree: root names are `length` and `discrepancy.values`; embedding under a named cable uses the cable-prefixed names. Execute the documented recipe in a test.
- [ ] Explain in the docstring that the transfer prior is a locally Gaussian approximation truncated without renormalisation, supported for MAP and linearisation. The numerical Kalman check validates covariance at the supplied linearisation point, not a nonlinear refit to an arbitrary new dataset.
- [ ] Run the joint-prediction, port-discrepancy and posterior-covariance integration regressions. Keep all changes in ParamRF; do not change domain documents in the implementation commit.

## Blocked by

#276 — Evaluate invalid parameter trials without exceptions.
