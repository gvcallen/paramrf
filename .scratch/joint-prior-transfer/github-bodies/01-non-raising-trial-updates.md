## What to build

Users can evaluate numerical line-search trials eagerly and under JIT with `update(..., on_invalid='nan')`. Invalid parameter leaves become NaN so the objective can reject the trial; ordinary updates remain strict.

## Reviewer policy gate

- [ ] Before implementation, the reviewer records the numerical-trial exception to strict update validation in ADR-0002's parameter/update policy or its existing update discussion, in the reviewer's own commit. Approve the exact API and whole-leaf NaN semantics below. The implementation agent does not edit ADRs or domain context and does not choose an alternative invalid-value policy.

## Implementation boundary

Work in the ParamRF parameter update, constraint-checking and numerical write-back functions, their public docstrings and their corresponding tests. Keep this option local to each update call. Preserve the wrapped variable's constraint, prior, scale, validity, name, metadata and fixed state. Use a safe in-domain value during reconstruction if necessary, then poison the variable's numerical value; rebuilding through an invalid constructor must not raise before the opt-in takes effect. Do not change Parax.


## Acceptance criteria: API and trial semantics

- [ ] Add keyword-only `on_invalid='raise'`; accept exactly `'raise'` and `'nan'`. Unknown policies raise `ValueError` even when the value would be valid.
- [ ] Support mapping updates, selector plus `value=`, and direct single-parameter `value=` updates, in declared, physical and raw space. Raw updates are checked after mapping to declared space. This includes values written through existing joint-prior whitening.
- [ ] For `on_invalid='nan'`, a value outside the parameter's effective constraint, on an excluded open edge, or containing a nonfinite entry makes the entire updated parameter leaf NaN. Other parameter leaves keep their requested valid values. Closed edges remain valid.
- [ ] A floating array leaf with one invalid entry becomes entirely NaN. A `vmap` over separate valid and invalid scalar trials produces finite results for valid trials and NaN for invalid trials without poisoning other mapped trials.
- [ ] Preserve the existing array dtype, shape/broadcast semantics and weak-type behavior of valid updates. In NaN mode, updating a non-floating, non-complex numerical leaf raises `TypeError` explaining that NaN trial updates require a floating dtype; no silent dtype promotion is introduced.
- [ ] `on_invalid='nan'` is accepted only for numerical value updates. Structural `node`/`fn` updates, fixed-state updates, and mixed mappings containing sub-model replacements raise `ValueError` when this opt-in is supplied. Existing structural semantics with the default are unchanged.

## Acceptance criteria: regression coverage and documentation

- [ ] Tests cover `Positive` at `-1` and `0`; a closed interval at both endpoints and just outside; a scalar truncated prior outside its range; a scaled physical-space parameter; an array parameter; and a raw joint-whitening trial that makes a member's declared value nonfinite. For the last case, use an existing raw-space Gaussian joint prior over LogNormal parameters and a large finite whitened coordinate that overflows the exponential. This fixture requires no unnormalised attachment; finite outside-validity trials for the new opt-in joint prior are tested in #277.
- [ ] Run each invalid-value test eagerly and under `eqx.filter_jit`, synchronizing results so deferred runtime errors are detected. Each returns NaN without raising. The same invalid finite values under the default continue to raise the existing constraint error.
- [ ] A valid update in NaN mode gives the same declared values and objective as the default, and has the same autodiff derivative. No host callback or exception-catching branch is required for NaN mode.
- [ ] Add a docstring example of a numerical objective using the returned trial and rejecting a nonfinite result. State that this mode makes parameter validation non-raising; independent model errors, such as a discrepancy-grid mismatch, retain their own behavior.
- [ ] Run the existing parameter validity, structural-update and joint-prior update tests together with the new cases. Keep all changes in ParamRF; escalate a required Parax change rather than making it.

## Blocked by

None (can start immediately).
