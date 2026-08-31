Release notes
=============

Next release
------------

Orthogonal Gaussian-process discrepancy now uses a column-scaled, explicitly
rank-truncated SVD of the free-parameter Jacobian. Its log density is evaluated through
the nonsingular block factorization of ``P K P^T + sigma^2 I`` and retains the tangent
block; this is the intended full-data model, not REML.

This is a correctness change. The former normal-equation projector was ill-conditioned,
included fixed model parameters, and could produce backend-dependent likelihoods.
Consequently, results from existing fits made with ``use_orthogonal_discrepancy=True``
must be considered invalid and should be rerun. No claim is yet made about how the
correction changes posterior parameters or in-fit sampling efficiency; that requires a
separate posterior study.

Users must now select ``orthogonal_rcond`` explicitly. By default the tangent basis is
fixed with ``evaluator.with_orthogonal_reference(model, frequency)``. Per-call
linearization remains available with ``orthogonal_recompute=True`` and includes the
full higher-order derivative rather than stopping its gradient.
