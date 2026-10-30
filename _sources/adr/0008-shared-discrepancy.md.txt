# Shared discrepancy is a joint evaluator over per-dataset likelihoods

A discrepancy whose latent functions are shared across datasets (or across event blocks
of one dataset) makes those datasets statistically dependent, so their joint likelihood
is not a sum of per-dataset terms. We add a new evaluator,
`SharedMarginalLogLikelihood`, that takes several ordinary `MarginalLogLikelihood`s
(each with no discrepancy of its own) plus a tuple of `SharedLatent`s, and returns one
joint log-likelihood. It goes into `SummedTerms` as a single term. Each `SharedLatent`
is a covariance kernel paired with a **mixing**: a function of the dataset's
event-space prediction and frequency that returns, per frequency, a real matrix from
the latent's rows to the dataset's event blocks. The joint covariance is low rank plus
diagonal and is solved by Woodbury in the latent dimension, never densely.

## Considered options

- **A shared object inside several `MarginalLogLikelihood` terms, marginalised jointly by
  `SummedTerms`.** Rejected: `SummedTerms` would have to detect sharing and stop being a
  sum, and the joint maths would be spread across classes that should not know about
  each other.
- **Extend `GaussianProcess` and its kernels to couple event blocks** (as in the
  correlated-GP note and the routing idea in #250). Rejected: a `GaussianProcess` is a
  discrepancy for one dataset, and its kernels cannot see the prediction, which
  multiplicative error (for example a VNA's tracking error, $e_t \Gamma$) needs.
- **An error-box model component with GP-distributed error terms.** This is the exact
  bilinear form, but its marginal is not Gaussian. It is a separate, later feature.

## Consequences

- `MarginalLogLikelihood` exposes its event-space residual, noise variance and
  conditional-transform log-determinant, so the joint evaluator reuses all per-dataset
  plumbing (predictor, observation, likelihood, event transform) without copying it.
- A latent holds a kernel, not a `GaussianProcess`. Every existing kernel is reusable.
- Mixings act in event space. Helpers that multiply by the prediction assume the
  default Re/Im event layout and say so.
- With a prediction-dependent mixing, the covariance depends on the model parameters;
  gradients flow through it.
- A `MarginalLogLikelihood` with a `GaussianProcess` is mathematically the one-dataset,
  identity-mixing case. Both stay: the existing class is the fast independent-block
  path. If the two ever duplicate maths rather than share plumbing, merge them into one
  evaluator with the independent path as an internal optimisation.
