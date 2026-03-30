"""
Distributions not present in distreqx.
"""


import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree
import equinox as eqx

from distreqx.distributions import AbstractSampleLogProbDistribution, AbstractProbDistribution

class Empirical(AbstractSampleLogProbDistribution, AbstractProbDistribution):
    """
    A distribution defined by a discrete set of joint samples.
    Good for encapsulating MCMC or Nested Sampling posteriors.
    """
    samples: PyTree[Array]
    num_samples: int = eqx.field(static=True)
    log_likelihoods: Array | None = None
    weights: Array | None = None

    def __init__(
        self, 
        samples: PyTree[Array], 
        log_likelihoods: Array | None = None, 
        weights: Array | None = None
    ):
        self.samples = samples
        
        # Extract the leading dimension (batch size) from the first leaf
        leaves = jax.tree_util.tree_leaves(samples)
        if not leaves:
            raise ValueError("Samples PyTree is empty.")
        self.num_samples = leaves[0].shape[0]

        if log_likelihoods is not None:
            if log_likelihoods.shape != (self.num_samples,):
                raise ValueError(f"log_likelihoods must have shape ({self.num_samples},)")
        self.log_likelihoods = log_likelihoods

        if weights is not None:
            if weights.shape != (self.num_samples,):
                raise ValueError(f"weights must have shape ({self.num_samples},)")
            # Normalize weights to sum to 1 to ensure correct weighted averages
            self.weights = weights / jnp.sum(weights)
        else:
            self.weights = None

    @property
    def event_shape(self) -> PyTree:
        # The shape of a single event is the shape of the samples minus the batch dim
        return jax.tree_util.tree_map(lambda x: x.shape[1:], self.samples)

    def sample(self, key: Key[Array, ""]) -> PyTree[Array]:
        if self.weights is None:
            # Uniformly draw a random index from the empirical samples
            idx = jax.random.randint(key, shape=(), minval=0, maxval=self.num_samples)
        else:
            # Draw an index weighted by the sample weights
            idx = jax.random.choice(key, self.num_samples, p=self.weights)
            
        return jax.tree_util.tree_map(lambda x: x[idx], self.samples)

    def mean(self) -> PyTree[Array]:
        if self.weights is None:
            return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), self.samples)
        else:
            def _weighted_mean(x):
                # Reshape weights to broadcast across trailing dimensions of x
                w = self.weights.reshape((-1,) + (1,) * (x.ndim - 1))
                return jnp.sum(x * w, axis=0)
            return jax.tree_util.tree_map(_weighted_mean, self.samples)

    def variance(self) -> PyTree[Array]:
        if self.weights is None:
            return jax.tree_util.tree_map(lambda x: jnp.var(x, axis=0), self.samples)
        else:
            mu = self.mean()
            def _weighted_var(x, m):
                w = self.weights.reshape((-1,) + (1,) * (x.ndim - 1))
                return jnp.sum(w * (x - m)**2, axis=0)
            return jax.tree_util.tree_map(_weighted_var, self.samples, mu)

    def stddev(self) -> PyTree[Array]:
        return jax.tree_util.tree_map(lambda x: jnp.sqrt(x), self.variance())

    def median(self) -> PyTree[Array]:
        if self.weights is None:
            return jax.tree_util.tree_map(lambda x: jnp.median(x, axis=0), self.samples)
        else:
            raise NotImplementedError("Weighted median requires sorting and is not currently implemented.")

    def mode(self) -> PyTree[Array]:
        """Returns the Maximum A Posteriori (MAP) estimate if log_likelihoods are provided."""
        if self.log_likelihoods is not None:
            idx = jnp.argmax(self.log_likelihoods)
            return jax.tree_util.tree_map(lambda x: x[idx], self.samples)
            
        raise NotImplementedError("Mode is not well-defined without log_likelihoods.")

    # --- Methods stubbed to satisfy AbstractProbDistribution ---

    def log_prob(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError(
            "Analytic log_prob is not defined for an empirical sample distribution. "
            "Use Kernel Density Estimation (KDE) if continuous evaluation is required."
        )

    def cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError("Analytic cdf is not defined for an empirical sample distribution.")

    def log_cdf(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError("Analytic log_cdf is not defined for an empirical sample distribution.")

    def survival_function(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError("Analytic survival_function is not defined for an empirical sample distribution.")

    def log_survival_function(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError("Analytic log_survival_function is not defined for an empirical sample distribution.")

    def entropy(self) -> PyTree[Array]:
        raise NotImplementedError("Analytic entropy is not defined for an empirical sample distribution.")

    def kl_divergence(self, other_distribution) -> PyTree[Array]:
        raise NotImplementedError("KL divergence is not defined for an empirical sample distribution.")
    
    def to_unweighted(self, key: Key[Array, ""], beta: float | str = 1.0) -> "Empirical":
        """
        Compresses the distribution into an unweighted/equally weighted 
        Empirical distribution using Fractional (or Systematic) Resampling.

        Instead of standard multinomial resampling (which introduces high Monte Carlo noise),
        this method scales the sample weights by the target Effective Sample Size (ESS). 
        Each sample is then deterministically repeated a number of times equal to the 
        integer part of its scaled weight. The remaining fractional part is treated as 
        a probability for including one additional copy of the sample. This probabilistic 
        rounding ensures that the total number of drawn samples closely matches the 
        continuous ESS without excessive sampling variance.

        The target ESS is determined dynamically using the Huggins-Roy family of 
        effective sample sizes, parameterized by `beta`.

        Args:
            key: JAX PRNG key for the probabilistic rounding of fractional weights.
            beta: Huggins-Roy family parameter dictating the compression level.
                  - beta=1.0 (or 'entropy'): Uses the theoretical channel capacity 
                    (exponentiated Shannon entropy). Default.
                  - beta=2.0 (or 'kish'): Uses Kish's effective sample size 
                    (inverse sum of squared weights).
                  - beta=math.inf (or 'inf', 'equal'): Resolves to the inverse 
                    of the maximum weight.
        """
        if self.weights is None:
            return self

        # 1. Calculate the target Effective Sample Size (ncompress)
        if beta == 'inf' or beta == 'equal' or beta == math.inf:
            ncompress = 1.0 / jnp.max(self.weights)
        elif beta == 'entropy' or (isinstance(beta, (float, int)) and math.isclose(float(beta), 1.0)):
            safe_w = jnp.where(self.weights > 0, self.weights, 1.0)
            ncompress = jnp.exp(-jnp.sum(self.weights * jnp.log(safe_w)))
        elif beta == 'kish' or (isinstance(beta, (float, int)) and math.isclose(float(beta), 2.0)):
            ncompress = 1.0 / jnp.sum(self.weights ** 2)
        else:
            beta_val = float(beta)
            ncompress = jnp.sum(self.weights ** beta_val) ** (1.0 / (1.0 - beta_val))

        # 2. Scale weights to the target channel capacity
        W = self.weights * ncompress

        # 3. Split into guaranteed integer counts and fractional probabilities
        integer_part = jnp.floor(W).astype(jnp.int32)
        fractional_part = W - integer_part

        # 4. Probabilistically round the fractional parts
        u = jax.random.uniform(key, shape=(self.num_samples,))
        extra = (u < fractional_part).astype(jnp.int32)
        
        # Total times each sample will be repeated
        counts = integer_part + extra

        # 5. Concretize counts to perform dynamic array reshaping
        # We must pull this to the CPU as a concrete numpy array because 
        # JAX cannot dynamically shape arrays inside jitted functions.
        import numpy as np
        concrete_counts = np.array(counts)

        # 6. Use jnp.repeat to duplicate samples based on their counts
        # jnp.repeat with axis=0 turns [A, B, C] with counts [2, 0, 1] into [A, A, C]
        new_samples = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, concrete_counts, axis=0), 
            self.samples
        )
        
        new_ll = None
        if self.log_likelihoods is not None:
            new_ll = jnp.repeat(self.log_likelihoods, concrete_counts, axis=0)

        # Return the unweighted distribution
        return Empirical(
            samples=new_samples,
            log_likelihoods=new_ll,
            weights=None
        )