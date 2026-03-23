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