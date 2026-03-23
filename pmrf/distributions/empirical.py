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

    def __init__(self, samples: PyTree[Array]):
        self.samples = samples
        # Extract the leading dimension (batch size) from the first leaf
        leaves = jax.tree_util.tree_leaves(samples)
        if not leaves:
            raise ValueError("Samples PyTree is empty.")
        self.num_samples = leaves[0].shape[0]

    @property
    def event_shape(self) -> PyTree:
        # The shape of a single event is the shape of the samples minus the batch dim
        return jax.tree_util.tree_map(lambda x: x.shape[1:], self.samples)

    def sample(self, key: Key[Array, ""]) -> PyTree[Array]:
        # Uniformly draw a random index from the empirical samples
        idx = jax.random.randint(key, shape=(), minval=0, maxval=self.num_samples)
        return jax.tree_util.tree_map(lambda x: x[idx], self.samples)

    def mean(self) -> PyTree[Array]:
        return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), self.samples)

    def variance(self) -> PyTree[Array]:
        return jax.tree_util.tree_map(lambda x: jnp.var(x, axis=0), self.samples)

    def stddev(self) -> PyTree[Array]:
        return jax.tree_util.tree_map(lambda x: jnp.std(x, axis=0), self.samples)

    def median(self) -> PyTree[Array]:
        return jax.tree_util.tree_map(lambda x: jnp.median(x, axis=0), self.samples)

    # --- Methods stubbed to satisfy AbstractProbDistribution ---

    def log_prob(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError(
            "Analytic log_prob is not defined for an empirical sample distribution. "
            "Use Kernel Density Estimation (KDE) if continuous evaluation is required."
        )

    def mode(self) -> PyTree[Array]:
        raise NotImplementedError("Mode is not well-defined for an unweighted empirical sample distribution.")

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