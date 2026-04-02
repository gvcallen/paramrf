"""
Distributions not present in distreqx.
"""
import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree
import equinox as eqx
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from distreqx.distributions import AbstractDistribution, AbstractSampleLogProbDistribution, AbstractProbDistribution, Independent, Normal, AbstractMultivariateNormalFromBijector
from distreqx.bijectors import AbstractLinearBijector, AbstractBijector, Block, Chain, Shift, TriangularLinear

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
    
    def icdf(self, value: PyTree[Array]) -> PyTree[Array]:
        raise NotImplementedError("Analytic icdf is not defined for an empirical sample distribution.")

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
        
    def marginalized(
        self, 
        keep_indices: Array | list | int | None = None, 
        drop_indices: Array | list | int | None = None, 
        axis: int = 1
    ) -> "Empirical":
        """
        Returns a new marginalized Empirical distribution by keeping or dropping 
        specific indices along a given axis.

        Args:
            keep_indices: Indices to retain. Cannot be used with drop_indices.
            drop_indices: Indices to remove. Cannot be used with keep_indices.
            axis: The axis to slice along. Defaults to 1 (the first event dimension), 
                  because axis 0 is strictly reserved for the batch/sample count.
        """
        if keep_indices is not None and drop_indices is not None:
            raise ValueError("Specify either `keep_indices` or `drop_indices`, not both.")
        
        if keep_indices is None and drop_indices is None:
            return self

        # 1. Slice the samples PyTree
        if keep_indices is not None:
            # jnp.take extracts specific indices along the axis
            new_samples = jax.tree_util.tree_map(
                lambda x: jnp.take(x, jnp.asarray(keep_indices), axis=axis), 
                self.samples
            )
        else:
            # jnp.delete removes specific indices along the axis
            new_samples = jax.tree_util.tree_map(
                lambda x: jnp.delete(x, jnp.asarray(drop_indices), axis=axis), 
                self.samples
            )

        # 2. Return the new Marginal Distribution
        # CRITICAL: We explicitly set log_likelihoods=None because the joint 
        # probability P(A, B) is not equal to the marginal probability P(A).
        # Sample weights are preserved as they remain valid for integration.
        return Empirical(
            samples=new_samples,
            log_likelihoods=None,
            weights=self.weights
        )


def _check_input_parameters_are_valid(
    scale: AbstractLinearBijector, loc: Array
) -> None:
    """Raises an error if `scale` and `loc` are not valid."""
    if loc.ndim < 1:
        raise ValueError("`loc` must have at least 1 dimension.")
    if scale.event_dims != loc.shape[-1]:
        raise ValueError(
            f"`scale` and `loc` have inconsistent dimensionality: "
            f"`scale.event_dims = {scale.event_dims} and "
            f"`loc.shape[-1] = {loc.shape[-1]}."
        )



def _check_full_cov_parameters(loc: Optional[Array], covariance_matrix: Optional[Array]) -> None:
    """Checks that the `loc` and `covariance_matrix` parameters are correct."""
    if covariance_matrix is not None:
        if covariance_matrix.ndim < 2:
            raise ValueError(
                "Argument `covariance_matrix` must have at least 2 dimensions."
            )
        if covariance_matrix.shape[-1] != covariance_matrix.shape[-2]:
            raise ValueError(
                f"The last two dimensions of `covariance_matrix` must be equal "
                f"(square matrices), but got shapes {covariance_matrix.shape[-2:]}."
            )

    if loc is not None and not loc.shape:
        raise ValueError("If provided, argument `loc` must have at least 1 dimension.")

    if (
        loc is not None
        and covariance_matrix is not None
        and (loc.shape[-1] != covariance_matrix.shape[-1])
    ):
        raise ValueError(
            f"The last dimension of arguments `loc` and `covariance_matrix` "
            f"must coincide, but {loc.shape[-1]} != {covariance_matrix.shape[-1]}."
        )


class MultivariateNormalFullCovariance(AbstractMultivariateNormalFromBijector, strict=True):
    """Multivariate normal distribution on `R^k` with full covariance matrix."""

    loc: Array
    scale: AbstractLinearBijector
    distribution: AbstractDistribution
    bijector: AbstractBijector
    covariance_matrix: Array

    def __init__(
        self, 
        loc: Optional[Array] = None, 
        covariance_matrix: Optional[Array] = None
    ):
        """Initializes a MultivariateNormalFullCovariance distribution.

        **Arguments:**

        - `loc`: Mean vector of the distribution. If not specified, it defaults
            to zeros. At least one of `loc` and `covariance_matrix` must be specified.
        - `covariance_matrix`: A positive-definite covariance matrix. If not specified,
            it defaults to the identity matrix. At least one of `loc` and 
            `covariance_matrix` must be specified.
        """
        _check_full_cov_parameters(loc, covariance_matrix)

        if covariance_matrix is None and loc is not None:
            # Default to Identity matrix
            covariance_matrix = jnp.eye(loc.shape[-1], dtype=loc.dtype)
        elif loc is None and covariance_matrix is not None:
            loc = jnp.zeros(covariance_matrix.shape[-1], covariance_matrix.dtype)

        if loc is None or covariance_matrix is None:
            raise ValueError("At least one of `loc` or `covariance_matrix` must be specified.")

        # 1. Compute the lower Cholesky factor of the covariance matrix
        # This will raise a LinAlgError if the matrix is not positive definite
        scale_tril = jax.scipy.linalg.cholesky(covariance_matrix, lower=True)
        
        # 2. Form the linear scale bijector
        scale = TriangularLinear(scale_tril)
        
        _check_input_parameters_are_valid(scale, loc)

        # 3. Build a standard multivariate Gaussian (mean 0, variance 1)
        std_mvn_dist = Independent(
            distribution=eqx.filter_vmap(Normal)(
                jnp.zeros_like(loc), jnp.ones_like(loc)
            ),
        )
        
        # 4. Form the transformation bijector `f(x) = Lz + loc`
        bijector = Chain([Block(Shift(loc), ndims=loc.ndim), scale])
        
        self.distribution = std_mvn_dist
        self.bijector = bijector
        self.scale = scale
        self.loc = loc
        self.covariance_matrix = covariance_matrix

    def icdf(self, value: Array) -> Array:
        """See `Distribution.icdf`."""
        raise NotImplementedError("ICDF is not analytically tractable for full covariance MVN.")

    def cdf(self, value: Array) -> Array:
        """See `Distribution.cdf`."""
        # For full covariance, CDF requires numerical integration over a hyper-rectangle.
        # JAX doesn't have a native batched MVN CDF solver yet.
        raise NotImplementedError("CDF is not analytically tractable for full covariance MVN.")

    def log_cdf(self, value: Array) -> Array:
        """See `Distribution.log_cdf`."""
        raise NotImplementedError("Log CDF is not analytically tractable for full covariance MVN.")