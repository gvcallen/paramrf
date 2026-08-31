"""
Discrepancy modeling between an RF model and actual data.

Useful for modeling discrepancy of RF models during fitting.
"""
from collections.abc import Callable
from abc import abstractmethod

import equinox as eqx
import jax.scipy as jsp
import jax.numpy as jnp
import distreqx.distributions as dist

from pmrf.covariance_kernels import gram
from pmrf.utils import field
from pmrf.modules.base import Module

class AbstractDiscrepancyModel(Module):
    """
    Abstract base class for discrepancy models.
    
    A discrepancy model maps a model prediction to an updated model prediction.
    This updated prediction can either be deterministic (e.g. a polynomial)
    or probabilistic (e.g. a Gaussian process) by either returning a `JAX` array
    or a `distreqx` probability distribution.
    
    Note that probabilistic discrepancy models operate in "event space".
    Here, probability events (e.g. frequency) are moved to the **last axis**.
    
    These models are commonly used in conjuction with a likelihood function
    via :class:`pmrf.evaluators.MarginalLogLikelihood`.
    
    See :mod:`pmrf.discrepancy_models` for built-in discrepancy models.
    """
    @abstractmethod
    def __call__(self, y_event: jnp.ndarray) -> jnp.ndarray | dist.AbstractDistribution:
        """
        Apply discrepancy correction to a model prediction.

        Parameters
        ----------
        y_event : jnp.ndarray
            The initial model prediction in event space.

        Returns
        -------
        jnp.ndarray | dist.AbstractDistribution
            The updated deterministic or probabilistic prediction.
        """        
        raise NotImplementedError

class GaussianProcess(AbstractDiscrepancyModel):
    """
    Gaussian process discrepancy model with a covariance kernel.
    
    Maps model predictions to a Gaussian Process distribution over frequency.
    
    The kernel is responsible for returning the correlation between two input points.
    Given an input `y` of shape `(*batch_shape, event_dims)`, the kernel must accept
    two inputs (x1, x2) of the same shape (scalar or vector), and return an array
    that is broadcastable to `*batch_shape`.
    
    This easily allows for kernel batching. For example, to create multiple RBF kernels
    that model the last batch dimension D with independent kernels, simply create a kernel
    with parameters of shape (D,).
    
    See :class:`pmrf.DiscrepancyModel` for more information on general discrepancy models.
    See :mod:`pmrf.covariance_kernels` for built-in covariance kernels.

    Parameters
    ----------
    kernel : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        The covariance kernel function that computes the correlation between two input arrays.
        Can be a function or a callable PyTree. See :mod:`pmrf.covariance_kernels`
        for built-in covariance kernels.
    jitter : float, default=1e-10
        A small scalar added to the diagonal of the covariance matrix for numerical stability.
    """
    #: The covariance kernel.
    kernel: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    
    #: The added jitter.
    jitter: float = field(default=1e-10, static=True)

    def orthogonal_log_prob(
        self,
        y_event: jnp.ndarray,
        observed: jnp.ndarray,
        x: jnp.ndarray,
        noise_variance: jnp.ndarray,
        basis,
    ) -> jnp.ndarray:
        r"""Evaluate the full-data ``P K P^T + sigma^2 I`` Gaussian density.

        This uses its nonsingular block factorization; it is not REML. In particular,
        the tangent-space block is retained because it depends on the fitted mean and
        measurement noise.
        """
        K = gram(self.kernel, x, jitter=self.jitter)
        variance = jnp.asarray(noise_variance)
        n = y_event.shape[-1]
        # Keep M at the smallest broadcast batch shape. Its Cholesky is then shared
        # automatically across any additional event/basis batch axes.
        M = K + variance[..., None, None] * jnp.eye(n, dtype=K.dtype)
        chol_M = jnp.linalg.cholesky(M)

        def chol_solve(chol, rhs):
            solved = jsp.linalg.solve_triangular(chol, rhs, lower=True)
            return jsp.linalg.solve_triangular(
                jnp.swapaxes(chol, -1, -2), solved, lower=False
            )

        residual = observed - y_event
        Q1 = basis.vectors
        mask = basis.mask
        minv_Q1 = chol_solve(chol_M, Q1)
        minv_r = chol_solve(chol_M, residual[..., None])[..., 0]
        Q1_T = jnp.swapaxes(Q1, -1, -2)
        small = Q1_T @ minv_Q1
        # Rejected SVD columns are padded zeros. Giving those slots an identity block
        # preserves a static shape without contributing to either determinant.
        small = small + (
            jnp.eye(small.shape[-1], dtype=small.dtype)
            * (~mask).astype(small.dtype)[..., None, :]
        )
        chol_small = jnp.linalg.cholesky(small)
        coupling = (Q1_T @ minv_r[..., None])[..., 0]
        corrected = jnp.sum(
            coupling * chol_solve(chol_small, coupling[..., None])[..., 0], axis=-1
        )
        tangent = (Q1_T @ residual[..., None])[..., 0]

        logdet_M = 2 * jnp.sum(
            jnp.log(jnp.diagonal(chol_M, axis1=-2, axis2=-1)), axis=-1
        )
        logdet_small = 2 * jnp.sum(
            jnp.log(jnp.diagonal(chol_small, axis1=-2, axis2=-1)), axis=-1
        )
        rank = jnp.sum(mask, axis=-1)
        quadratic = (
            jnp.sum(tangent**2, axis=-1) / variance
            + jnp.sum(residual * minv_r, axis=-1)
            - corrected
        )
        normalizer = (
            n * jnp.log(2 * jnp.pi)
            + rank * jnp.log(variance)
            + logdet_M
            + logdet_small
        )
        return -0.5 * (normalizer + quadratic)

    def __call__(self, y_event: jnp.ndarray, x: jnp.ndarray, orthogonal_projection: jnp.ndarray | None = None) -> dist.AbstractDistribution:
        """
        Evaluate the Gaussian process distribution over the given inputs.

        Parameters
        ----------
        y_event : jnp.ndarray
            The model prediction in event space, with shape `(..., N)`.
        x : jnp.ndarray
            The frequency points, with shape `(N,)`.
        orthogonal_projection : jnp.ndarray, optional
            An optional matrix P of shape ``(..., N, N)`` which the kernel
            matrix is projected onto using ``P @ K @ P^T``. This can be used
            to specify the subspace which the kernel is allowed.

        Returns
        -------
        dist.AbstractDistribution
            A multivariate normal distribution parameterized by the mean `y_event` 
            and the covariance matrix generated by the kernel.
            
        See Also
        --------
        pmrf.covariance_kernels.gram : Builds the covariance matrix used here.
        """
        K = gram(self.kernel, x, jitter=self.jitter)
        
        if orthogonal_projection is not None:
            projection_T = jnp.swapaxes(orthogonal_projection, -1, -2)
            K = orthogonal_projection @ K @ projection_T

        target_K_shape = y_event.shape[:-1] + K.shape[-2:]
        K = jnp.broadcast_to(K, target_K_shape)

        init_fn = dist.MultivariateNormalFullCovariance
        for _ in range(y_event.ndim - 1):
            init_fn = eqx.filter_vmap(init_fn)
            
        return init_fn(y_event, K)
    
__all__ = [
    'AbstractDiscrepancyModel',
    'GaussianProcess',
]
