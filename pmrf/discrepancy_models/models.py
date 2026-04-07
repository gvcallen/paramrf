"""
Models that cater for the discrepancy between an RF model and data.
"""
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import distreqx.distributions as dist

from pmrf.core import DiscrepancyModel

class GaussianProcess(DiscrepancyModel):
    """
    Gaussian process discrepancy model with a covariance kernel.
    
    Maps model predictions to a Gaussian Process distribution over frequency.
    
    The kernel is responsible for returning the correlation between two input points.
    Given an input `y` of shape (*batch_shape, event_dims), the kernel must accept
    two inputs (x1, x2) of the same shape (scalar or vector), and return an array
    that is broadcastable to *batch_shape.
    
    This easily allows for kernel batching. For example, to create multiple RBF kernels
    that model the last batch dimension D with independent kernels, simply create a kernel
    with parameters of shape (D,).
    
    See :class:`pmrf.DiscrepancyModel` for more information on general discrepancy models.
    """
    kernel: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    jitter: float = eqx.field(default=1e-10, static=True)

    def __call__(self, y: jnp.ndarray, x: jnp.ndarray) -> dist.AbstractDistribution:
        """
        y: shape (..., N)
        x: shape (N,)
        """
        x_feat = x[:, None]
        
        # 1. Double vmap using out_axes natively pushes the N x N grid to the back
        inner_vmap = jax.vmap(self.kernel, in_axes=(None, 0), out_axes=-1)
        outer_vmap = jax.vmap(inner_vmap, in_axes=(0, None), out_axes=-2)
        
        # K shape is now guaranteed to be (N, N) OR (*batch_dims, N, N)
        K = outer_vmap(x_feat, x_feat) 
            
        # 2. Add jitter (broadcasting handles the batch dims automatically)
        K = K + jnp.eye(x.shape[0]) * self.jitter

        # 3. Build the batched distribution safely
        init_fn = dist.MultivariateNormalFullCovariance
        for _ in range(y.ndim - 1):
            init_fn = eqx.filter_vmap(init_fn)
            
        return init_fn(loc=y, covariance_matrix=K)