"""
Built-in discrepancy models.
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
        
        inner_vmap = jax.vmap(self.kernel, in_axes=(None, 0), out_axes=-1)
        outer_vmap = jax.vmap(inner_vmap, in_axes=(0, None), out_axes=-2)
        
        K = outer_vmap(x_feat, x_feat) 
        K = K + jnp.eye(x.shape[0]) * self.jitter

        # ---> NEW FIX: Explicitly broadcast K to match y's batch dimensions <---
        target_K_shape = y.shape[:-1] + K.shape[-2:]
        K = jnp.broadcast_to(K, target_K_shape)

        init_fn = dist.MultivariateNormalFullCovariance
        for _ in range(y.ndim - 1):
            init_fn = eqx.filter_vmap(init_fn)
            
        return init_fn(y, K)