"""
Statistical likelihood models.
"""

from typing import Callable
import jax.numpy as jnp
import distreqx.distributions as dist
import parax as prx
import equinox as eqx
from pmrf.core import Likelihood


class GaussianLikelihood(Likelihood):
    """
    Gaussian likelihood with independent noise.
    
    Maps measurement noise to a normal distribution over frequency.
    
    The noise model is responsibile for returning the measurement variance.
    Given an input `y` of shape (*batch_shape, event_dims), the noise model
    must accept accept the prediction `y` and return an array that is either
    broadcastable to (*batch_shape) or the full shape of `y`.
    
    This allows for complex, independent heteroskedastic and homoskedastic
    noise for a given inputs.
    """
    noise: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray]

    def __call__(self, y: jnp.ndarray | dist.AbstractDistribution) -> dist.MultivariateNormalFullCovariance:
        is_dist = isinstance(y, dist.AbstractDistribution)
        y_mean = y.mean() if is_dist else y
        
        # Evaluate noise
        var = self.noise(y_mean) if callable(self.noise) else self.noise
        var = jnp.asarray(var)
        
        # Broadcast noise onto y_mean
        batch_shape = y_mean.shape[:-1]
        if var.shape == batch_shape:
            var = var[..., None]
        mapped_var = jnp.broadcast_to(var, y_mean.shape)
        num_batch_dims = y_mean.ndim - 1

        if not is_dist:
            mapped_normal = dist.Normal
            for _ in range(num_batch_dims):
                mapped_normal = eqx.filter_vmap(mapped_normal)
                
            return mapped_normal(y_mean, mapped_var)
        else:
            if not hasattr(y, "covariance"):
                raise TypeError("The predicted distribution must natively implement `covariance()`.")
            
            def get_cov(d): return d.covariance()
            mapped_get_cov = get_cov
            for _ in range(num_batch_dims):
                mapped_get_cov = eqx.filter_vmap(mapped_get_cov)
            pred_cov = mapped_get_cov(y)
            
            def add_noise(cov, var_diag):
                return cov + jnp.diag(var_diag)
            
            mapped_add = add_noise
            for _ in range(num_batch_dims):
                mapped_add = eqx.filter_vmap(mapped_add)
            new_cov = mapped_add(pred_cov, mapped_var)
            
            init_fn = dist.MultivariateNormalFullCovariance
            for _ in range(num_batch_dims):
                init_fn = eqx.filter_vmap(init_fn)
                
            return init_fn(loc=y_mean, covariance_matrix=new_cov)