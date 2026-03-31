"""
An abstract object that represents an arbitrary evaluation of a model over frequency.
"""
import jax.numpy as jnp
import parax as prx

class Metric(prx.Module):
    """
    A callable model that computes a metric between two arrays.
    
    Used for losses and likelihoods.
    """
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError