"""
Likelihood models for statistical fitting.
"""

from abc import abstractmethod
from typing import Callable

import jax.numpy as jnp
import equinox as eqx
from distreqx.distributions import AbstractDistribution, Normal, MultivariateNormalFullCovariance

from pmrf.parameters import Param, param

class AbstractLikelihood(eqx.Module):
    r"""
    Abstract base class for likelihood models.
    
    A likelihood in ParamRF specifies a mapping from model predictions to a probability over observed data.
    It operates in "event space", where the probabilistic event, such as frequency, is the last axis.
    
    This works for both deterministic and probabilistic models (e.g. Gaussian processes):

    * For deterministic predictions, returns the conditional distribution $p(y_{true} \mid y_{pred})$.
    * For probabilistic predictions, returns the marginal distribution $p(y_{true})$ over $y_{pred}$.
    
    See :mod:`pmrf.likelihoods` for built-in likelihood models.
    """
    @abstractmethod
    def __call__(self, y_event: jnp.ndarray | AbstractDistribution) -> AbstractDistribution:
        """
        Evaluate the likelihood given model predictions.

        Parameters
        ----------
        y_event : jnp.ndarray | AbstractDistribution
            The model prediction or predictive distribution in event space.

        Returns
        -------
        AbstractDistribution
            The probability distribution over the observed data.
        """
        raise NotImplementedError


class GaussianLikelihood(AbstractLikelihood):
    r"""
    Gaussian likelihood with independent noise.
    
    Maps measurement noise to a normal distribution over frequency.
    The noise model is responsibile for returning the measurement variance.

    Given an input `y` of shape `(*batch_shape, event_dims)`, the noise model
    must accept accept the prediction `y` and return an array that is either
    broadcastable to `(*batch_shape)` or to the full (*batch_shape, event_dims).

    Parameters
    ----------
    noise: Param | Callable[[jnp.ndarray], jnp.ndarray]
        A parameter or callable that evaluates to the measurement variance. 
        It can accept the mean prediction as an argument.
    """
    #: The noise parameter or a callable returning the measurement variance.
    noise: Param | Callable[[jnp.ndarray], jnp.ndarray]

    def __call__(self, y_event: jnp.ndarray | AbstractDistribution) -> AbstractDistribution:
        is_dist = isinstance(y_event, AbstractDistribution)
        y_mean = y_event.mean() if is_dist else y_event
        
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
            mapped_normal = Normal
            for _ in range(num_batch_dims):
                mapped_normal = eqx.filter_vmap(mapped_normal)
                
            return mapped_normal(y_mean, jnp.sqrt(mapped_var))
        else:
            if not hasattr(y_event, "covariance"):
                raise TypeError("The predicted distribution must natively implement `covariance()`.")
            
            def get_cov(d): return d.covariance()
            mapped_get_cov = get_cov
            for _ in range(num_batch_dims):
                mapped_get_cov = eqx.filter_vmap(mapped_get_cov)
            pred_cov = mapped_get_cov(y_event)
            
            def add_noise(cov, var_diag):
                return cov + jnp.diag(var_diag)
            
            mapped_add = add_noise
            for _ in range(num_batch_dims):
                mapped_add = eqx.filter_vmap(mapped_add)
            new_cov = mapped_add(pred_cov, mapped_var)
            
            init_fn = MultivariateNormalFullCovariance
            for _ in range(num_batch_dims):
                init_fn = eqx.filter_vmap(init_fn)
                
            return init_fn(y_mean, new_cov)
        
        
__all__ = [
    'AbstractLikelihood',
    'GaussianLikelihood',
]