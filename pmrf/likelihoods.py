"""
Likelihood models for statistical fitting.
"""

from abc import abstractmethod
from typing import Callable

import jax.numpy as jnp
import equinox as eqx
from distreqx.distributions import AbstractDistribution, Normal, MultivariateNormalFullCovariance

from pmrf.parameters import Param
from pmrf.modules.base import Module

class AbstractLikelihood(Module):
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
        The callable accepts the mean prediction as an argument and
        outputs the variance, and can be a function or a callable PyTree
        with additional parameters.
    """
    #: The noise parameter or a callable returning the measurement variance.
    noise: Param | Callable[[jnp.ndarray], jnp.ndarray]

    def variance(self, y_event: jnp.ndarray) -> jnp.ndarray:
        """Return noise variance broadcastable over the non-event batch shape.

        Orthogonal GP discrepancy requires variance that is constant along the event
        axis so that the tangent covariance block is ``sigma^2 I``.
        """
        var = self.noise(y_event) if callable(self.noise) else self.noise
        var = jnp.asarray(var)
        batch_shape = y_event.shape[:-1]
        if var.shape == y_event.shape:
            raise ValueError(
                "Orthogonal GP discrepancy requires Gaussian noise variance to be "
                "constant along the event axis."
            )
        try:
            broadcast_shape = jnp.broadcast_shapes(var.shape, batch_shape)
        except ValueError as error:
            raise ValueError(
                "Gaussian noise variance is not broadcastable to the event batch shape."
            ) from error
        if broadcast_shape != batch_shape:
            raise ValueError(
                "Gaussian noise variance must not add dimensions to the event batch shape."
            )
        return var

    def __call__(self, y_event: jnp.ndarray | AbstractDistribution) -> AbstractDistribution:
        # If y_event is an array, the prediction is deterministic
        # and we can simply use a regular gaussian likelihood.
        # Otherwise, the second else branch performs "marginalization",
        # effectivelly adding the covariances together.

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
            
            try:
                pred_cov = mapped_get_cov(y_event)
            except Exception as e:
                raise ValueError(f"Error encounted when trying to compute the covariance of `GaussianLikelihood`: {e}")
            
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
