from abc import ABC, abstractmethod

from distreqx.distributions import AbstractDistribution

import jax.numpy as jnp
import parax as prx


class Likelihood(prx.Module, ABC):
    r"""
    Abstract base class for likelihood models.
    
    A likelihood in ParamRF specifies a mapping from model predictions to a probability over observed data.
    It operates in "event space", where the probabilistic event, such as frequency, is the last axis.
    
    This works for both deterministic and probabilistic models (e.g. from Gaussian processes):

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
    

class NoiseModel(prx.Module, prx.Operator, ABC):
    """
    Abstract base class for likelihood noise models.
    
    A noise model maps a model prediction to a noise parameter, such as variance.
    For example, for a Gaussian likelihood, a noise model can be used to model
    the variance with non-standard broadcasting rules.
    
    See :mod:`pmrf.likelihoods.noise_models` for built-in noise models.
    """
    @abstractmethod
    def __call__(self, y_event: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray | jnp.ndarray]:
        """
        Map model predictions to noise parameters.

        Parameters
        ----------
        y_event : jnp.ndarray
            The mean model prediction in event space.

        Returns
        -------
        jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]
            The noise parameter or a tuple of noise parameters.
        """        
        raise NotImplementedError