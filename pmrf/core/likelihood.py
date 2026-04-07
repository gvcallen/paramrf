from distreqx.distributions import AbstractDistribution

import jax.numpy as jnp
import parax as prx


class Likelihood(prx.Module):
    """
    Base class for likelihood models.
    
    A likelihood in ParamRF specifies a mapping from model predictions to a probability over observerd data.
    It functions in "event space", where the probabilistic event, such as frequency, is the last axis.
    
    This works for both deterministic and probabilstic models (e.g. from Gaussian processes):
    1) For deterministic predictions, returns the conditional distribution p(y_true | y_pred)
    2) For probabilistic predictions, returns the marginal distribution p(y_true) over y_pred.
    
    See :mod:`pmrf.likelihoods` for built-in likelihood models.
    """
    def __call__(self, y_event: jnp.ndarray | AbstractDistribution) -> AbstractDistribution:
        """Returns (distreqx.distributions.AbstractDistribution)"""
        raise NotImplementedError
    

class NoiseModel(prx.Module, prx.Operator):
    """
    Base class for likelihood noise models.
    
    A noise model maps a model prediction to a noise parameter, such as variance.
    For example, for a Gaussian likelihood (:class:`pmrf.likelihoods.GaussianLikehood`),
    a noise model can be used to model the variance with non-standard broadcasting rules.
    
    See :mod:`pmrf.noise_models` for built-in noise models.
    """
    def __call__(self, y_pred: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray | jnp.ndarray]:
        raise NotImplementedError