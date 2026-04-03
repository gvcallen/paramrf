from distreqx.distributions import AbstractDistribution

import jax.numpy as jnp
import parax as prx

class Likelihood(prx.Module, prx.Operator):
    """
    Base class for likelihood models.
    
    Maps predictions to a distribution over observed data.
    Works for either deterministic predictions or predictions with uncertainty (e.g. from Gaussian processes).
    
    1) For deterministic predictions, returns the conditional distribution p(y_true | y_pred)
    2) For probabilistic predictions, returns the marginal distribution p(y_true) over y_pred.
    
    For complex input data, the output distribution is defined over R^2 for [real, imag].
    In general, the distribution is defined over R^2, where the first component is real and the second is imaginary.
    """
    def __call__(self, y_pred: jnp.ndarray | AbstractDistribution) -> AbstractDistribution:
        """Returns (distreqx.distributions.AbstractDistribution)"""
        raise NotImplementedError