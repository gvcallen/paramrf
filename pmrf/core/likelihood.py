from distreqx.distributions import AbstractDistribution

import jax.numpy as jnp
import parax as prx

class Likelihood(prx.Module, prx.Operator):
    """
    Base class for likelihood models.
    
    Maps predictions y_pred to a conditional distribution p(y | y_pred) over observed data.
    
    For complex input data, the output distribution is defined over R^2 for [real, imag].
    In general, the distribution is defined over R^2, where the first component is real and the second is imaginary.
    """
    def __call__(self, y_pred: jnp.ndarray) -> AbstractDistribution:
        """Returns (distreqx.distributions.AbstractDistribution)"""
        raise NotImplementedError