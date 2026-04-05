from distreqx.distributions import AbstractDistribution

import jax.numpy as jnp
import parax as prx

class Likelihood(prx.Module):
    """
    Base class for likelihood models.
    
    A likelihood in ParamRF specifies a mapping from model predictions to a probability over observerd data.
    
    This works for both deterministic and probabilstic models (e.g. from Gaussian processes):
    1) For deterministic predictions, returns the conditional distribution p(y_true | y_pred)
    2) For probabilistic predictions, returns the marginal distribution p(y_true) over y_pred.
    """
    def __call__(self, y_pred: jnp.ndarray | AbstractDistribution) -> AbstractDistribution:
        """Returns (distreqx.distributions.AbstractDistribution)"""
        raise NotImplementedError