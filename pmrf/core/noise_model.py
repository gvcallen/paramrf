from jaxtyping import PyTree

import jax.numpy as jnp
import parax as prx

class NoiseModel(prx.Module, prx.Operator):
    """
    Base class for noise models.
    
    Maps model predictions to likelihood noise parameters. For example,
    for a real Gaussian likelihood (:class:`pmrf.likelihoods.GaussianLikehood`),
    the required output is the likelihood's variance.
    """
    def __call__(self, y_pred: jnp.ndarray) -> PyTree:
        raise NotImplementedError