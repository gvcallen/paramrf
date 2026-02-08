from typing import Callable

import jax.numpy as jnp

from pmrf.sampling.adaptive import AdaptiveSampler
from pmrf.models.model import Model

class MaximumVarianceSampler(AdaptiveSampler):
    def __init__(self, model: Model, variance_estimator: Callable[[list[Model], Model], float], **kwargs):
        self.variance_estimator = variance_estimator
        
        super.__init__(self, model)
    
    def _generate(self, N: int, d: int, u: jnp.ndarray, features: jnp.ndarray, key=None) -> jnp.ndarray:
        """
        Generate N samples in the hypercube for D dimensions.
        
        Previous hypercube samples are passed in ``u`` of shape (M x D), alongside their corresponding ``features`` of shape (N x ...).
        New samples of shape (N x D) must be returned.
        """
        raise NotImplementedError