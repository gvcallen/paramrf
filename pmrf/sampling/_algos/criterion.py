from typing import Callable

import jax
import jax.numpy as jnp
from jax import vmap, random

from pmrf.sampling.adaptive import AdaptiveSampler
from pmrf.models.model import Model

class CriterionSampler(AdaptiveSampler):
    def __init__(
        self,
        model: Model,
        criterion_fn,
        threshold = 0.01,
        initial_models: list[Model] | int = 10,
        *args,
        **kwargs
    ):
        self.criterion_fn = criterion_fn # 
        self.criterion_threshold = threshold
        return super().__init__(model=model, initial_models=initial_models, *args, **kwargs)
    
    def _generate(self, N: int, d: int, samples: jnp.ndarray, features: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N samples in the hypercube for d dimensions.
        
        Note that not all samplers support an arbitrary N.
        
        Previous hypercube samples are passed in ``u`` of shape (M x d), alongside their corresponding ``features`` of shape (N x ...).
        New samples of shape (N x d) must be returned.
        """
        raise NotImplementedError    