from typing import Callable
from abc import abstractmethod

import jax.random as jr
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler
from pmrf.sampling._algos.latin_hypercube import LatinHypercubeSampler
from pmrf.models.model import Model

from pmrf.constants import FeatureInputT
from pmrf.frequency import Frequency

class AdaptiveSampler(BaseSampler):
    def __init__(
        self,
        model: Model,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        sparam_kind: str = 'all',        
        initial_models: list[Model] | int = 10,
    ):    
        self.inital_models = list(initial_models) if not isinstance(initial_models, int) else LatinHypercubeSampler(model).run(initial_models)
        super().__init__(model, frequency=frequency, features=features, sparam_kind=sparam_kind, initial_models=initial_models)
        
    def run(self, jit_feature=False, max_iterations: int = 100, key=None, **kwargs) -> list[Model]:
        if key is None:
            key = jr.key(0)
        
        models = self.inital_models
        
        icdf_fn = self.make_inverse_cumulative_distribution_fn()
        cdf_fn = self.make_cumulative_distribution_fn()
        feature_fn = self.make_feature_function(jit=jit_feature)
        
        params = [model.flat_param_values() for model in models]
        U_current = [cdf_fn(model.flat_param_values()) for model in models]
        features = []
        
        self.logger.info('Computing initial sample outputs...')
        for u_next in params:
            features_i = feature_fn(u_next)
            features.append(features_i)
        
        iteration = 0
        d = self.model.num_flat_params
        while iteration < max_iterations:
            U_next = self._generate(1, d, jnp.array(U_current), jnp.array(features), key=key, **kwargs)
            
            if U_next is None == 0:
                break
            
            U_current.extend([u_next for u_next in U_next])
            for u_next in U_next:
                params_next = icdf_fn(u_next)
                features_i = feature_fn(params_next)
                features.append(features_i)
            
            iteration += 1
            
        if iteration == max_iterations:
            self.logger.warning("Maximum iterations were reached during adaptive sampling")

        return models
    
    @abstractmethod
    def _generate(self, N: int, d: int, samples: jnp.ndarray, features: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N samples in the hypercube for d dimensions.
        
        Note that not all samplers support an arbitrary N.
        
        Previous hypercube samples are passed in ``u`` of shape (M x d), alongside their corresponding ``features`` of shape (N x ...).
        New samples of shape (N x d) must be returned.
        """
        raise NotImplementedError