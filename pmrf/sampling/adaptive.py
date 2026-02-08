from abc import abstractmethod

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
        
    def run(self, jit=False, max_iterations: int = 100) -> list[Model]:
        models = self.inital_models
        
        params = [model.flat_param_values() for model in models]
        features = []
        
        feature_fn = self.make_feature_function(jit=jit)
        
        self.logger.info('Fetching initial sample outputs...')
        for params_i in params:
            features_i = feature_fn(params_i)
            features.append(features_i)
        
        iteration = 0
        while iteration < max_iterations:
            next_params = self._generate(1, jnp.array(params), jnp.array(features))
            if len(next_params) == 0:
                break
            
            params.extend(next_params)
            for params_i in next_params:
                features_i = feature_fn(params_i)
                features.append(features_i)
            
            iteration += 1

        return models
    
    @abstractmethod
    def _generate(self, N: int, d: int, u: jnp.ndarray, features: jnp.ndarray, key=None) -> jnp.ndarray:
        """
        Generate N samples in the hypercube for D dimensions.
        
        Previous hypercube samples are passed in ``u`` of shape (M x D), alongside their corresponding ``features`` of shape (N x ...).
        New samples of shape (N x D) must be returned.
        """
        raise NotImplementedError