from abc import abstractmethod

import jax.random as jr
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model

from pmrf.constants import FeatureInputT
from pmrf.frequency import Frequency

class AdaptiveSampler(BaseSampler):
    def __init__(
        self,
        model: Model,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        initial_models: list[Model] | int = 10,
    ):    
        from pmrf.sampling._algos.latin_hypercube import LatinHypercubeSampler
        
        self.inital_models = list(initial_models) if not isinstance(initial_models, int) else LatinHypercubeSampler(model).run(initial_models)[0]
        super().__init__(model, frequency=frequency, features=features)
        
    def run(self, N: int | None = None, max_iterations: int = 100, key=None, jit_feature_fn=False, **kwargs) -> tuple[list[Model], SampleResults]:
        if key is None:
            key = jr.key(0)
        
        models = self.inital_models
        
        theta_current = [model.flat_param_values() for model in models]
        U_current = [self.cumulative_distribution_fn(model.flat_param_values()) for model in models]
        features = []
        
        self.logger.info('Computing initial sample outputs...')
        for theta in theta_current:
            features_i = self.feature_fn(theta, jit=jit_feature_fn)
            features.append(features_i)
        
        iteration = 0
        d = self.model.num_flat_params
        while iteration < max_iterations:
            U_next = self._generate(1, d, jnp.array(U_current), jnp.array(features), key=key, **kwargs)
            if U_next is None == 0:
                break
            
            U_current.extend([u_next for u_next in U_next])
            for u_next in U_next:
                theta_next = self.inverse_cumulative_distribution_fn(u_next)
                theta_current.append(theta_next)
                features_i = self.feature_fn(theta_next, jit=jit_feature_fn)
                features.append(features_i)
            
            if len(theta_current) >= N:
                break
            iteration += 1
            
        if iteration == max_iterations:
            self.logger.warning("Maximum iterations were reached during adaptive sampling")

        results = SampleResults()
        models = [self.model.with_params(theta) for theta in theta_current]
        return models, results
    
    @abstractmethod
    def _generate(self, N: int, d: int, samples: jnp.ndarray, features: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N samples in the hypercube for d dimensions.
        
        Note that not all samplers support an arbitrary N.
        
        Previous hypercube samples are passed in ``u`` of shape (M x d), alongside their corresponding ``features`` of shape (N x ...).
        New samples of shape (N x d) must be returned.
        """
        raise NotImplementedError