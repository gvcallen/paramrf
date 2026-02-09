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
        self.initial_models = list(initial_models) if not isinstance(initial_models, int) else initial_models
        super().__init__(model, frequency=frequency, features=features)
        
    def run(self, N: int | None = None, max_iterations: int = 100, key=None, jit_feature_fn=False, **kwargs) -> tuple[list[Model], SampleResults]:
        if key is None:
            key = jr.key(0)
            
        if isinstance(self.initial_models, int):
            # TODO key should be SPLIT here but we already have 10 simulations without splitting so leave for now
            from pmrf.sampling._algos.latin_hypercube import LatinHypercubeSampler
            self.initial_models = LatinHypercubeSampler(self.model).run(self.initial_models, key=key)[0]            
        
        models = self.initial_models
        param_names = self.model.flat_param_names()
        
        theta_current = [model.flat_param_values() for model in models]
        U_current = [self.cumulative_distribution_fn(model.flat_param_values()) for model in models]
        features = []
        
        sample_idx = 0
        def compute_sample(theta: jnp.ndarray):
            nonlocal sample_idx
            self.logger.info(f"Computing Sample #{sample_idx} with {dict(zip(param_names, theta.tolist()))}")
            features_i = self.feature_fn(theta, jit=jit_feature_fn)
            sample_idx += 1
            return features_i
            
        self.logger.info('Computing initial sample outputs...')
        for theta in theta_current:
            features.append(compute_sample(theta))
        
        iteration = 0
        d = self.model.num_flat_params
        while iteration < max_iterations:
            U_next = self._generate(1, d, jnp.array(U_current), jnp.array(features), key=key, **kwargs)
            if U_next is None:
                break
            
            U_current.extend([u_next for u_next in U_next])
            for u in U_next:
                theta = self.inverse_cumulative_distribution_fn(u)
                features.append(compute_sample(theta))
                theta_current.append(theta)
            
            if N is not None and len(theta_current) >= N:
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