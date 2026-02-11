from abc import abstractmethod

import jax
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model

class OneshotSampler(BaseSampler):
    """Generates a fixed number of samples in one go."""
    def run(self, N: int, key=None) -> tuple[list[Model], SampleResults]:
        if key is None:
            raise Exception('key needed for OneshotSampler')
        
        u_samples = self._generate(N, self.model.num_flat_params, key=key)
        params = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(u_samples)
        models = [self.model.with_params(params_i) for params_i in params]
        
        results = SampleResults(
            initial_model=self.model,
            sampled_models=models,
            backened_results=u_samples,
        )
        
        return models, results

    @abstractmethod
    def _generate(self, N: int, d: int, key=None) -> jnp.ndarray:
        """
        Generate N samples in the hypercube for D dimensions.
        """
        pass