from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model

class OneshotSampler(BaseSampler, ABC):
    """Generates a fixed number of samples in one go."""
    def run(self, N: int, *, batch_size: int = 1, plot=None, key=None) -> tuple[list[Model], SampleResults]:
        if key is None:
            raise Exception('key needed for OneshotSampler')
        
        U_samples = self._generate(N, self.model.num_flat_params, key=key)
        thetas = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(U_samples)
        
        num_samples = len(thetas)
        for i in range(0, num_samples, batch_size):
            batch_theta = thetas[i : i + batch_size]
            self.add_samples(batch_theta, plot=plot)
        
        models = [self.model.with_params(theta) for theta in thetas]
        results = SampleResults(initial_model=self.model, sampled_models=models, sampled_params=self.sampled_params, sampled_features=self.sampled_features)
        return models, results

    @abstractmethod
    def _generate(self, N: int, d: int, *, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N new samples in the hypercube for d dimensions.
        """
        raise NotImplementedError     