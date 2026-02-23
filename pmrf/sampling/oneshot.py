from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model
from pmrf._util.random import generate_key

class OneshotSampler(BaseSampler, ABC):
    """Generates a fixed number of samples in one go."""
    def run(self, N: int, *, batch_size: int = None, return_models=True, return_batched=False, plot=None, key=None) -> tuple[list[Model], SampleResults]:
        if key is None:
            key = generate_key()
        
        U_samples = self._generate(N, self.model.num_flat_params, key=key)
        thetas = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(U_samples)
        
        num_samples = len(thetas)
        if batch_size is not None:
            for i in range(0, num_samples, batch_size):
                batch_theta = thetas[i : i + batch_size]
                self.add_samples(batch_theta, plot=plot)
        else:
            self.add_samples(thetas, plot=plot)
        
        models = None
        if return_models:
            models = self.construct_models(thetas, return_batched=return_batched)
        # models = [eqx.combine(model_with_params(theta), static) for theta in thetas]
        # models = [self.model.with_params(theta) for theta in thetas]
        results = SampleResults(initial_model=self.model, sampled_models=models, sampled_params=self.sampled_params, sampled_features=self.sampled_features)

        if return_models:
            return models, results
        return results

    @abstractmethod
    def _generate(self, N: int, d: int, *, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N new samples in the hypercube for d dimensions.
        """
        raise NotImplementedError     
    
    def construct_models(self, thetas, return_batched=False) -> list[Model]:
        N = thetas.shape[0]
        base_model = self.model
        
        if return_batched:
            batched_model = eqx.filter_vmap(base_model.with_params)(thetas)
            return batched_model
        
        return [base_model.with_params(theta) for theta in thetas]
