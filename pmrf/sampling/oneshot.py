import jax

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model

class OneshotSampler(BaseSampler):
    """Generates a fixed number of samples in one go."""
    def run(self, N: int, plot=None, key=None) -> tuple[list[Model], SampleResults]:
        if key is None:
            raise Exception('key needed for OneshotSampler')
        
        U_samples = self._generate(N, self.model.num_flat_params, key=key)
        thetas = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(U_samples)
        
        for theta in thetas:
            self.add_sample(theta, plot=plot)    
        
        models = [self.model.with_params(theta) for theta in thetas]
        results = SampleResults(initial_model=self.model, sampled_models=models, sampled_params=self.sampled_params, sampled_features=self.sampled_features)
        return models, results