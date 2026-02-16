import jax
import jax.random as jr
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model
from pmrf.constants import FeatureInputT
from pmrf.frequency import Frequency
from pmrf._util import lhs_sample

class AdaptiveSampler(BaseSampler):
    def __init__(
        self,
        model: Model,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        initial_models: list[Model] | int = 10,
        **kwargs,
    ):    
        self.initial_models = list(initial_models) if not isinstance(initial_models, int) else initial_models
        super().__init__(model, frequency=frequency, features=features, **kwargs)
        
    def run(self, N: int | None = None, max_iterations: int | None = None, key=None, plot=None, **kwargs) -> tuple[list[Model], SampleResults]:
        if key is None:
            raise Exception('Key needed for AdaptiveSampler')
        
        d = self.model.num_flat_params
        if isinstance(self.initial_models, int):
            key, initial_key = jr.split(key)
            initial_Us = lhs_sample(self.initial_models, d, key=initial_key)
            initial_thetas = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(initial_Us)
            self.initial_models = [self.model.with_params(theta) for theta in initial_thetas]
        
        initial_thetas = [model.flat_param_values() for model in self.initial_models]
        for theta in initial_thetas:
            self.add_sample(theta, plot=plot)
        
        iteration = 0
        while True:
            key, generate_key = jr.split(key)
            U_next = self._generate(1, d, key=generate_key, **kwargs)
            if U_next is None:
                break
                
            for u in U_next:
                theta = self.inverse_cumulative_distribution_fn(u)
                self.add_sample(theta, plot=plot)
            
            if N is not None and len(self.sampled_params) >= N:
                break
            iteration += 1
            if max_iterations is not None and iteration == max_iterations:
                break            
            
        if max_iterations is not None and iteration == max_iterations:
            self.logger.warning("Maximum iterations were reached during adaptive sampling")

        models = [self.model.with_params(theta) for theta in self.sampled_params]
        results = SampleResults(initial_model=self.model, sampled_models=models, sampled_params=self.sampled_params, sampled_features=self.sampled_features)
        return models, results
    
    def _check_convergence(self, values: list[float], threshold=None, patience=None, spike_reset_ratio=2.0, title=None) -> bool:
        if title is not None:
            prefix = f"Convergence for {title} reached"
        else:
            prefix = "Convergence reached"

        # Ignore values if we detect a sudden spike
        if spike_reset_ratio is not None and len(values) > 1:
            for i in range(len(values) - 1, 0, -1):
                curr = values[i]
                prev = values[i-1]
                
                # Check if error jumped up significantly (e.g. Current > 2x Previous)
                if curr > (prev * spike_reset_ratio):
                    self.logger.info(f"Resetting patience for {title} due to spike.")
                    values = values[i:] # Slice: ignore everything before this
                    break            
        
        # Check if we have converged via the threshold
        if threshold is not None and jnp.all(values[-1] < threshold):
            self.logger.info(f"{prefix} via threshold.")
            return True
            
        # Check if we have converged via maximum patience (no improvement)
        if len(values) >= patience:
            values = list(values)
            best_idx = min(range(len(values)), key=lambda i: values[i])
            no_recent_improvement = len(values) - 1 - best_idx > patience

            if no_recent_improvement:
                self.logger.info(f"{prefix} via maximum patience.")
                return True
        
        return False