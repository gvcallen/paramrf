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
    
    def _check_convergence(
        self, 
        values: list[float], 
        threshold: float = None, 
        patience: int = None, 
        min_delta: float = None, 
        smoothing_window: int = 1,
        title: str = None
    ) -> bool:
        
        if not values:
            return False

        prefix = f"Convergence for {title} reached" if title else "Convergence reached"
        
        # 1. Safety Check: Divergence (NaN/Inf)
        current_val = values[-1]
        if not jnp.isfinite(current_val):
            self.logger.warning(f"Stopping: Divergence detected (value is {current_val}).")
            return True

        # Apply smoothing if window > 1
        # We look at the average of the last 'n' items to reduce noise
        if smoothing_window > 1 and len(values) >= smoothing_window:
            val_to_check = jnp.mean(jnp.array(values[-smoothing_window:]))
        else:
            val_to_check = current_val

        # 2. Check Threshold (Absolute Target)
        if threshold is not None:
            # Assuming minimization (val < threshold). Flip logic if maximization.
            if val_to_check < threshold:
                self.logger.info(f"{prefix} via threshold ({val_to_check:.5f} < {threshold}).")
                return True

        # 3. Check Minimum Delta (The "Plateau")
        # Checks if the change between the last two points is negligible
        if min_delta is not None and len(values) > 1:
            diff = abs(values[-1] - values[-2])
            if diff < min_delta:
                self.logger.info(f"{prefix} via min_delta (change {diff:.2e} < {min_delta}).")
                return True

        # 4. Check Patience (Early Stopping)
        if patience is not None and len(values) >= patience:
            # Convert to standard list for easier indexing/slicing if it's a JAX array
            vals_list = list(values)
            
            # Find the index of the best value (min)
            best_idx = min(range(len(vals_list)), key=lambda i: vals_list[i])
            
            # Calculate how many steps have passed since the best value
            steps_since_best = len(vals_list) - 1 - best_idx

            if steps_since_best > patience:
                self.logger.info(f"{prefix} via maximum patience (no improvement for {steps_since_best} steps).")
                return True
                
        return False