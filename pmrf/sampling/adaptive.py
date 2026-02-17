from abc import ABC, abstractmethod
import jax
import jax.random as jr
import jax.numpy as jnp

from pmrf.sampling.base import BaseSampler, SampleResults
from pmrf.models.model import Model
from pmrf.constants import FeatureInputT
from pmrf.frequency import Frequency
from pmrf._util import lhs_sample

class AdaptiveSampler(BaseSampler, ABC):
    def __init__(
        self,
        model: Model,
        *,
        frequency: Frequency | None = None,
        features: FeatureInputT | None = None,
        initial_models_factor: int | None = 10,
        initial_models: list[Model] | int | None = None,
        **kwargs,
    ):
        if initial_models_factor is not None and initial_models is not None:
            raise Exception("Cannot pass both initial_models_factor and initial_models to AdaptiveSampler")
        if initial_models_factor is not None:
            initial_models = initial_models_factor * model.num_flat_params
        
        self.initial_models = list(initial_models) if not isinstance(initial_models, int) else initial_models
        super().__init__(model, frequency=frequency, features=features, **kwargs)

    def run(self, N: int | None = None, *, batch_size: int = 1, max_iterations: int | None = None, key=None, plot=None, **kwargs) -> tuple[list[Model], SampleResults]:
        if key is None:
            raise Exception('Key needed for AdaptiveSampler')
        
        d = self.model.num_flat_params
        if isinstance(self.initial_models, int):
            key, initial_key = jr.split(key)
            initial_Us = lhs_sample(self.initial_models, d, key=initial_key)
            initial_thetas = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(initial_Us)
            self.initial_models = [self.model.with_params(theta) for theta in initial_thetas]
        
        initial_thetas = jnp.array([model.flat_param_values() for model in self.initial_models])
        num_initial_samples = len(initial_thetas)
        for i in range(0, num_initial_samples, batch_size):
            batch_theta = initial_thetas[i : i + batch_size]
            self.add_samples(batch_theta, plot=plot)
            
        
        iteration = 0
        while True:
            # We try ask for self.batch_size samples at a time, but may receive less
            key, generate_key = jr.split(key)
            
            U_next = self._generate(batch_size, d, key=generate_key, **kwargs)
            
            if U_next is None:
                break
                
            thetas = jnp.array([self.inverse_cumulative_distribution_fn(u) for u in U_next])
            num_samples = len(thetas)
            for i in range(0, num_samples, batch_size):
                batch_theta = thetas[i : i + batch_size]
                self.add_samples(batch_theta, plot=plot)            
            
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

    @abstractmethod
    def _generate(self, N: int, d: int, *, key=None, **kwargs) -> jnp.ndarray:
        """
        Generate N new samples in the hypercube for d dimensions.
        
        It is not a requirement to return N samples: 1 <= n_samples < N may returned.
        To signify convergence, `None` may be returned.
        
        Note that `self.sampled_params` and `self.sampled_features` may be inspected at each iteration.
        """
        raise NotImplementedError            
            
        
    def _check_convergence(self, values, *, threshold=None, patience=None, iqr_factor=1.5, relative_epsilon=0.05, title=None) -> bool:
        if title is not None:
            prefix = f"Convergence for {title} reached"
        else:
            prefix = "Convergence reached"

        values = jnp.array(values)
        
        # Check if we have converged via the threshold
        if threshold is not None and values[-1] < threshold:
            self.logger.info(f"{prefix} via threshold.")
            return True            

        # Check if we have converged via maximum patience (no improvement in last N samples)
        if patience is not None and len(values) >= 2 * patience + 1:
            # 1. Spike Detection
            # We iterate BACKWARDS (newest -> oldest) in the window.
            # If the newest value is a spike, we log it.
            # If an older value in the window is a spike, we detect it (so we stop convergence), 
            # but we don't log it again.
            spike_detected = False
            
            for idx in range(len(values) - 1, len(values) - patience - 1, -1):
                target_value = values[idx]
                
                # The history for this specific target value is the N points before it
                history_start = idx - patience
                history_end = idx
                history_window = values[history_start:history_end]
                
                # Calculate IQR on that history
                Q1, Q3 = jnp.percentile(jnp.array(history_window), 25), jnp.percentile(jnp.array(history_window), 75)
                actual_iqr = Q3 - Q1
                min_iqr_floor = (jnp.abs(jnp.median(jnp.array(history_window))) * relative_epsilon)
                effective_iqr = jnp.maximum(actual_iqr, min_iqr_floor)
                
                iqr_threshold = Q3 + effective_iqr * iqr_factor
                
                if target_value > iqr_threshold:
                    spike_detected = True
                    
                    # ONLY log if the spike is the very latest value added
                    if idx == len(values) - 1:
                        self.logger.info(f"Spike detected for {title} (value {target_value:.4f}). Skipping patience check.")
                    
                    # We break immediately because one spike in the window is enough 
                    # to invalidate convergence.
                    break 

            # 2. Patience Check (only if no spikes found in window)
            if not spike_detected:
                current_window = values[-patience:]
                overall_best_so_far = jnp.min(values[:-patience])
                window_best = jnp.min(current_window)
                
                # If the best value in our recent window is NOT better (smaller) than
                # the best value we had before the window started... we have stagnated.
                if window_best >= overall_best_so_far:
                    self.logger.info(f"{prefix} via maximum patience (no improvement in last {patience} steps).")
                    return True

        return False