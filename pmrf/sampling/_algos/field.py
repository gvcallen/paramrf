from typing import Callable, Any

import jax
import jax.numpy as jnp
import jax.random as jr

from pmrf.frequency import Frequency
from pmrf.sampling.base import BaseSampler
from pmrf.sampling.adaptive import AdaptiveSampler
from pmrf.models.model import Model
from pmrf._util import lhs_sample

class FieldSampler(AdaptiveSampler):
    """
    Samples new points at the maxima of a scalar field.
    
    At each iteration, the scalar field can first be "trained" using the current samples, and then "evaluated" at new input points.
    For example, this sampler can be used to train a surrogate model that is able to predict the current variance at new sample points.
    Then, this sampler will choose new sample points where that variance is a maximum.
    
    Convergence is reached when the field maxima stops decreasing.
    """
    def __init__(
        self,
        model: Model,
        train_fn: Callable[[jnp.ndarray, jnp.ndarray, Frequency], Any] | None, # params, features, frequency, and `key` is a key-word argument
        eval_fn: Callable[[Any, jnp.ndarray, Frequency], float],
        grid_sampler: BaseSampler | None = None,
        *args,
        **kwargs
    ):
        if not 'frequency' in kwargs:
            raise Exception("SurrogateFieldSampler without a frequency")
        
        if train_fn is None:
            train_fn = lambda params, features: {'params': params, 'features': features}

        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.grid_sampler = grid_sampler
        self.worst_field_values = []
        self.all_field_thetas = []
        self.figure = None
        
        return super().__init__(model=model, *args, **kwargs)

    def _generate(self, N: int, d: int, *, key=None, threshold=None, patience=10, num_grid_per_dim=1024) -> jnp.ndarray | None:
        # Train the field
        self.logger.info(f"Training...")
        key, field_key = jr.split(key)
        field = self.train_fn(self.sampled_params, self.sampled_features, self.frequency, key=field_key)

        # Get the field thetas
        K = num_grid_per_dim * d # Or however you determine grid size
        key, grid_key = jr.split(key)
        if self.grid_sampler is not None:
            grid_models, _ = self.grid_sampler.run(K, key=grid_key)
            theta_grid = jnp.array([model.flat_param_values() for model in grid_models])
        else:
            U_grid = lhs_sample(K, d, key=grid_key)
            theta_grid = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(U_grid)

        # Sample the field values
        key, eval_key = jr.split(key)
        eval_keys = jr.split(eval_key, len(theta_grid))
        def eval_theta_fn(theta, k) -> float:
            return self.eval_fn(field, theta, self.frequency, key=k)        
        grid_field = jax.vmap(eval_theta_fn)(theta_grid, eval_keys)

        # Select N Diverse Points
        selected_field_thetas, selected_field_values = self._select_field_thetas(N, theta_grid, grid_field)
        
        # Log the maximum and append to the errors
        max_field = jnp.max(grid_field)
        self.worst_field_values.append(jnp.max(grid_field))
        self.all_field_thetas.extend(max_field_theta for max_field_theta in selected_field_thetas)
        if N == 1:
            self.logger.info(f"Field maximum = {float(max_field):.2f}")
        else:
            value_str = ""
            for value in selected_field_values:
                value_str += f"{value:.2f}, "
            value_str = value_str[0:len(value_str)-2]
            self.logger.info(f"Field maxima = [{value_str}]")
        
        # Check for convergence
        if self._check_convergence(self.worst_field_values, threshold=threshold, patience=patience, title="field"):
            return None

        # Return the next hypercube samples (U-space)
        return jnp.array([self.cumulative_distribution_fn(theta) for theta in selected_field_thetas])
    
    def _select_field_thetas(self, N: int, thetas: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        """
        Selects N points iteratively using a penalized greedy strategy to ensure diversity.
        """
        # 1. Handle edge cases
        K = len(values)
        if N >= K:
            return thetas, values
        if N <= 0:
            return jnp.array([]), jnp.array([])
        if N == 1:
            best_idx = jnp.argmax(values)
            return thetas[best_idx][None, :], values[best_idx].reshape(1)

        # 2. Normalize inputs for calculation stability
        # We perform distance calculations in a normalized coordinate system 
        # so that parameters with large magnitudes don't dominate the distance metric.
        t_min = jnp.min(thetas, axis=0)
        t_max = jnp.max(thetas, axis=0)
        t_range = t_max - t_min
        # Avoid division by zero
        t_range = jnp.where(t_range == 0, 1.0, t_range) 
        norm_thetas = (thetas - t_min) / t_range

        # Normalize values to [0, 1] to apply consistent penalization
        v_min = jnp.min(values)
        v_max = jnp.max(values)
        # If the field is flat, just return the first N
        if v_max == v_min:
            return thetas[:N]
            
        # Work with a copy of scores that we can modify
        scores = (values - v_min) / (v_max - v_min)

        # 3. Define Penalization Parameters
        # L is the characteristic length scale. 
        # Points within this radius will be heavily suppressed.
        # We use a heuristic: e.g., 5-10% of the normalized domain diagonal.
        d = thetas.shape[1]
        L = 0.1 * jnp.sqrt(d) # 10% of unit hypercube diagonal

        selected_indices = []

        # 4. Iterative Selection Loop (Greedy with Penalization)
        # Note: We use a Python loop here because N is typically small (batch size).
        # This is more readable and flexible than jax.lax.scan for this specific logic.
        for _ in range(N):
            # Pick the point with the highest current score
            best_idx = jnp.argmax(scores)
            selected_indices.append(best_idx)

            # --- The Penalization Step ---
            # Don't penalize on the last step
            if len(selected_indices) < N:
                # Calculate distances from the just-selected point to all other points
                # Use normalized coordinates for distance!
                dist_sq = jnp.sum((norm_thetas - norm_thetas[best_idx])**2, axis=1)
                
                # Gaussian-style penalization multiplier
                # Drops to 0 at dist=0, approaches 1 at dist=infinity
                penalty = 1.0 - jnp.exp(-dist_sq / (2 * L**2))
                
                # Apply penalty to scores (multiplicative suppression)
                scores = scores * penalty
                
                # Hard set the selected index to -infinity so it's not picked again
                scores = scores.at[best_idx].set(-jnp.inf)

        return thetas[jnp.array(selected_indices)], values[jnp.array(selected_indices)]