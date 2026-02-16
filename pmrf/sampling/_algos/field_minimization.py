from typing import Callable, Any

import jax
import jax.numpy as jnp
import jax.random as jr

from pmrf.frequency import Frequency
from pmrf.sampling.base import BaseSampler
from pmrf.sampling.adaptive import AdaptiveSampler
from pmrf.models.model import Model
from pmrf._util import lhs_sample, no_recent_improvement

class FieldSampler(AdaptiveSampler):
    """
    Samples new points by minimizing a scalar field that is a function of the input parameters.
    
    At each iteration, the scalar field can first be "trained" using the current samples, and then "evaluated" at new input points.
    For example, this sampler can be used to train a surrogate model that is able to predict the current variance at new sample points.
    Then, this sampler will choose new sample points where that variance is a maximum.
    
    Convergence is reached when the scalar field stops decreasing.
    """
    def __init__(
        self,
        model: Model,
        train_fn: Callable[[jnp.ndarray, jnp.ndarray, Frequency], Any] | None, # params, features, frequency, and `key` is a key-word argument
        eval_fn: Callable[[Any, jnp.ndarray, Frequency], float],
        initial_models: list[Model] | int = 10,
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
        self.field_values = []
        self.figure = None
        
        return super().__init__(model=model, initial_models=initial_models, *args, **kwargs)

    def _generate(self, N: int, d: int, key=None, threshold=None, patience=10, num_grid_per_dim=1024) -> jnp.ndarray | None:
        # For each pass, we train the field model on the current samples and features.
        self.logger.info(f"Training...")
        key, field_key = jr.split(key)
        field = self.train_fn(self.sampled_params, self.sampled_features, self.frequency, key=field_key)

        # Next, we sample new points using the grid sampler (e.g. latin hupercube) to get grid_theta of shape (K, d)
        K = num_grid_per_dim * d
        
        key, grid_key = jr.split(key)
        if self.grid_sampler is not None:
            grid_models, _ = self.grid_sampler.run(K, key=grid_key)
            theta_grid = jnp.array([model.flat_param_values() for model in grid_models])
        else:
            U_grid = lhs_sample(K, d, key=grid_key)
            theta_grid = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(U_grid)

        # We calculate the scalar field at each grid point to get grid_field of shape (K,)
        key, eval_key = jr.split(key)
        eval_keys = jr.split(eval_key, len(theta_grid))
        def eval_theta_fn(theta, k) -> float:
            return self.eval_fn(field, theta, self.frequency, key=k)        
        grid_field = jax.vmap(eval_theta_fn)(theta_grid, eval_keys)

        # Get the largest field value
        max_field = jnp.max(grid_field)
        self.field_values.append(max_field)
        
        # Check if we have converged
        if self._check_convergence(self.field_values, threshold, patience):
            return None

        # Pick the N points in the grid with the largest field values
        self.logger.info(f"Field maximum = {float(max_field):.2f}")
        indices = jnp.argsort(grid_field, descending=True)
        max_field_theta = theta_grid[indices][0:N]
        
        # Return the next hypercube samples
        return jnp.array([self.cumulative_distribution_fn(theta) for theta in max_field_theta])