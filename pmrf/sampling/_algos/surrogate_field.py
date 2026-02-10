from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from pmrf.frequency import Frequency
from pmrf.sampling.base import BaseSampler
from pmrf.sampling.adaptive import AdaptiveSampler
from pmrf.models.model import Model
from pmrf.sampling._algos.latin_hypercube import LatinHypercubeSampler
from pmrf._util import has_converged, lhs_sample, no_recent_improvement

class SurrogateFieldSampler(AdaptiveSampler):
    """
    Samples new points by minimizing a scalar field induced by a surrogate model.
    
    For example, if the surrogate model is able to predict the current variance for new samples,
    this sampler will pick new points to minimize that variance until a threshold is met.

    """
    def __init__(
        self,
        model: Model,
        train_fn: Callable[[jnp.ndarray, jnp.ndarray, Frequency], Model], # params, features, frequency, and `key` is a key-word argument
        field_fn: Callable[[Model], float],
        initial_models: list[Model] | int = 10,
        grid_sampler: BaseSampler | None = None,
        *args,
        **kwargs
    ):
        if not 'frequency' in kwargs:
            raise Exception("SurrogateFieldSampler without a frequency")

        self.train_fn = train_fn
        self.field_fn = field_fn
        self.grid_sampler = grid_sampler
        self.field_values = []
        self.figure = None
        
        return super().__init__(model=model, initial_models=initial_models, *args, **kwargs)

    def _generate(self, N: int, d: int, U_samples: jnp.ndarray, features: jnp.ndarray, key=None, patience=5, threshold=None, num_grid_per_dim=1000) -> jnp.ndarray | None:
        # For each pass, we train the surrogate model on the current samples and features.
        self.logger.info(f"Training surrogate...")
        theta_samples = jax.vmap(lambda u: self.inverse_cumulative_distribution_fn(u))(U_samples)
        
        key, surrogate_key = jr.split(key)
        surrogate = self.train_fn(theta_samples, features, self.frequency, key=surrogate_key)

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
        def field_theta_fn(theta) -> float:
            nonlocal self, surrogate
            surrogate_with_params = surrogate.with_params(theta)
            return self.field_fn(surrogate_with_params)
        grid_field = jax.vmap(field_theta_fn)(theta_grid)

        # Get the largest field value
        max_field = jnp.max(grid_field)
        self.field_values.append(max_field)
        
        # Check if we have converged via the threshold
        if threshold is not None and jnp.all(max_field < threshold):
            self.logger.info(f"Convergence reached via threshold (maximum field value of {float(max_field)} is less than threshold {threshold}")
            return None
            
        # Check if we have converged via maximum patience (no improvement)
        if len(self.field_values) >= patience and no_recent_improvement(self.field_values, patience):
            self.logger.info(f"Convergence reached via maximum patience (no improvement over past {patience} samples)")
            return None

        # Pick the N points in the grid with the largest field values
        self.logger.info(f"Maximum = {float(max_field)}")
        indices = jnp.argsort(grid_field, descending=True)
        max_field_theta = theta_grid[indices][0:N]
        
        # Return the next hypercube samples
        return jnp.array([self.cumulative_distribution_fn(theta) for theta in max_field_theta])