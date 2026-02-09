from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from pmrf.frequency import Frequency
from pmrf.sampling.base import BaseSampler
from pmrf.sampling.adaptive import AdaptiveSampler
from pmrf.models.model import Model
from pmrf.sampling._algos.latin_hypercube import LatinHypercubeSampler

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
        threshold,
        num_success=3,
        num_grid_per_dim=1000,
        initial_models: list[Model] | int = 10,
        grid_sampler: BaseSampler | None = None,
        *args,
        **kwargs
    ):
        if not 'frequency' in kwargs:
            raise Exception("SurrogateFieldSampler without a frequency")

        self.train_fn = train_fn
        self.field_fn = field_fn
        self.threshold = threshold
        self.num_grid_per_dim = num_grid_per_dim
        self.num_success = num_success
        self.grid_sampler = grid_sampler if grid_sampler is not None else LatinHypercubeSampler(model)
        
        return super().__init__(model=model, initial_models=initial_models, *args, **kwargs)

    def _generate(self, N: int, d: int, U_samples: jnp.ndarray, features: jnp.ndarray, key=None) -> jnp.ndarray | None:
        # For each pass, we train the surrogate model on the current samples and features.
        self.logger.info(f"Training surrogate...")
        theta_samples = jnp.array([self.inverse_cumulative_distribution_fn(u) for u in U_samples])
        
        key, surrogate_key = jax.random.split(key)
        surrogate = self.train_fn(theta_samples, features, self.frequency, key=surrogate_key)

        # Next, we sample new points using the grid sampler (e.g. latin hupercube) to get grid_theta of shape (K, d)
        K = self.num_grid_per_dim * d
        grid_models, _ = self.grid_sampler.run(K)
        grid_theta = jnp.array([model.flat_param_values() for model in grid_models])

        # We calculate the scalar field at each grid point to get grid_field of shape (K,)
        def field_theta_fn(theta) -> float:
            nonlocal self, surrogate
            surrogate_with_params = surrogate.with_params(theta)
            return self.field_fn(surrogate_with_params)
        grid_field = jax.vmap(field_theta_fn)(grid_theta)

        # Pick the N points in the grid with the largest field values
        indices = jnp.argsort(grid_field, descending=True)
        max_field = grid_field[indices][0:N]
        
        if jnp.all(max_field < self.threshold):
            self.logger.info(f"Convergence reached. Maximum field value of {float(max_field)} is less than threshold {self.threshold}")
            return None
        else:
            self.logger.info(f"Maximum = {float(max_field)}, threshold = {self.threshold}")

        # Return the next hypercube samples
        max_field_theta = grid_theta[indices][0:N]
        return jnp.array([self.cumulative_distribution_fn(theta) for theta in max_field_theta])