from abc import ABC, abstractmethod
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr

from pmrf.sampling.acqusition import AcquisitionSampler
from pmrf.models.model import Model
from pmrf.sampling.base import BaseSampler
from pmrf.util import lhs_sample, LivePlotter
from pmrf.algorithms import has_converged

class FieldSampler(AcquisitionSampler, ABC):
    """
    Abstract Base Class for sampling new points at the maxima of a learnt scalar field.
    
    Subclasses must define how to train the field and how to evaluate it.
    """
    def __init__(self, model: Model, **kwargs):
        super().__init__(model, **kwargs)
        
        self.converge_plotters: list[LivePlotter] = []
        self.converge_values: list[tuple[float, ...]] = []
        self.field = None

    # --------------------------------------------------------------------------
    # Abstract Field Hooks (The Contract)
    # --------------------------------------------------------------------------
    @abstractmethod
    def train_field(self, key=None) -> Any:
        """Trains and returns the scalar field (e.g., a surrogate model)."""
        pass

    @abstractmethod
    def evaluate_field(self, field: Any, theta: jnp.ndarray, key=None) -> float:
        """Evaluates the scalar field at a given physical parameter point."""
        pass

    def calculate_convergence(self, key=None) -> float | Sequence[float] | None:
        """
        Optional: Returns a metric to track convergence. 
        If not overridden, the maximum value of the grid field will be used.
        """
        return None

    # --------------------------------------------------------------------------
    # The Template Method
    # --------------------------------------------------------------------------
    def acquire(
        self, 
        batch_size: int, 
        d: int, 
        *, 
        grid_sampler: BaseSampler | None = None,
        rtol=0.01, 
        atol=None, 
        patience=5, 
        window=5, 
        num_grid_per_dim=1024, 
        key=None,
        **kwargs
    ) -> jnp.ndarray | None:
        
        # 1. Train the field (using subclass implementation)
        key, field_key = jr.split(key)
        self.field = self.train_field(key=field_key)

        # 2. Generate Grid Points
        K = num_grid_per_dim * d
        key, grid_key = jr.split(key)
        
        if grid_sampler is not None:
            grid_models, _ = grid_sampler.run(N=K, key=grid_key)
            theta_grid = jnp.array([m.flat_param_values() for m in grid_models])
        else:
            U_grid = lhs_sample(K, d, key=grid_key)
            theta_grid = jax.vmap(self.icdf)(U_grid)

        # 3. Evaluate Field on the Grid (using subclass implementation)
        key, eval_key = jr.split(key)
        eval_keys = jr.split(eval_key, len(theta_grid))
        
        def eval_theta_fn(theta, k) -> float:
            return self.evaluate_field(self.field, theta, key=k)        
            
        grid_field = jax.vmap(eval_theta_fn)(theta_grid, eval_keys)

        # 4. Select N Diverse Points
        selected_field_thetas, selected_field_values = self._select_field_thetas(batch_size, theta_grid, grid_field)
        
        max_field_value = jnp.max(grid_field)
        if batch_size == 1:
            self.logger.info(f"Field maximum = {float(max_field_value):.2f}")
        else:
            value_str = ", ".join([f"{v:.2f}" for v in selected_field_values])
            self.logger.info(f"Field maxima = [{value_str}]")
        
        # 5. Convergence Tracking (using subclass implementation)
        key, validate_key = jr.split(key)
        conv_metric = self.calculate_convergence(key=validate_key)
        
        if conv_metric is not None:
            new_converge_values = tuple(conv_metric) if isinstance(conv_metric, Sequence) else (conv_metric,)
        else:
            new_converge_values = (float(max_field_value),)
            
        self.converge_values.append(new_converge_values)
            
        # Check for convergence
        converged = True
        has_converged_vals = list(map(list, zip(*self.converge_values)))
        for values in has_converged_vals:
            if not has_converged(values, rtol=rtol, atol=atol, patience=patience, window=window):
                converged = False
                break
        
        for i, new_value in enumerate(new_converge_values):
            if len(self.converge_plotters) <= i:
                self.converge_plotters.append(LivePlotter(f"Convergence {i}", "Iteration", "Value"))
            self.converge_plotters[i].add_point(f"Convergence {i}", new_value)
        
        # 6. Return mapped hypercube points
        if converged:
            return None
            
        return jnp.array([self.cdf(theta) for theta in selected_field_thetas])

    def _select_field_thetas(self, N: int, thetas: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
        """Selects N points iteratively using a penalized greedy strategy to ensure diversity."""
        K = len(values)
        if N >= K: return thetas, values
        if N <= 0: return jnp.array([]), jnp.array([])
        if N == 1:
            best_idx = jnp.argmax(values)
            return thetas[best_idx][None, :], values[best_idx].reshape(1)

        t_min, t_max = jnp.min(thetas, axis=0), jnp.max(thetas, axis=0)
        t_range = jnp.where((t_max - t_min) == 0, 1.0, t_max - t_min) 
        norm_thetas = (thetas - t_min) / t_range

        v_min, v_max = jnp.min(values), jnp.max(values)
        if v_max == v_min: return thetas[:N], values[:N]
            
        scores = (values - v_min) / (v_max - v_min)
        L = 0.1 * jnp.sqrt(thetas.shape[1]) 

        selected_indices = []
        for _ in range(N):
            best_idx = jnp.argmax(scores)
            selected_indices.append(best_idx)

            if len(selected_indices) < N:
                dist_sq = jnp.sum((norm_thetas - norm_thetas[best_idx])**2, axis=1)
                penalty = 1.0 - jnp.exp(-dist_sq / (2 * L**2))
                scores = scores * penalty
                scores = scores.at[best_idx].set(-jnp.inf)

        return thetas[jnp.array(selected_indices)], values[jnp.array(selected_indices)]