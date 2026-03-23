import abc
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from pmrf.utils.random import lhs_sample


class SamplerState(eqx.Module):
    """
    Standardized state object passed between iteration steps.
    """
    sampled_params: jnp.ndarray
    sampled_features: jnp.ndarray
    num_samples: int
    # Optional field for backends to store surrogate models or convergence metrics
    backend_state: Any = None 


class AbstractSampler(eqx.Module):
    """
    Abstract base class for all exploration engines, mirroring the Optimistix 
    AbstractIterativeSolver paradigm.
    """
    
    @abc.abstractmethod
    def init(
        self, 
        eval_fn: Callable[[jnp.ndarray], jnp.ndarray], 
        d: int, 
        key: jax.Array, 
        options: dict[str, Any]
    ) -> SamplerState:
        """Initialize the sampler state and perform the first evaluations."""
        pass

    @abc.abstractmethod
    def step(
        self, 
        eval_fn: Callable[[jnp.ndarray], jnp.ndarray], 
        d: int, 
        state: SamplerState, 
        key: jax.Array, 
        options: dict[str, Any]
    ) -> SamplerState:
        """Perform one iteration of the sampling algorithm."""
        pass

    @abc.abstractmethod
    def terminate(self, state: SamplerState, target_N: int) -> bool:
        """Determine whether the sampling loop should stop."""
        pass


# --------------------------------------------------------------------------
# One-Shot Samplers
# --------------------------------------------------------------------------

class AbstractOneShotSampler(AbstractSampler):
    """
    One-shot samplers generate all points immediately during `init`.
    `step` is effectively a no-op, and `terminate` always returns True.
    """
    
    @abc.abstractmethod
    def generate(self, N: int, d: int, key: jax.Array) -> jnp.ndarray:
        """Propose N points within the unit hypercube."""
        pass
        
    def init(self, eval_fn, d, key, options) -> SamplerState:
        # For one-shot, we assume target_N is passed in options
        N = options.get("target_N", 100)
        U = self.generate(N, d, key)
        thetas, features = eval_fn(U)
        return SamplerState(
            sampled_params=thetas, 
            sampled_features=features, 
            num_samples=N
        )

    def step(self, eval_fn, d, state, key, options) -> SamplerState:
        return state

    def terminate(self, state, target_N) -> bool:
        return True


class LatinHypercube(AbstractOneShotSampler):
    """Sampler using Latin Hypercube Sampling (LHS)."""
    def generate(self, N: int, d: int, key: jax.Array) -> jnp.ndarray:
        return lhs_sample(N, d, key)


class Uniform(AbstractOneShotSampler):
    """Sampler using uniform random sampling."""
    def generate(self, N: int, d: int, key: jax.Array) -> jnp.ndarray:
        return jax.random.uniform(key, shape=(N, d))


# --------------------------------------------------------------------------
# Adaptive (Active Learning) Samplers
# --------------------------------------------------------------------------

class AbstractAdaptiveSampler(AbstractSampler):
    """
    Adaptive samplers iterate until a budget is reached or convergence is met.
    """
    initial_models: int = 10
    batch_size: int = 1
    
    def init(self, eval_fn, d, key, options) -> SamplerState:
        # Initialize with LHS
        U_init = lhs_sample(self.initial_models, d, key)
        thetas, features = eval_fn(U_init)
        
        return SamplerState(
            sampled_params=thetas,
            sampled_features=features,
            num_samples=self.initial_models
        )
        
    def terminate(self, state: SamplerState, target_N: int) -> bool:
        # Stop if we hit the sample budget
        return state.num_samples >= target_N


class FieldSampler(AbstractAdaptiveSampler):
    """Samples new points at the maxima of a learned scalar field."""
    num_grid_per_dim: int = 1024
    grid_sampler: AbstractOneShotSampler = LatinHypercube()

    @abc.abstractmethod
    def train_field(self, params: jnp.ndarray, features: jnp.ndarray, key: jax.Array) -> Any:
        pass

    @abc.abstractmethod
    def evaluate_field(self, field: Any, theta: jnp.ndarray, key: jax.Array) -> float:
        pass

    def step(self, eval_fn, d, state, key, options) -> SamplerState:
        key, field_key, grid_key, eval_key = jr.split(key, 4)
        
        # 1. Train the field on current state
        field = self.train_field(state.sampled_params, state.sampled_features, field_key)

        # 2. Generate Candidate Grid (Hypercube)
        K = self.num_grid_per_dim * d
        U_grid = self.grid_sampler.generate(K, d, grid_key)
        
        # 3. Evaluate Field on Grid
        eval_keys = jr.split(eval_key, K)
        grid_field = jax.vmap(lambda u, k: self.evaluate_field(field, u, k))(U_grid, eval_keys)

        # 4. Greedy Diversity Selection to get new hypercube proposals
        U_next, _ = self._select_field_points(self.batch_size, U_grid, grid_field)
        
        # 5. Evaluate the physical model (using the black-box function)
        new_thetas, new_features = eval_fn(U_next)
        
        # 6. Update State
        return SamplerState(
            sampled_params=jnp.vstack((state.sampled_params, new_thetas)),
            sampled_features=jnp.vstack((state.sampled_features, new_features)),
            num_samples=state.num_samples + len(new_thetas),
            backend_state=field
        )

    def _select_field_points(self, N: int, points: jnp.ndarray, values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Selects N points iteratively using a penalized greedy strategy."""
        if N >= len(values): return points, values
        if N == 1:
            best_idx = jnp.argmax(values)
            return points[best_idx][None, :], values[best_idx].reshape(1)

        p_min, p_max = jnp.min(points, axis=0), jnp.max(points, axis=0)
        p_range = jnp.where((p_max - p_min) == 0, 1.0, p_max - p_min) 
        norm_points = (points - p_min) / p_range

        v_min, v_max = jnp.min(values), jnp.max(values)
        if v_max == v_min: return points[:N], values[:N]
            
        scores = (values - v_min) / (v_max - v_min)
        L = 0.1 * jnp.sqrt(points.shape[1]) 

        selected_indices = []
        for _ in range(N):
            best_idx = jnp.argmax(scores)
            selected_indices.append(best_idx)

            if len(selected_indices) < N:
                dist_sq = jnp.sum((norm_points - norm_points[best_idx])**2, axis=1)
                penalty = 1.0 - jnp.exp(-dist_sq / (2 * L**2))
                scores = scores * penalty
                scores = scores.at[best_idx].set(-jnp.inf)

        idx_array = jnp.array(selected_indices)
        return points[idx_array], values[idx_array]


class EqxLearnUncertainty(FieldSampler):
    """Adaptive sampler targeting regions of high surrogate uncertainty."""
    surrogate: Any
    fit_kwargs: dict = eqx.field(default_factory=dict)

    def train_field(self, params: jnp.ndarray, features: jnp.ndarray, key: jax.Array) -> Any:
        from eqxlearn import fit
        
        # Flatten features for eqx-learn
        flat_features = features.reshape(features.shape[0], -1)
        fitted_model, _ = fit(self.surrogate, X=params, y=flat_features, key=key, **self.fit_kwargs)
        return fitted_model

    def evaluate_field(self, field: Any, theta: jnp.ndarray, key: jax.Array) -> float:
        _y_mean, y_var = field(theta, return_var=True)
        rayleigh_factor = jnp.sqrt(jnp.pi) / 2.0
        total_std = jnp.sqrt(y_var.real + y_var.imag)
        expected_mae = jnp.mean(rayleigh_factor * total_std)
        return 20 * jnp.log10(expected_mae)