from abc import ABC

import jax
import jax.numpy as jnp
from typing import Literal

# Assuming imports for BaseOptimizer, Goal, Model, Frequency
from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.optimize.base import BaseOptimizer
from pmrf.optimize.goal import Goal

class FrequentistOptimizer(BaseOptimizer, ABC):
    """
    A frequentist optimization wrapper that translates design goals into a 
    JAX-compiled mathematical penalty function.
    """
    def __init__(
        self,
        model: Model,
        goals: list[Goal],
        *,
        frequency: Frequency | None = None,
        aggregation: Literal['least_squares', 'minimax'] = 'least_squares',
        tikhonov_lambda: float = 0.0,
        **feature_kwargs
    ):
        if 'features' in feature_kwargs:
            raise Exception("'features' should not be passed to a frequentist optimizer. Pass goals instead.")

        super().__init__(model, goals, frequency=frequency, **feature_kwargs)
        
        self.aggregation = aggregation
        self.tikhonov_lambda = tikhonov_lambda
        self.initial_theta = self.model.flat_param_values()
        
        # Calculate safe parameter ranges for normalization in Tikhonov penalty
        lower_bounds, upper_bounds = self.model.distribution().bounds
        param_ranges = jnp.array(upper_bounds) - jnp.array(lower_bounds)
        self.param_ranges = jnp.where(param_ranges == 0.0, 1.0, param_ranges)
        
        self._cost_fn = None

    def cost(self, theta: jnp.ndarray) -> jnp.ndarray:
        r"""
        Calculate the scalar penalty (cost) for the current parameters.
        Lazily compiles the evaluation graph via ``jax.jit``.
        """
        if self._cost_fn is None:
            self.logger.debug("Lazily compiling Frequentist Goal Cost function...")
            
            # Capture class attributes for the JAX closure at compile time
            reg_weight = self.tikhonov_lambda
            theta_0 = self.initial_theta
            p_ranges = self.param_ranges
            goals = self.goals
            agg_type = self.aggregation

            @jax.jit
            def cost_fn(theta):
                # Simulated features shape is roughly (N_frequencies, N_goals)
                simulated_features = self.model_features(theta)
                
                # Arrays to hold the total penalty per goal
                penalties = []
                
                for i, goal in enumerate(goals):
                    # Handle shapes safely: slice the i-th feature column
                    if simulated_features.ndim == 2:
                        val = simulated_features[:, i]
                    else:
                        val = simulated_features[i]

                    val = jnp.real(val)
                        
                    # 1. Apply Inequality/Equality operator
                    if goal.operator == '<':
                        penalty = jnp.maximum(0.0, val - goal.target)
                    elif goal.operator == '>':
                        penalty = jnp.maximum(0.0, goal.target - val)
                    elif goal.operator == '==':
                        penalty = jnp.abs(val - goal.target)
                        
                    # 2. Apply Frequency Mask and calculate N points
                    if goal.mask is not None:
                        penalty = jnp.where(goal.mask, penalty, 0.0)
                        # Sum the boolean mask to get the number of active points
                        num_points = jnp.sum(goal.mask) 
                    else:
                        num_points = val.shape[0]
                        
                    # Prevent division by zero if a mask is entirely False
                    num_points = jnp.maximum(num_points, 1.0)

                    # 3. Square, normalize by number of points, and weight
                    goal_cost = goal.weight * (jnp.sum(penalty ** 2) / num_points)
                    penalties.append(goal_cost)
                
                penalties_array = jnp.array(penalties)
                
                # 4. Aggregate across all goals
                if agg_type == 'least_squares':
                    base_cost = jnp.sum(penalties_array)
                elif agg_type == 'minimax':
                    # Minimax strictly targets the worst-violating goal
                    base_cost = jnp.max(penalties_array) 

                # 5. Apply Parameter Regularization
                if reg_weight > 0.0 and theta_0 is not None:
                    normalized_diff = (theta - theta_0) / p_ranges
                    l2_penalty = reg_weight * jnp.sum(normalized_diff ** 2)
                    return base_cost + l2_penalty
                
                return base_cost

            self._cost_fn = cost_fn

        return self._cost_fn(jnp.array(theta))