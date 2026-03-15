from abc import ABC
from typing import Literal

import jax
import jax.numpy as jnp

from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.optimize.base import BaseOptimizer
from pmrf.optimize.goal import Goal

class BayesianOptimizer(BaseOptimizer, ABC):
    r"""
    A Bayesian optimization wrapper that translates design goals into a 
    log-likelihood function for design space exploration.
    """
    def __init__(
        self,
        model: Model,
        goals: list[Goal],
        *,
        frequency: Frequency | None = None,
        aggregation: Literal['least_squares', 'minimax'] = 'least_squares',
        **feature_kwargs
    ):
        if 'features' in feature_kwargs:
            raise Exception("'features' should not be passed to a Bayesian optimizer. Pass goals instead.")

        super().__init__(model, goals, frequency=frequency, **feature_kwargs)
        
        self.aggregation = aggregation
        
        # Function caches for lazy compilation
        self._cdf_fn = None
        self._icdf_fn = None
        self._log_prior_fn = None
        self._log_likelihood_fn = None

    @property
    def num_params(self) -> int:
        r"""
        int: Total number of active parameters. 
        For optimization, this is strictly the model parameters.
        """
        return self.model.num_flat_params

    def cdf(self, theta: jnp.ndarray) -> jnp.ndarray:
        if self._cdf_fn is None:
            model_distribution = self.model.distribution()
            @jax.jit
            def cdf_fn(u):
                return model_distribution.cdf(u)
            self._cdf_fn = cdf_fn
        return self._cdf_fn(jnp.array(theta))

    def icdf(self, u: jnp.ndarray) -> jnp.ndarray:
        if self._icdf_fn is None:
            model_distribution = self.model.distribution()
            @jax.jit
            def icdf_fn(u):
                return model_distribution.icdf(u)
            self._icdf_fn = icdf_fn
        return self._icdf_fn(jnp.array(u))

    def log_prior(self, theta: jnp.ndarray) -> jnp.ndarray:
        if self._log_prior_fn is None:
            self.logger.debug('Lazily compiling log-prior...')
            model_dist = self.model.distribution()
            
            @jax.jit
            def prior_fn(p):
                return jnp.sum(model_dist.log_prob(p))
                
            self._log_prior_fn = prior_fn
        return self._log_prior_fn(jnp.array(theta))

    def log_likelihood(self, theta: jnp.ndarray) -> jnp.ndarray:
        r"""
        Evaluate the synthetic log-likelihood of the design goals.
        
        This lazily compiles the penalty graph. The log-likelihood is 
        defined as the negative cost. When all goals are perfectly met, 
        the penalty is 0, and the log-likelihood peaks at 0.
        """
        if self._log_likelihood_fn is None:
            self.logger.debug("Lazily compiling Bayesian Goal Likelihood function...")
            
            goals = self.goals
            agg_type = self.aggregation

            @jax.jit
            def ll_fn(theta_vals):
                simulated_features = self.model_features(theta_vals)
                penalties = []
                
                for i, goal in enumerate(goals):
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
                        
                    # 2. Apply Frequency Mask
                    if goal.mask is not None:
                        penalty = jnp.where(goal.mask, penalty, 0.0)
                        num_points = jnp.sum(goal.mask) 
                    else:
                        num_points = val.shape[0]
                        
                    num_points = jnp.maximum(num_points, 1.0)

                    # 3. Square, normalize by number of points, and weight
                    goal_cost = goal.weight * (jnp.sum(penalty ** 2) / num_points)
                    penalties.append(goal_cost)
                
                penalties_array = jnp.array(penalties)
                
                # 4. Aggregate across all goals
                if agg_type == 'least_squares':
                    base_cost = jnp.sum(penalties_array)
                elif agg_type == 'minimax':
                    base_cost = jnp.max(penalties_array) 

                # The Likelihood is the negative cost.
                # Maximum likelihood (0.0) is achieved when all goals are met.
                return -base_cost

            self._log_likelihood_fn = ll_fn

        return self._log_likelihood_fn(jnp.array(theta))