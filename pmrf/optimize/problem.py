from typing import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.scipy.special as jss
from jax import flatten_util
import equinox as eqx

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.optimize.goal import Goal
from pmrf.utils import make_reconstruct_fn, make_flat_fn, validate_bounds

@dataclass
class OptimizeProblem:
    model: Model
    frequency: Frequency
    cost_fn: Callable[[Model, Frequency], jnp.ndarray]

    def __init__(self, model: Model, frequency: Frequency, cost: Callable[[Model, Frequency], jnp.ndarray] | Sequence[Goal]):
        """
        model : Model
            The initial model containing the parameters to optimize.
        frequency : Frequency
            The frequency grid to evaluate the cost over.
        cost : Callable[[Model, Frequency], float | jnp.ndarray] | list[Goal]
            A scalar cost to minimize. Can either be an arbitrary callable or a list of goals.
            For the latter case, the goals are simply summed. See the :meth:`pmrf.features.Goal`.
        """        
        self.model = model

        if isinstance(cost, Sequence):
            def cost_fn(model, freq):
                return jnp.sum(jnp.array([c(model, freq) for c in cost]))
            self.cost_fn = cost_fn
        else:
            self.cost_fn = cost
        
        self.frequency = frequency

    @property
    def flat_param_names(self):
        return self.model.flat_param_names()
    
    @property
    def flat_param_values(self):
        return self.model.flat_param_values()
    
    @property
    def bounds(self):
        bounds = self.model.distribution().bounds
        validate_bounds(self.flat_param_values, bounds[0], bounds[1], self.flat_param_names)
        return bounds
    
    @property
    def reconstruct_fn(self):
        x0, reconstruct_fn = make_reconstruct_fn(self.model)
        return reconstruct_fn
    
    @property
    def flat_cost_fn(self):
        return make_flat_fn(self.cost)

    @property
    def flat_prob_initial_guess(self) -> jnp.ndarray:
        """Maps physical parameters to the [0, 1] probability space using CDF."""
        u = self.model.distribution().cdf(self.flat_param_values)
        # Clip to avoid exact 0 or 1, which causes infinities in logit/icdf
        return jnp.clip(u, 1e-7, 1.0 - 1e-7)

    @property
    def flat_unbounded_initial_guess(self) -> jnp.ndarray:
        """Maps physical parameters to the [-inf, inf] unbounded space using logit."""
        u = self.flat_prob_initial_guess
        return jss.logit(u)

    def make_prob_cost_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Returns a flat cost function that expects inputs in [0, 1] space."""
        dist = self.model.distribution()
        flat_physical_cost = self.flat_cost_fn

        def prob_cost(u: jnp.ndarray) -> jnp.ndarray:
            u_clipped = jnp.clip(u, 1e-7, 1.0 - 1e-7)
            physical_x = dist.icdf(u_clipped)
            return flat_physical_cost(physical_x)
        return prob_cost

    def make_unbounded_cost_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Returns a flat cost function that expects inputs in [-inf, inf] space."""
        dist = self.model.distribution()
        flat_physical_cost = self.flat_cost_fn

        def unbounded_cost(y: jnp.ndarray) -> jnp.ndarray:
            u = jnn.sigmoid(y)
            u_clipped = jnp.clip(u, 1e-7, 1.0 - 1e-7)
            physical_x = dist.icdf(u_clipped)
            return flat_physical_cost(physical_x)
        return unbounded_cost