from typing import Callable, Sequence, Literal

import jax.numpy as jnp
import equinox as eqx

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.goal import Goal
from pmrf.utils import make_flat_fn, make_reconstruct_fn, validate_bounds

SpaceType = Literal['physical', 'hypercube', 'unbounded']

class OptimizeProblem(eqx.Module):
    model: Model
    frequency: Frequency
    cost_fn: Callable[[Model, Frequency], jnp.ndarray]

    def __init__(self, model: Model, frequency: Frequency, cost: Callable[[Model, Frequency], jnp.ndarray] | Sequence[Goal]):
        self.model = model
        self.frequency = frequency

        if isinstance(cost, Sequence):
            def cost_fn(model, freq):
                return jnp.sum(jnp.array([c(model, freq) for c in cost]))
            self.cost_fn = cost_fn
        else:
            self.cost_fn = cost

    @property
    def flat_param_names(self):
        return self.model.flat_param_names()
    
    def get_initial_guess(self) -> jnp.ndarray:
        return self.model.flat_param_values()

    def get_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        bounds = self.model.distribution().bounds
        initial_guess = self.get_initial_guess()
        validate_bounds(initial_guess, bounds[0], bounds[1], self.flat_param_names)
        return bounds[0], bounds[1]
        
    def make_flat_cost_fn(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        return make_flat_fn(self.cost_fn)
        
    def reconstruct(self, params: jnp.ndarray) -> Model:
        physical_x = params

        _, reconstruct_fn = make_reconstruct_fn(self.model)
        return reconstruct_fn(physical_x)