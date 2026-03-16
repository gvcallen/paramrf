from typing import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.optimize.goal import Goal
from pmrf.utils import make_reconstruct_fn, make_flat_fn, validate_bounds

@dataclass
class FrequentistProblem:
    x0: jnp.ndarray
    bounds: tuple[jnp.ndarray, jnp.ndarray]
    cost_fn: Callable[[Model, jnp.ndarray], jnp.ndarray]
    flat_cost_fn: Callable[[jnp.ndarray], jnp.ndarray]
    reconstruct_fn: Callable[[jnp.ndarray], Model]

    def __init__(self, model: Model, cost: Callable[[Model, Frequency], jnp.ndarray] | list[Goal], frequency: Frequency):
        """
        model : Model
            The initial ParamRF model containing free parameters to optimize.
        cost : Callable[[Model, Frequency], float | jnp.ndarray] | list
            A custom function evaluating the model over frequency and returning a scalar loss.
            A list of callables can be passed, in which case they are simply summed.
            See the :meth:`pmrf.features.Goal` class for an easy way to define model costs.
        frequency : Frequency | None, optional
            The frequency grid to evaluate goals over.    
        """        
        if isinstance(cost, Sequence):
            cost_seq = cost
            def cost(*args, **kwargs):
                return jnp.sum(jnp.array([c(*args, **kwargs) for c in cost_seq]))

        x0, reconstruct_fn = make_reconstruct_fn(model)
        flat_cost_fn = make_flat_fn(cost, reconstruct_fn, frequency)
        
        dist = model.distribution()
        bounds = dist.bounds

        validate_bounds(x0, bounds[0], bounds[1], model.flat_param_names())

        self.x0 = x0
        self.bounds = bounds
        self.cost_fn = cost
        self.flat_cost_fn = flat_cost_fn
        self.reconstruct_fn = reconstruct_fn