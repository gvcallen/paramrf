from typing import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import equinox as eqx
from jax import flatten_util

from pmrf.models import Model
from pmrf.frequency import Frequency

def make_reconstruct_fn(model: Model) -> tuple[jnp.ndarray, Callable[[jnp.ndarray], Model]]:
    """
    Extracts flat free parameters from a model and provides a pure function to reconstruct it.
    
    This is useful for optimizers and samplers that require 1D array inputs 
    (like SciPy, PolyChord, or emcee).
    """
    params_tree, static_tree = model.partition()
    flat_params, unravel_fn = flatten_util.ravel_pytree(params_tree)
    
    def reconstruct_fn(flat_x: jnp.ndarray) -> Model:
        unraveled_params = unravel_fn(flat_x)
        return eqx.combine(unraveled_params, static_tree)
        
    return flat_params, reconstruct_fn

def make_flat_fn(
    cost_fn: Callable[[Model, Frequency], float | jnp.ndarray],
    reconstruct_fn: Callable[[jnp.ndarray], Model],
    frequency: Frequency # Pass it into the bridge generator
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Converts a Model-based cost function into a JAX-jitted flat-array objective.
    """
    @eqx.filter_jit
    def flat_objective(flat_x: jnp.ndarray):
        model = reconstruct_fn(flat_x)
        # Pass the frequency through to the user's function
        return cost_fn(model, frequency) 
        
    return flat_objective