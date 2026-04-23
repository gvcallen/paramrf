from typing import Any

import jax.numpy as jnp

import parax as prx
from pmrf.core import Model, Frequency
from pmrf.explore._backend import AbstractAdaptiveSampler

class ExploreResult(prx.Module):
    """
    Container for the results of a parameter space exploration process.
    """
    model: Model                 # The model updated with empirical joint distributions
    frequency: Frequency | None  # The frequency sweep used
    
    # Raw Sample Data
    sampled_models: Model        # A batched Model containing all N sampled states
    sampled_features: jnp.ndarray # Array of shape (N, ...) extracted RF features
    
    solver_results: Any = None          # Results/trace from the underlying backend
    
    
def is_explorer(x):
    """
    Returns if a solver is suitable for design space exploration in :mod:`pmrf.explore`.

    Returns `True` for :class:`pmrf.explore.AbstractSampler`.
    """    
    return isinstance(x, AbstractAdaptiveSampler)