import equinox as eqx
import jax.numpy as jnp
from typing import Any

from pmrf.core import Model, Frequency

class ExploreResult(eqx.Module):
    """
    Container for the results of a parameter space exploration process.
    """
    model: Model                 # The model updated with empirical joint distributions
    frequency: Frequency | None  # The frequency sweep used
    
    # Raw Sample Data
    sampled_models: Model        # A batched Model containing all N sampled states
    sampled_features: jnp.ndarray # Array of shape (N, ...) extracted RF features
    
    history: Any = None          # Results/trace from the underlying sampler backend