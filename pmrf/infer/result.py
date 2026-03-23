from typing import Any

import parax as prx
import jax.numpy as jnp

from pmrf.core import Model, Evaluator

class InferResult(prx.Module):
    """
    Standardized return object for Bayesian inference routines.
    """
    model: Model               # The model updated with empirical posterior distributions
    likelihood: Evaluator      # The likelihood evaluator used
    
    # Raw Sample Data
    sampled_models: Model      # A batched Model containing all accepted sample states
    sampled_likelihoods: jnp.ndarray # The evaluated log-likelihoods for each sample
    
    history: Any = None        # Results/trace from the underlying nested sampler