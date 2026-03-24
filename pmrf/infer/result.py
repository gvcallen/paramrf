from typing import Any

import parax as prx
import jax.numpy as jnp

from pmrf.core import Model, Evaluator

class InferResult(prx.Module):
    """
    Standardized return object for inference routines.

    Attributes
    ----------
    model : Model
        The circuit model holding the finalized, optimized parameter state and posterior.
    likelihood : Evaluator
        The evaluator (e.g., :class:`pmrf.evaluators.Likelihood`) used to calculate the likelihood.
    sampled_models : Model
        The final batched model of sampled models.
    sampled_likelihoods : jnp.ndarray
        The evaluated log-likelihoods for each sample.
    history : Any
        The underlying solution object returned by the solver.
    """
    model: Model               # The model updated with empirical posterior distributions
    likelihood: Evaluator      # The likelihood evaluator used
    
    # Raw Sample Data
    sampled_models: Model      # A batched Model containing all accepted sample states
    sampled_likelihoods: jnp.ndarray # The evaluated log-likelihoods for each sample
    
    history: Any = None        # Results/trace from the underlying nested sampler