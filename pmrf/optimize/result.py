from dataclasses import dataclass
from typing import Dict, Any

import jax.numpy as jnp

from pmrf.core import Model, Evaluator


@dataclass
class OptimizeResult:
    """
    Standardized return object for parameter optimization routines.

    Attributes
    ----------
    model : Model
        The circuit model holding the finalized, optimized parameter state.
    cost : Evaluator
        The evaluator (e.g., metric, sum of goals) used to calculate the objective.
    value : jnp.ndarray
        The final scalar numerical loss value achieved by the optimizer.
    history : Dict[str, Any]
        Solver-specific execution stats (e.g., num_evals, num_steps, messages).
    success : bool
        True if the optimizer met its convergence criteria, False otherwise.
    """
    model: Model
    cost: Evaluator
    value: jnp.ndarray
    
    history: Dict[str, Any]
    success: bool