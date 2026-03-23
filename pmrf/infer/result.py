from dataclasses import dataclass
from typing import Dict, Any

import jax.numpy as jnp

from pmrf.core import Model, Evaluator

@dataclass
class InferenceResult:
    model: Model # the model with its joint distribution updated
    
    model_samples: Model # the batched model with its samples
    likelihood_samples: Evaluator # the batched likelihood with its samples 
    history: Any = None # results from the underlying solver