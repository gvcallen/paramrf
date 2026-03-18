from dataclasses import dataclass
from typing import Dict, Any

import jax.numpy as jnp

from pmrf.core import Model, Evaluator

@dataclass
class OptimizeResult:
    model: Model
    evaluator: Evaluator
    value: jnp.ndarray
    
    history: Dict[str, Any]
    success: bool