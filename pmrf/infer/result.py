from dataclasses import dataclass
from typing import Dict, Any

import jax.numpy as jnp

from pmrf.core import Model, Evaluator

@dataclass
class InferResult:
    model: Model
    likelihood: Evaluator
    value: jnp.ndarray | None = None
    
    history: Dict[str, Any]
    success: bool