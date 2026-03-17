from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.extractor import Extractor

class Objective(eqx.Module):
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError

class Error(Objective):
    extractor: Extractor
    target: jnp.ndarray
    error_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        prediction = self.extractor(model, freq)
        self.error_fn(self.target, prediction)

class Goal(Objective):
    extractor: Extractor
    condition: str = eqx.field(static=True) # '<' or '>'
    target: float
    
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        prediction = self.extractor(model, freq)
        
        if self.condition == '<':
            violation = jax.nn.relu(prediction - self.target)
        elif self.condition == '>':
            violation = jax.nn.relu(self.target - prediction)
        else:
            raise ValueError("Condition must be '<' or '>'")
            
        return jnp.mean(violation**2)