import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency
from pmrf.core.evaluator import Evaluator

class Problem(prx.Module):
    evaluator: Evaluator
    model: Model
    frequency: Frequency
    
    def __call__(self) -> jnp.ndarray:
        return self.evaluator(self.model, self.frequency)    