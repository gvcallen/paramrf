import jax.numpy as jnp
import equinox as eqx

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.evaluators import Evaluator

class Problem(eqx.Module):
    model: Model
    frequency: Frequency
    evaluator: Evaluator
    
    def __call__(self) -> jnp.ndarray:
        return self.evaluator(self.model, self.frequency)