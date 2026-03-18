import jax.numpy as jnp
import equinox as eqx

from pmrf.core import Model
from pmrf.core import Frequency
from pmrf.evaluators import Evaluator

class Problem(eqx.Module):
    model: Model
    frequency: Frequency
    evaluator: Evaluator
    
    def __call__(self) -> jnp.ndarray:
        return self.evaluator(self.model, self.frequency)