import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency
from pmrf.core.evaluator import Evaluator

class Problem(prx.Module):
    """
    A class representing a callable problem to solve.
    
    For example, this class can be used to encapsulate a model and a loss or likelihood function
    """
    
    model: Model
    frequency: Frequency
    evaluator: Evaluator
    
    def __call__(self) -> jnp.ndarray:
        return self.evaluator(self.model, self.frequency)