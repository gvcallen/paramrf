import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency

class Evaluator(prx.Operator[[Model, Frequency], jnp.ndarray]):
    """
    Base class for callables that evaluate a model over frequency.
    """
    def __call__(self, model: Model, freq: Frequency, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

class Metric(prx.Operator[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
    """
    Base class for callables that compare two arrays and return a metric.
    """
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

class Problem(prx.Module):
    model: Model
    frequency: Frequency
    evaluator: Evaluator
    
    def __call__(self) -> jnp.ndarray:
        return self.evaluator(self.model, self.frequency)