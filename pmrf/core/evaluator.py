from typing import Callable

import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency

class Evaluator(prx.Module, prx.Operator):
    """
    Base class for callables that evaluate a model over frequency.
    """
    def __call__(self, model: Model, freq: Frequency, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
EvaluatorFn = Callable[[Model, Frequency], jnp.ndarray]
EvaluatorLike = str | list[str] | EvaluatorFn | list[EvaluatorFn]