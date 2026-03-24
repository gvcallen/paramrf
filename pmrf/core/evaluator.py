"""
The a model and frequency evaluator.
"""
import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency

class Evaluator(prx.Module):
    """
    Represents any callable that extracts some frequency-dependent feature from a model.

    These are created automatically to define cost, likelihood and sampling functions,
    but can also be defined manually for more complex use-cases.
    Evaluators can also have parameters (such as for likelihood functions).

    See :mod:`pmrf.evaluators` for a list of built-in evaluators.
    """
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError