"""
Abstract base class for callables that evaluate a model over frequency.
"""

from abc import ABC, abstractmethod

from typing import Callable

import jax.numpy as jnp
import parax as prx

from pmrf.core.model import Model
from pmrf.core.frequency import Frequency

class Evaluator(prx.Module, prx.Operator, ABC):
    """
    Abstract base class for callables that evaluate a model over frequency.
    """
    @abstractmethod
    def __call__(self, model: Model, freq: Frequency, **kwargs) -> jnp.ndarray:
        """
        Evaluate the model response over the specified frequency range.

        Parameters
        ----------
        model : Model
            The model instance to evaluate.
        freq : Frequency
            The frequency object defining the evaluation points.
        **kwargs : dict
            Additional keyword arguments for the evaluation process.

        Returns
        -------
        jnp.ndarray
            The evaluated model response.
        """
        raise NotImplementedError
    
EvaluatorFn = Callable[[Model, Frequency], jnp.ndarray]
EvaluatorLike = str | list[str] | EvaluatorFn | list[EvaluatorFn]