"""
Abstract base class for frequentist loss functions.
"""
from abc import abstractmethod

from typing import Callable

import jax.numpy as jnp
import equinox as eqx

class Loss(eqx.Module):
    """
    Abstract base class for frequentist loss functions.
    
    A loss function accepts (y_true, y_pred) and returns a loss value
    representing the discrepancy between the true data and the model prediction.
    """
    @abstractmethod
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Compute the loss between true data and model predictions.

        Parameters
        ----------
        y_true : jnp.ndarray
            The observed ground-truth data.
        y_pred : jnp.ndarray
            The model's predicted data.
        **kwargs : dict
            Additional keyword arguments for loss computation.

        Returns
        -------
        jnp.ndarray
            The calculated loss value.
        """        
        raise NotImplementedError

LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
LossLike = str | LossFn | list[LossFn]