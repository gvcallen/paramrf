"""
Loss models for frequentist optimization or generalized Bayesian inference.

These classes wrap pure mathematical loss functions into a :class:`pmrf.Loss`.
All losses take the true and predict arrays as inputs, and return the loss
value when called.
"""
from abc import abstractmethod
from typing import Callable, Literal

import jax.numpy as jnp
import equinox as eqx
import parax as prx

from pmrf.math import losses as F
from pmrf.jax_utils import field, unwrap


class AbstractLoss(eqx.Module):
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

class MSELoss(AbstractLoss):
    """
    Mean Squared Error (MSE) metric.

    Forwards to :func:`pmrf.math.losses.mean_squared_error`.
    """

    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)

        return F.mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )


class RMSELoss(AbstractLoss):
    """
    Root Mean Squared Error (RMSE) metric.

    Forwards to :func:`pmrf.math.losses.root_mean_squared_error`.
    """

    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)

        return F.root_mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )


class LogMSELoss(AbstractLoss):
    """
    Log of Mean Squared Error (RMSE) metric.

    Forwards to :func:`pmrf.math.losses.log_mean_squared_error`.
    """
    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)

        return F.log_mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )


class MAPELoss(AbstractLoss):
    """
    Mean Absolute Percentage Error (MAPE) metric.

    Forwards to :func:`pmrf.math.losses.mean_absolute_percentage_error`.
    """
    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)
        
        return F.mean_absolute_percentage_error(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )


class HuberLoss(AbstractLoss):
    """
    Huber loss metric.
    
    A robust loss function that transitions from squared error to absolute error 
    depending on the delta threshold.

    Forwards to :func:`pmrf.math.losses.huber_loss`.
    """
    #: The threshold at which to change between squared error and absolute error.
    delta: float = field(default=1.0, static=True)
    
    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)
        kwargs.setdefault('delta', self.delta)

        return F.huber_loss(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )
        
        
class HingeLoss(AbstractLoss):
    """
    Applies a one-sided constraint (hinge) before evaluating a base metric.

    Forwards to :func:`pmrf.math.losses.hinge_loss`.
    """
    #: The logical constraint operator ('<', '>', '==', etc.).
    operator: Literal['<', '<=', '>', '>=', '==', '='] = field(default='==', static=True)

    #: A scalar or array multiplier to scale the importance of the penalty.
    weight: float = field(default=1.0, static=True)
    
    #: A boolean array filtering which data points apply to this loss.
    mask: jnp.ndarray | None = field(default=None, converter=prx.as_frozen)
    
    #: The underlying loss function.
    base_loss: str | Callable | AbstractLoss = field(default=RMSELoss())
    
    #: Defines the aggregation strategy across multiple output dimensions.    
    multioutput: str | jnp.ndarray | Callable = field(default='uniform_average', static=True)

    def __call__(
        self, 
        y_true: jnp.ndarray, 
        y_pred: jnp.ndarray, 
        **kwargs,
    ) -> jnp.ndarray:
        self = unwrap(self)
        
        kwargs.setdefault('multioutput', self.multioutput)

        return F.hinge_loss(
            y_true,
            y_pred,
            operator=self.operator,
            weight=self.weight,
            mask=self.mask,
            base_loss=self.base_loss,
            **kwargs,
        )


__all__ = [
    'LogMSELoss',
    'RMSELoss',
    'MAPELoss',
    'HuberLoss',
    'HingeLoss',
]