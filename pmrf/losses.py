"""
Loss models for frequentist optimization or generalized Bayesian inference.

These classes wrap pure mathematical loss functions into a :class:`pmrf.Loss`.
All losses take the true and predict arrays as inputs, and return the loss
value when called.
"""

from typing import Callable, Literal
import jax.numpy as jnp
import parax as prx

from pmrf.math import losses as F
from pmrf.core import Loss

class MSELoss(Loss):
    """
    Mean Squared Error (MSE) metric.

    Forwards to :func:`pmrf.math.losses.mean_squared_error`.
    """

    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)

        return F.mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )


class RMSELoss(Loss):
    """
    Root Mean Squared Error (RMSE) metric.

    Forwards to :func:`pmrf.math.losses.root_mean_squared_error`.
    """

    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)

        return F.root_mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )


class LogMSELoss(Loss):
    """
    Log of Mean Squared Error (RMSE) metric.

    Forwards to :func:`pmrf.math.losses.log_mean_squared_error`.
    """
    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)

        return F.log_mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )


class MAPELoss(Loss):
    """
    Mean Absolute Percentage Error (MAPE) metric.

    Forwards to :func:`pmrf.math.losses.mean_absolute_percentage_error`.
    """
    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)
        
        return F.mean_absolute_percentage_error(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )


class HuberLoss(Loss):
    """
    Huber loss metric.
    
    A robust loss function that transitions from squared error to absolute error 
    depending on the delta threshold.

    Forwards to :func:`pmrf.math.losses.huber_loss`.
    """
    #: The threshold at which to change between squared error and absolute error.
    delta: float = prx.field(default=1.0, static=True)
    
    #: Defines the aggregation strategy across multiple output dimensions.
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)
        kwargs.setdefault('delta', self.delta)

        return F.huber_loss(
            y_true=y_true, 
            y_pred=y_pred, 
            **kwargs,
        )
        
        
class HingeLoss(Loss):
    """
    Applies a one-sided constraint (hinge) before evaluating a base metric.

    Forwards to :func:`pmrf.math.losses.hinge_loss`.
    """
    #: The logical constraint operator ('<', '>', '==', etc.).
    operator: Literal['<', '<=', '>', '>=', '==', '='] = prx.field(default='==', static=True)

    #: A scalar or array multiplier to scale the importance of the penalty.
    weight: float | jnp.ndarray = 1.0
    
    #: A boolean array filtering which data points apply to this loss.
    mask: jnp.ndarray | None = prx.field(default=None)
    
    #: The underlying loss function.
    base_loss_fn: str | Callable | Loss = prx.field(default=RMSELoss())
    
    #: Defines the aggregation strategy across multiple output dimensions.    
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(
        self, 
        y_true: jnp.ndarray, 
        y_pred: jnp.ndarray, 
        **kwargs,
    ) -> jnp.ndarray:
        kwargs.setdefault('multioutput', self.multioutput)

        return F.hinge_loss(
            y_true,
            y_pred,
            operator=self.operator,
            weight=self.weight,
            mask=self.mask,
            base_loss_fn=self.base_loss_fn,
            **kwargs,
        )


__all__ = [
    'LogMSELoss',
    'RMSELoss',
    'MAPELoss',
    'HuberLoss',
    'HingeLoss',
]