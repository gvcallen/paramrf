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
    
    Measures the average of the squares of the errors.

    Attributes
    ----------
    multioutput : str | jnp.ndarray | Callable, default='uniform_average'
        Defines the aggregation strategy across multiple output dimensions.
    """
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
        return F.mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            sample_weight=sample_weight,
            multioutput=self.multioutput
        )


class RMSELoss(Loss):
    """
    Root Mean Squared Error (RMSE) metric.
    
    Attributes
    ----------
    multioutput : str | jnp.ndarray | Callable, default='uniform_average'
        Defines the aggregation strategy across multiple output dimensions.
    """
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
        return F.root_mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            sample_weight=sample_weight,
            multioutput=self.multioutput
        )


class LogMSELoss(Loss):
    """
    Log of Mean Squared Error (RMSE) metric.
    
    Attributes
    ----------
    multioutput : str | jnp.ndarray | Callable, default='uniform_average'
        Defines the aggregation strategy across multiple output dimensions.
    """
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
        return F.log_mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            sample_weight=sample_weight,
            multioutput=self.multioutput
        )


class L1Loss(Loss):
    """
    L1 Loss (Mean Absolute Error) metric.
    
    Measures the mean absolute value of the element-wise differences.

    Attributes
    ----------
    multioutput : str | jnp.ndarray | Callable, default='uniform_average'
        Defines the aggregation strategy across multiple output dimensions.
    """
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
        return F.mean_absolute_error(
            y_true=y_true, 
            y_pred=y_pred, 
            sample_weight=sample_weight,
            multioutput=self.multioutput
        )


class MAPELoss(Loss):
    """
    Mean Absolute Percentage Error (MAPE) metric.

    Attributes
    ----------
    multioutput : str | jnp.ndarray | Callable, default='uniform_average'
        Defines the aggregation strategy across multiple output dimensions.
    """
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
        return F.mean_absolute_percentage_error(
            y_true=y_true, 
            y_pred=y_pred, 
            sample_weight=sample_weight,
            multioutput=self.multioutput
        )


class HuberLoss(Loss):
    """
    Huber loss metric.
    
    A robust loss function that transitions from squared error to absolute error 
    depending on the delta threshold.

    Attributes
    ----------
    delta : float, default=1.0
        The threshold at which to change between squared error and absolute error.
    multioutput : str | jnp.ndarray | Callable, default='uniform_average'
        Defines the aggregation strategy across multiple output dimensions.
    """
    delta: float = prx.field(default=1.0, static=True)
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
        return F.huber_loss(
            y_true=y_true, 
            y_pred=y_pred, 
            delta=self.delta,
            sample_weight=sample_weight,
            multioutput=self.multioutput
        )
        
        
class HingeLoss(Loss):
    """
    Applies a one-sided constraint (hinge) before evaluating a base metric.
    
    Attributes
    ----------
    operator : str
        The logical constraint operator ('<', '>', '==', etc.).
    base_metric : Metric
        The underlying mathematical metric applied to the constraint residual.
    weight : float | jnp.ndarray
        A scalar or array multiplier to scale the importance of the penalty.
    mask : jnp.ndarray | None
        A boolean array filtering which data points apply to this loss.
    """
    operator: Literal['<', '<=', '>', '>=', '==', '='] = prx.field(default='==', static=True)
    weight: float | jnp.ndarray = 1.0
    mask: jnp.ndarray | None = prx.field(default=None, static=True)
    base_loss_fn: str | Callable | Loss = prx.field(default='rmse', static=True)
    multioutput: str | jnp.ndarray | Callable = prx.field(default='uniform_average', static=True)

    def __call__(
        self, 
        y_true: jnp.ndarray, 
        y_pred: jnp.ndarray, 
        sample_weight: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        return F.hinge_loss(
            y_true,
            y_pred,
            operator=self.operator,
            weight=self.weight,
            mask=self.mask,
            base_loss_fn=self.base_loss_fn,
            sample_weight=sample_weight,
            multioutput=self.multioutput,
        )


__all__ = [
    'LogMSELoss',
    'MSELoss',
    'RMSELoss',
    'L1Loss',
    'MAPELoss',
    'HuberLoss',
    'HingeLoss',
]