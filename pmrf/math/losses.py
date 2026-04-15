"""
Common loss functions for optimization.
"""

from typing import Callable, Literal
import jax
import jax.numpy as jnp

from pmrf.math.aggregations import aggregate, weighted_sum

def mean_squared_error(
    y_true: jnp.ndarray, 
    y_pred: jnp.ndarray, 
    sample_weight: jnp.ndarray | None = None,
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    Computes the Mean Squared Error (MSE) between true and predicted values.

    Parameters
    ----------
    y_true : jnp.ndarray
        Ground truth (correct) target values.
    y_pred : jnp.ndarray
        Estimated target values.
    sample_weight : jnp.ndarray, optional
        Optional array of weights for each sample.
    multioutput : str, jnp.ndarray, or Callable, default='uniform_average'
        Defines aggregating of multiple output values.
        Can be 'uniform_average', 'raw_values', 'geometric_mean', 'convolution', or a callable.

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    loss = (jnp.abs(y_true - y_pred))**2
    mean_loss = weighted_sum(loss, sample_weight)
    return aggregate(mean_loss, multioutput)


def log_mean_squared_error(
    y_true: jnp.ndarray, 
    y_pred: jnp.ndarray, 
    sample_weight: jnp.ndarray | None = None,
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    Computes the log of the Mean Squared Error (MSE) between true and predicted values.

    Parameters
    ----------
    y_true : jnp.ndarray
        Ground truth (correct) target values.
    y_pred : jnp.ndarray
        Estimated target values.
    sample_weight : jnp.ndarray, optional
        Optional array of weights for each sample.
    multioutput : str, jnp.ndarray, or Callable, default='uniform_average'
        Defines aggregating of multiple output values.
        Can be 'uniform_average', 'raw_values', 'geometric_mean', 'convolution', or a callable.

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    # jnp.abs is redundant before squaring
    loss = (y_true - y_pred)**2 
    
    # Assuming weighted_sum is defined elsewhere in your codebase
    mean_loss = weighted_sum(loss, sample_weight) 
    
    # Dynamically fetch the smallest safe float for the current dtype
    epsilon = jnp.finfo(mean_loss.dtype).tiny
    safe_mean_loss = jnp.maximum(mean_loss, epsilon)
    
    # Calculate the log safely
    log_mse_loss = jnp.log(safe_mean_loss)
    
    # Assuming aggregate is defined elsewhere in your codebase
    return aggregate(log_mse_loss, multioutput)


def root_mean_squared_error(
    y_true: jnp.ndarray, 
    y_pred: jnp.ndarray, 
    sample_weight: jnp.ndarray | None = None,
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : jnp.ndarray
        Ground truth (correct) target values.
    y_pred : jnp.ndarray
        Estimated target values.
    sample_weight : jnp.ndarray, optional
        Optional array of weights for each sample.
    multioutput : str, jnp.ndarray, or Callable, default='uniform_average'
        Defines aggregating of multiple output values.
        Can be 'uniform_average', 'raw_values', 'geometric_mean', 'convolution', or a callable.

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    loss = (jnp.abs(y_true - y_pred))**2
    mean_loss = weighted_sum(loss, sample_weight)
    rmse_loss = jnp.sqrt(mean_loss)
    return aggregate(rmse_loss, multioutput)


def mean_absolute_error(
    y_true: jnp.ndarray, 
    y_pred: jnp.ndarray, 
    sample_weight: jnp.ndarray | None = None,
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    Computes the Mean Absolute Error (MAE) between true and predicted values.

    Parameters
    ----------
    y_true : jnp.ndarray
        Ground truth (correct) target values.
    y_pred : jnp.ndarray
        Estimated target values.
    sample_weight : jnp.ndarray, optional
        Optional array of weights for each sample.
    multioutput : str, jnp.ndarray, or Callable, default='uniform_average'
        Defines aggregating of multiple output values.
        Can be 'uniform_average', 'raw_values', 'geometric_mean', 'convolution', or a callable.

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    loss = jnp.abs(y_true - y_pred)
    mean_loss = weighted_sum(loss, sample_weight)
    return aggregate(mean_loss, multioutput)


def mean_absolute_percentage_error(
    y_true: jnp.ndarray, 
    y_pred: jnp.ndarray, 
    sample_weight: jnp.ndarray | None = None,
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    Computes the Mean Absolute Percentage Error (MAPE) between true and predicted values.

    Parameters
    ----------
    y_true : jnp.ndarray
        Ground truth (correct) target values.
    y_pred : jnp.ndarray
        Estimated target values.
    sample_weight : jnp.ndarray, optional
        Optional array of weights for each sample.
    multioutput : str, jnp.ndarray, or Callable, default='uniform_average'
        Defines aggregating of multiple output values.
        Can be 'uniform_average', 'raw_values', 'geometric_mean', 'convolution', or a callable.

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    epsilon = 1e-12
    loss = jnp.abs((y_true - y_pred) / jnp.maximum(jnp.abs(y_true), epsilon))
    mean_loss = weighted_sum(loss, sample_weight)
    return aggregate(mean_loss, multioutput)


def huber_loss(
    y_true: jnp.ndarray, 
    y_pred: jnp.ndarray, 
    delta: float = 1.0, 
    sample_weight: jnp.ndarray | None = None,
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    Computes the Huber loss, a robust loss function that transitions from squared error 
    to absolute error depending on the delta threshold.

    Parameters
    ----------
    y_true : jnp.ndarray
        Ground truth (correct) target values.
    y_pred : jnp.ndarray
        Estimated target values.
    delta : float, default=1.0
        The threshold at which to change between squared error and absolute error.
    sample_weight : jnp.ndarray, optional
        Optional array of weights for each sample.
    multioutput : str, jnp.ndarray, or Callable, default='uniform_average'
        Defines aggregating of multiple output values.
        Can be 'uniform_average', 'raw_values', 'geometric_mean', 'convolution', or a callable.

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    diff = jnp.abs(y_true - y_pred)
    quadratic = jnp.minimum(diff, delta)
    linear = diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    mean_loss = weighted_sum(loss, sample_weight)
    return aggregate(mean_loss, multioutput)


def hinge_loss(
    y_true: jnp.ndarray, 
    y_pred: jnp.ndarray, 
    operator: Literal['<', '<=', '>', '>=', '==', '='],
    weight: float | jnp.ndarray = 1.0,
    mask: jnp.ndarray | None = None,
    base_loss_fn: str | Callable = 'rmse', 
    multioutput: str | jnp.ndarray | Callable = 'uniform_average',
    **kwargs,
) -> jnp.ndarray:
    """
    Applies a differentiable one-sided constraint (hinge) before evaluating a base metric.

    Utilizes a differentiable clamping technique to ensure the optimizer 
    only experiences a penalty gradient when the constraint is violated.

    Parameters
    ----------
    y_true : jnp.ndarray
        The target threshold or ground truth values.
    y_pred : jnp.ndarray
        The estimated circuit feature values.
    operator : Literal['<', '<=', '>', '>=', '==', '=']
        The logical constraint operator defining the goal boundary.
    weight : float | jnp.ndarray, default=1.0
        A scalar or array multiplier to scale the importance of the penalty.
    mask : jnp.ndarray | None, default=None
        A boolean array filtering which data points apply to this loss.
    base_loss_fn : str | Callable, default='rmse'
        The underlying mathematical metric applied to the constraint residual.
        Can be a string alias (resolved via LOSS_LOOKUP) or a custom callable.
    **kwargs
        Key-word arguments to forward to the underlying loss function.

    Returns
    -------
    jnp.ndarray
        The calculated scalar or array penalty.

    Raises
    ------
    ValueError
        If an unknown constraint operator is provided.
    """
    # 1. Apply the differentiable hinge
    if operator in ['<', '<=']:
        effective_pred = jnp.maximum(y_pred, y_true)
    elif operator in ['>', '>=']:
        effective_pred = jnp.minimum(y_pred, y_true)
    elif operator in ['==', '=']:
        effective_pred = y_pred
    else:
        raise ValueError(f"Unknown Hinge operator: '{operator}'")
        
    # 2. Weighting & Masking in Residual Space
    residual = effective_pred - y_true
    weighted_residual = residual * weight
    
    if mask is not None:
        weighted_residual = jnp.where(mask, weighted_residual, 0.0)
        
    # 3. Shift back to target space
    final_pred = y_true + weighted_residual
    
    # Resolve the base metric if a string alias is provided
    if isinstance(base_loss_fn, str):
        base_loss_fn = LOSS_LOOKUP[base_loss_fn][1]
    
    # 4. Defer actual distance calculation to the base metric
    return base_loss_fn(
        y_true,
        final_pred,
        **kwargs,
    )
    
LOSS_LOOKUP: dict[str, tuple[str, Callable | None]] = {
    'log_rmse': ('Log root mean squared error', log_mean_squared_error),
    'rmse': ('Root mean squared error', root_mean_squared_error),
    'mse': ('Mean squared error', mean_squared_error),
    'mae': ('Mean absolute error', mean_absolute_error),
    'mape': ('Mean absolute percentage error', mean_absolute_percentage_error),
    'huber': ('Huber loss', huber_loss),
}