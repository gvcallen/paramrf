from typing import Callable
import jax
import jax.numpy as jnp

def geometric_mean(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the geometric mean over all elements of the input array.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        The geometric mean of the input array.
    """

    grouped = x.reshape(-1, 2)
    global_rms = jnp.sqrt(jnp.mean(grouped**2, axis=0))
    combined_rms = jnp.sqrt(jnp.prod(global_rms))
    return combined_rms

    # return jnp.exp(jnp.mean(jnp.log(x.reshape(-1))))


def convolution_aggregate(x: jnp.ndarray, epsilon: float = 1e-12) -> jnp.ndarray:
    """
    Generalized convolutional aggregation over flattened features.
    
    Parameters
    ----------
    x : jnp.ndarray
        Input array of features/losses.
    epsilon : float, default=1e-12
        Small constant to prevent numerical instability.
        
    Returns
    -------
    jnp.ndarray
        A scalar JAX array representing the aggregated value.
    """
    x = jnp.maximum(x, epsilon)
    x_flat = x.reshape(-1)
    n = x_flat.shape[0]

    # Use scan instead of Python loop (JAX-friendly)
    def step(carry, xi):
        return jnp.convolve(carry, jnp.array([xi])), None

    init = x_flat[0:1]
    convolved, _ = jax.lax.scan(step, init, x_flat[1:])

    # RMS reduction
    combined = jnp.sqrt(jnp.mean(convolved**2))
    
    return combined ** (1.0 / n)


def reduce_samples(loss: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Reduces the loss over the sample (batch) dimension, applying sample weights if provided.
    
    Parameters
    ----------
    loss : jnp.ndarray
        The raw loss array of shape (n_samples, n_outputs).
    sample_weight : jnp.ndarray, optional
        Optional array of weights for each sample, shape (n_samples,).
        
    Returns
    -------
    jnp.ndarray
        A JAX array of shape (n_outputs,) containing the sample-reduced loss.
    """
    if sample_weight is not None:
        w = sample_weight
        for _ in range(loss.ndim - 1):
            w = w[..., None]
        loss = loss * w
        weight_sum = jnp.sum(w, axis=0)
    else:
        weight_sum = loss.shape[0]
    
    return jnp.sum(loss, axis=0) / jnp.maximum(weight_sum, 1e-12)


def aggregate_multioutput(
    mean_loss: jnp.ndarray, 
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    Aggregates the sample-reduced loss across multiple output dimensions.
    
    Parameters
    ----------
    mean_loss : jnp.ndarray
        The loss array reduced over the batch dimension, shape (n_outputs,).
    multioutput : str, jnp.ndarray, or Callable, default='uniform_average'
        String alias ('raw_values', 'uniform_average', 'geometric_mean', 
        'convolution'), an array of custom weights for each output, or a custom 
        callable function.
                     
    Returns
    -------
    jnp.ndarray
        The fully aggregated loss as a scalar (or array if 'raw_values' is selected).
    """
    if callable(multioutput):
        return multioutput(mean_loss)
    elif isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return mean_loss
        elif multioutput == 'uniform_average':
            return jnp.mean(mean_loss)
        elif multioutput == 'geometric_mean':
            return geometric_mean(mean_loss)
        elif multioutput == 'convolution':
            return convolution_aggregate(mean_loss)
        else:
            raise ValueError(f"Unknown multioutput value: {multioutput}")
    else:
        weights = jnp.asarray(multioutput)
        if weights.shape != mean_loss.shape:
            raise ValueError(
                f"multioutput weights shape {weights.shape} does not match output shape {mean_loss.shape}"
            )
        return jnp.sum(mean_loss * weights) / jnp.sum(weights)


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

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    loss = (jnp.abs(y_true - y_pred))**2
    mean_loss = reduce_samples(loss, sample_weight)
    return aggregate_multioutput(mean_loss, multioutput)


def root_mean_squared_error(
    y_true: jnp.ndarray, 
    y_pred: jnp.ndarray, 
    sample_weight: jnp.ndarray | None = None,
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.
    Applies the square root per-output before aggregating across multiple outputs.

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

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    loss = (jnp.abs(y_true - y_pred))**2
    mean_loss = reduce_samples(loss, sample_weight)
    rmse_loss = jnp.sqrt(mean_loss)
    return aggregate_multioutput(rmse_loss, multioutput)


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

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    loss = jnp.abs(y_true - y_pred)
    mean_loss = reduce_samples(loss, sample_weight)
    return aggregate_multioutput(mean_loss, multioutput)


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

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    epsilon = 1e-12
    loss = jnp.abs((y_true - y_pred) / jnp.maximum(jnp.abs(y_true), epsilon))
    mean_loss = reduce_samples(loss, sample_weight)
    return aggregate_multioutput(mean_loss, multioutput)


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

    Returns
    -------
    jnp.ndarray
        The calculated aggregated loss.
    """
    diff = jnp.abs(y_true - y_pred)
    quadratic = jnp.minimum(diff, delta)
    linear = diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    mean_loss = reduce_samples(loss, sample_weight)
    return aggregate_multioutput(mean_loss, multioutput)


def loss_from_alias(alias: str | Callable) -> Callable:
    """
    Resolves a loss function from a string alias or passes a callable through.
    
    Parameters
    ----------
    alias : str or Callable
        A string identifier for a standard loss (e.g., 'mse', 'rmse') 
        or a custom callable function.
            
    Returns
    -------
    Callable
        A callable loss function that takes (y_true, y_pred).
        
    Raises
    ------
    ValueError
        If the alias is a string but not recognized.
    """
    # 1. The Power-User Path: Pass custom JAX math directly
    if callable(alias):
        return alias
        
    # 2. The Standard DX Path: Clean aliases for official losses
    standard_losses = {
        'rms': root_mean_squared_error,
        'rmse': root_mean_squared_error,
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'mape': mean_absolute_percentage_error,
        'huber': huber_loss,
    }
    
    # Normalize the string
    clean_alias = str(alias).strip().lower()
    
    if clean_alias in standard_losses:
        return standard_losses[clean_alias]

    raise ValueError(
        f"Unknown loss alias: '{alias}'. "
        f"Supported aliases: {list(standard_losses.keys())}. "
        f"For custom losses, pass a callable taking (y_true, y_pred)."
    )

__all__ = [
    'geometric_mean',
    'convolution_aggregate',
    'reduce_samples',
    'aggregate_multioutput',
    'root_mean_squared_error',
    'mean_squared_error',
    'mean_absolute_error',
    'mean_absolute_percentage_error',
    'huber_loss',
]