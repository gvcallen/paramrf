from typing import Callable, Union
from functools import partial
import jax
import jax.numpy as jnp

def geometric_mean(x: jnp.ndarray, epsilon: float = 1e-12) -> jnp.ndarray:
    """
    Geometric mean over all elements of x.
    (Note: Currently implemented with custom pair-wise logic for debugging).
    """
    feature_rms = x
    
    grouped = feature_rms.reshape(-1, 2)
    global_rms = jnp.sqrt(jnp.mean(grouped**2, axis=0))
    product = jnp.prod(global_rms)
    combined_rms = product ** (1.0 / 2)
    
    return combined_rms

    # x = jnp.maximum(x, epsilon)  # prevent collapse
    # x_flat = x.reshape(-1)
    
    # return jnp.exp(jnp.mean(jnp.log(x_flat)))

def convolution_aggregate(x: jnp.ndarray, epsilon: float = 1e-12) -> jnp.ndarray:
    """
    Generalized convolutional aggregation over flattened features.
    
    Args:
        x: Input array of features/losses.
        epsilon: Small constant to prevent numerical instability.
        
    Returns:
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
    
    Args:
        loss: The raw loss array of shape (n_samples, n_outputs).
        sample_weight: Optional array of weights for each sample, shape (n_samples,).
        
    Returns:
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

def aggregate_multioutput(mean_loss: jnp.ndarray, multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    """
    Aggregates the sample-reduced loss across multiple output dimensions.
    
    Args:
        mean_loss: The loss array reduced over the batch dimension, shape (n_outputs,).
        multioutput: String alias ('raw_values', 'uniform_average', 'geometric_mean', 
                     'convolution') or an array of custom weights for each output.
                     
    Returns:
        The fully aggregated loss as a scalar (or array if 'raw_values' is selected).
    """
    if isinstance(multioutput, str):
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

def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None,
                       multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    """
    Computes the Mean Squared Error (MSE) between true and predicted values.
    """
    loss = (jnp.abs(y_true - y_pred))**2
    mean_loss = reduce_samples(loss, sample_weight)
    return aggregate_multioutput(mean_loss, multioutput)

def root_mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None,
                            multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.
    Applies the square root per-output before aggregating across multiple outputs.
    """
    loss = (jnp.abs(y_true - y_pred))**2
    mean_loss = reduce_samples(loss, sample_weight)
    rmse_loss = jnp.sqrt(mean_loss)
    return aggregate_multioutput(rmse_loss, multioutput)

def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None,
                        multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    """
    Computes the Mean Absolute Error (MAE) between true and predicted values.
    """
    loss = jnp.abs(y_true - y_pred)
    mean_loss = reduce_samples(loss, sample_weight)
    return aggregate_multioutput(mean_loss, multioutput)

def mean_absolute_percentage_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None,
                                   multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    """
    Computes the Mean Absolute Percentage Error (MAPE) between true and predicted values.
    """
    epsilon = 1e-12
    loss = jnp.abs((y_true - y_pred) / jnp.maximum(jnp.abs(y_true), epsilon))
    mean_loss = reduce_samples(loss, sample_weight)
    return aggregate_multioutput(mean_loss, multioutput)

def huber_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, delta: float = 1.0, sample_weight: jnp.ndarray | None = None,
               multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    """
    Computes the Huber loss, a robust loss function that transitions from squared error 
    to absolute error depending on the delta threshold.
    """
    diff = jnp.abs(y_true - y_pred)
    quadratic = jnp.minimum(diff, delta)
    linear = diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    mean_loss = reduce_samples(loss, sample_weight)
    return aggregate_multioutput(mean_loss, multioutput)

def metric_from_alias(alias: str | Callable, multioutput: str = 'uniform_average') -> Callable:
    """
    Resolves a metric function from a string alias or passes a callable through.
    
    Args:
        alias: A string identifier for a standard metric (e.g., 'mse', 'rmse') 
               or a custom callable function.
        multioutput: Defines how multiple outputs are aggregated.
        
    Returns:
        A callable loss function that takes (y_true, y_pred).
    """
    # 1. The Power-User Path: Pass custom JAX math directly
    if callable(alias):
        return alias
        
    # 2. The Standard DX Path: Clean aliases for official metrics
    standard_metrics = {
        'rms': root_mean_squared_error,
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'mape': mean_absolute_percentage_error,
        'huber': huber_loss,
    }
    
    # Normalize the string
    clean_alias = str(alias).strip().lower()
    
    if clean_alias in standard_metrics:
        return partial(standard_metrics[clean_alias], multioutput=multioutput)

    raise ValueError(
        f"Unknown metric alias: '{alias}'. "
        f"Supported aliases: {list(standard_metrics.keys())}. "
        f"For custom metrics, pass a callable taking (y_true, y_pred)."
    )