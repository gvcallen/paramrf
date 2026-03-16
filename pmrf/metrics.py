from typing import Callable
import jax.numpy as jnp

def _weighted_mean(loss: jnp.ndarray, sample_weight: jnp.ndarray | None) -> jnp.ndarray:
    """Safely computes the mean across the frequency axis (0), applying optional weights/masks."""
    if sample_weight is None:
        return jnp.mean(loss, axis=0)
    
    # Expand weight dimensions if loss has multiple features
    w = sample_weight[:, None] if loss.ndim == 2 and sample_weight.ndim == 1 else sample_weight
    
    weighted_sum = jnp.sum(loss * w, axis=0)
    weight_sum = jnp.maximum(jnp.sum(w, axis=0), 1e-12) # Prevent divide-by-zero
    return weighted_sum / weight_sum

def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Mean Squared Error (MSE)."""
    loss = jnp.abs(y_true - y_pred)**2
    return _weighted_mean(loss, sample_weight)

def root_mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Root Mean Squared Error (RMSE)."""
    return jnp.sqrt(mean_squared_error(y_true, y_pred, sample_weight))

def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Mean Absolute Error (MAE)."""
    loss = jnp.abs(y_true - y_pred)
    return _weighted_mean(loss, sample_weight)

def mean_absolute_percentage_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    epsilon = 1e-12 
    loss = jnp.abs((y_true - y_pred) / jnp.maximum(jnp.abs(y_true), epsilon))
    return _weighted_mean(loss, sample_weight)

def huber_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, delta: float = 1.0, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Huber loss, transitioning from L2 to L1 at the delta threshold."""
    diff = jnp.abs(y_true - y_pred)
    quadratic = jnp.minimum(diff, delta)
    linear = diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return _weighted_mean(loss, sample_weight)

def metric_from_alias(alias: str) -> Callable:
    if alias == 'rms' or alias == 'root_mean_squared':
        return root_mean_squared_error
    elif alias == 'mae' or alias == 'mean_absolute_error':
        return mean_absolute_error
    else:
        raise Exception("Unknown metric alias in metric_from_alias")