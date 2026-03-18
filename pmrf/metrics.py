from typing import Callable
import jax.numpy as jnp

def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Mean Squared Error (MSE)."""
    loss = jnp.abs(y_true - y_pred)**2
    return weighted_mean(loss, sample_weight)

def root_mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Root Mean Squared Error (RMSE)."""
    return jnp.sqrt(mean_squared_error(y_true, y_pred, sample_weight))

def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Mean Absolute Error (MAE)."""
    loss = jnp.abs(y_true - y_pred)
    return weighted_mean(loss, sample_weight)

def mean_absolute_percentage_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    epsilon = 1e-12 
    loss = jnp.abs((y_true - y_pred) / jnp.maximum(jnp.abs(y_true), epsilon))
    return weighted_mean(loss, sample_weight)

def huber_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, delta: float = 1.0, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """Calculates the Huber loss, transitioning from L2 to L1 at the delta threshold."""
    diff = jnp.abs(y_true - y_pred)
    quadratic = jnp.minimum(diff, delta)
    linear = diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return weighted_mean(loss, sample_weight)

def weighted_residual(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Returns the unreduced, element-wise residuals.
    This is the required format for Least Squares solvers.
    """
    res = y_pred - y_true
    
    if sample_weight is not None:
        # We use the square root of the weight! 
        # Why? Because the Least Squares solver will square the entire residual later.
        # (sqrt(w) * res)^2 = w * res^2
        w = sample_weight[:, None] if res.ndim == 2 and sample_weight.ndim == 1 else sample_weight
        res = res * jnp.sqrt(w)
        
    return res

def metric_from_alias(alias: str) -> Callable:
    if alias == 'rms' or alias == 'root_mean_squared':
        return root_mean_squared_error
    elif alias == 'mae' or alias == 'mean_absolute_error':
        return mean_absolute_error
    else:
        raise Exception("Unknown metric alias in metric_from_alias")
    
def weighted_mean(loss: jnp.ndarray, sample_weight: jnp.ndarray | None) -> jnp.ndarray:
    """Safely computes the mean across the frequency axis (0), applying optional weights/masks."""
    if sample_weight is None:
        return jnp.mean(loss, axis=0)
    
    # Expand weight dimensions if loss has multiple features
    w = sample_weight[:, None] if loss.ndim == 2 and sample_weight.ndim == 1 else sample_weight
    
    weighted_sum = jnp.sum(loss * w, axis=0)
    weight_sum = jnp.maximum(jnp.sum(w, axis=0), 1e-12) # Prevent divide-by-zero
    return weighted_sum / weight_sum