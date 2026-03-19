from typing import Callable, Union
import jax.numpy as jnp

from sklearn.metrics import root_mean_squared_error

def weighted_mean(loss: jnp.ndarray, 
                  sample_weight: jnp.ndarray | None = None, 
                  multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    # Step 1: apply sample weights
    if sample_weight is not None:
        w = sample_weight
        # Expand weights to match remaining dimensions
        for _ in range(loss.ndim - 1):
            w = w[..., None]
        loss = loss * w
        weight_sum = jnp.sum(w, axis=0)
    else:
        weight_sum = loss.shape[0]
    
    # Step 2: mean over samples (axis 0)
    mean_loss = jnp.sum(loss, axis=0) / jnp.maximum(weight_sum, 1e-12)
    
    # Step 3: multioutput aggregation
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return mean_loss  # keep remaining axes
        elif multioutput == 'uniform_average':
            return jnp.mean(mean_loss)  # scalar across all output dims
        else:
            raise ValueError(f"Unknown multioutput value: {multioutput}")
    else:
        # weighted aggregation across all remaining axes
        weights = jnp.asarray(multioutput)
        if weights.shape != mean_loss.shape:
            raise ValueError(f"multioutput weights shape {weights.shape} does not match output shape {mean_loss.shape}")
        return jnp.sum(mean_loss * weights) / jnp.sum(weights)

def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None,
                       multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    loss = (jnp.abs(y_true - y_pred))**2
    return weighted_mean(loss, sample_weight, multioutput)

def root_mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None,
                            multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    return jnp.sqrt(mean_squared_error(y_true, y_pred, sample_weight, multioutput))

def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None,
                        multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    loss = jnp.abs(y_true - y_pred)
    return weighted_mean(loss, sample_weight, multioutput)

def mean_absolute_percentage_error(y_true: jnp.ndarray, y_pred: jnp.ndarray, sample_weight: jnp.ndarray | None = None,
                                   multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    epsilon = 1e-12
    loss = jnp.abs((y_true - y_pred) / jnp.maximum(jnp.abs(y_true), epsilon))
    return weighted_mean(loss, sample_weight, multioutput)

def huber_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, delta: float = 1.0, sample_weight: jnp.ndarray | None = None,
               multioutput: Union[str, jnp.ndarray] = 'uniform_average') -> jnp.ndarray:
    diff = jnp.abs(y_true - y_pred)
    quadratic = jnp.minimum(diff, delta)
    linear = diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return weighted_mean(loss, sample_weight, multioutput)

def metric_from_alias(alias: str | Callable) -> Callable:
    """
    Resolves a metric function from a string alias or passes a callable through.
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
        return standard_metrics[clean_alias]

    raise ValueError(
        f"Unknown metric alias: '{alias}'. "
        f"Supported aliases: {list(standard_metrics.keys())}. "
        f"For custom metrics, pass a callable taking (y_true, y_pred)."
    )