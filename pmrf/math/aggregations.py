from typing import Callable
import jax
import jax.numpy as jnp


def weighted_sum(x: jnp.ndarray, weights: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Reduces the value over the sample (batch) dimension, applying weights if provided.
    
    Parameters
    ----------
    loss : jnp.ndarray
        The raw loss array of shape (n_samples, n_outputs).
    weights : jnp.ndarray, optional
        Optional array of weights for each sample, shape (n_samples,).
        
    Returns
    -------
    jnp.ndarray
        A JAX array of shape (n_outputs,) containing the sample-reduced loss.
    """
    if weights is not None:
        w = weights
        for _ in range(x.ndim - 1):
            w = w[..., None]
        x = x * w
        weight_sum = jnp.sum(w, axis=0)
    else:
        weight_sum = x.shape[0]
    
    return jnp.sum(x, axis=0) / jnp.maximum(weight_sum, 1e-12)


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
    return jnp.exp(jnp.mean(jnp.log(x.reshape(-1))))


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


def multioutput_aggregate(
    mean_loss: jnp.ndarray, 
    multioutput: str | jnp.ndarray | Callable = 'uniform_average'
) -> jnp.ndarray:
    """
    General aggregation of across multiple output dimensions given a multioutput string
    or callable.
    
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