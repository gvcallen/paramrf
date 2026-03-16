import jax.numpy as jnp

def mean_squared_aggregate(x: jnp.ndarray) -> jnp.ndarray:
    """
    Aggregates x by calculating their Mean Square.
    """
    return jnp.mean(x**2)

def root_mean_squared_aggregate(x: jnp.ndarray) -> jnp.ndarray:
    """
    Aggregates x by calculating their Root Mean Square (quadratic mean).
    """
    return jnp.sqrt(jnp.mean(x**2))

def mean_absolute_aggregate(x: jnp.ndarray) -> jnp.ndarray:
    """
    Aggregates x by calculating their Mean Absolute value.
    (Functionally identical to a simple mean, as metrics are typically strictly positive).
    """
    return jnp.mean(jnp.abs(x))

def max_aggregate(x: jnp.ndarray) -> jnp.ndarray:
    """
    Aggregates x by taking the maximum value.
    Equivalent to a Chebyshev (L-infinity) norm. Excellent for Minimax optimization.
    """
    return jnp.max(x)

def sum_aggregate(x: jnp.ndarray) -> jnp.ndarray:
    """
    Aggregates x by calculating their absolute sum.
    """
    return jnp.sum(x)

def geometric_mean_aggregate(x: jnp.ndarray) -> jnp.ndarray:
    """
    Aggregates x using a geometric mean. 
    Floors values at epsilon to prevent a single zero-error feature from collapsing the cost.
    """
    num_features = x.shape[0]
    
    product = jnp.prod(x)
    return product ** (1.0 / num_features)

def convolutional_aggregate(x: jnp.ndarray) -> jnp.ndarray:
    """
    Aggregates x by iteratively convolving them, followed by an RMS reduction.
    Floors values at epsilon to prevent zero-collapse during the convolution product.
    """
    num_features = x.shape[0]
    
    # Slice as [0:1] to ensure it remains a 1D array for jnp.convolve
    convolved = x[0:1] 
    for i in range(1, num_features):
        convolved = jnp.convolve(convolved, x[i:i+1])
        
    convolved_scaled = convolved / (1.0 ** (num_features - 1))
    combined = jnp.sqrt(jnp.mean(convolved_scaled**2))
    
    return combined ** (1.0 / num_features)

def aggregate_from_alias(alias: str):
    if alias == 'rms' or alias == 'root_mean_squared':
        return root_mean_squared_aggregate
    elif alias == 'sum':
        return sum_aggregate
    elif alias == 'geometric_mean':
        return geometric_mean_aggregate
    elif alias == 'convolutional':
        return convolutional_aggregate