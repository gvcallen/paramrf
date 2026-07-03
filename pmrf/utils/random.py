from typing import Any, Tuple

import secrets
import jax
import jax.numpy as jnp

def generate_key() -> jnp.ndarray:
    random_seed = secrets.randbits(32)
    return jax.random.key(random_seed)

def compress_samples(
    samples: Any, 
    weights: jax.Array, 
    key: jax.Array
) -> Tuple[Any, jax.Array]:
    """
    Compresses a PyTree of samples to their approximate channel capacity (entropy) 
    using stochastic rounding, yielding equally weighted samples.
    
    Args:
        samples: A PyTree of JAX arrays where the leading axis is the sample dimension.
        weights: A 1D JAX array of weights corresponding to the samples.
        key: A JAX PRNGKey for resolving fractional probabilities.
        
    Returns:
        A tuple containing:
            - padded_samples: A PyTree mirroring `samples`, containing the compressed 
              samples. It is padded to the original size to remain JIT-compatible.
            - num_valid_samples: A JAX scalar integer indicating the exact number of 
              valid compressed samples.
    """
    # Normalize weights to probabilities
    p = weights / jnp.sum(weights)

    # Calculate optimal ncompress via Shannon Entropy
    # We add a tiny epsilon to prevent NaN errors from log(0)
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p + 1e-12), 0.0))
    ncompress = jnp.exp(entropy)

    # Scale weights to the calculated channel capacity
    w_scaled = p * ncompress

    # Split into integer guarantees and probabilistic fractions
    integer_part = jnp.floor(w_scaled).astype(jnp.int32)
    fractional_part = w_scaled - integer_part

    u = jax.random.uniform(key, shape=w_scaled.shape)
    extra = (u < fractional_part).astype(jnp.int32)
    
    # Final occurrence count for each original sample
    counts = integer_part + extra
    num_valid_samples = jnp.sum(counts)
    max_size = weights.shape[0]

    c_sum = jnp.cumsum(counts)
    grid = jnp.arange(max_size)
    indices = jnp.searchsorted(c_sum, grid, side='right')
    
    indices = jnp.clip(indices, 0, max_size - 1)
    padded_samples = jax.tree.map(lambda leaf: leaf[indices], samples)
    return padded_samples, num_valid_samples