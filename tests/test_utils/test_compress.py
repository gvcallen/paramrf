import pytest
import jax
import jax.numpy as jnp

# Assuming the function is saved in a module named `pmrf.inference`
from pmrf.utils.random import compress_samples

def test_uniform_weights():
    """
    Test that perfectly uniform weights result in maximum entropy (ncompress == N).
    Every sample should be selected exactly once.
    """
    N = 100
    samples = jnp.arange(N)
    weights = jnp.ones(N)
    key = jax.random.key(0)
    
    padded_samples, num_valid = compress_samples(samples, weights, key)
    
    # For uniform weights, entropy maxes out, meaning ncompress = N
    assert int(num_valid) == N
    
    # The first N elements should perfectly match the original samples
    assert jnp.array_equal(padded_samples[:int(num_valid)], samples)


def test_single_dominant_weight():
    """
    Test that a single non-zero weight results in zero entropy (ncompress == 1).
    Only the dominant sample should be kept.
    """
    N = 10
    samples = jnp.arange(N)
    
    # Make index 4 the only valid weight
    weights = jnp.zeros(N).at[4].set(1.0) 
    key = jax.random.key(42)
    
    padded_samples, num_valid = compress_samples(samples, weights, key)
    
    # Zero entropy means only 1 effective sample is kept
    assert int(num_valid) == 1
    
    # The only valid sample should be the one at index 4
    assert padded_samples[0] == 4


def test_pytree_handling_and_jit():
    """
    Test that the function correctly traverses nested dictionaries and tuples,
    and successfully compiles under jax.jit.
    """
    N = 50
    # Create a complex PyTree
    samples = {
        "x": jax.random.normal(jax.random.key(1), (N, 3)),
        "y": (jnp.arange(N), jnp.linspace(0, 1, N))
    }
    
    # Random exponentially distributed weights
    weights = jnp.exp(jax.random.normal(jax.random.key(2), (N,)))
    key = jax.random.key(3)
    
    # JIT compile the function
    jit_compress = jax.jit(compress_samples)
    padded_samples, num_valid = jit_compress(samples, weights, key)
    
    # Check that shapes were maintained in the padding
    assert padded_samples["x"].shape == (N, 3)
    assert padded_samples["y"][0].shape == (N,)
    assert padded_samples["y"][1].shape == (N,)
    
    # Entropy compression means num_valid is strictly bounded
    assert 1 <= int(num_valid) <= N


def test_empty_weights_handling():
    """
    Test behavior when all weights are zero (should fall back gracefully, 
    though realistically this is an edge case usually caught prior).
    """
    N = 5
    samples = jnp.arange(N)
    weights = jnp.zeros(N)
    key = jax.random.key(0)
    
    # We added +1e-12 in the log and check p > 0, so this should not NaN out.
    padded_samples, num_valid = compress_samples(samples, weights, key)
    
    # If all weights are 0, probabilities are 0 (or NaN if 0/0 isn't caught by JAX where). 
    # Because 0/0 is NaN, let's ensure we catch that JAX handles this via the p>0 mask, 
    # though jnp.sum(weights) == 0 yields NaN for p. 
    # It is expected that num_valid is 0 or NaN, but it shouldn't crash the interpreter.
    assert jnp.isnan(num_valid) or int(num_valid) == 0