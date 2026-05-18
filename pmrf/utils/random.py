import secrets
import jax
import jax.numpy as jnp

def generate_key() -> jnp.ndarray:
    random_seed = secrets.randbits(32)
    return jax.random.key(random_seed)