import secrets
import jax
import jax.numpy as jnp

def generate_key() -> jnp.ndarray:
    # Generate a random 32-bit integer using system entropy
    random_seed = secrets.randbits(32)
    return jax.random.key(random_seed)

def lhs_sample(N: int, d: int, key=None) -> jnp.ndarray:
    key_perm, key_noise = jax.random.split(key)
    keys_perm = jax.random.split(key_perm, d)
    perms = jax.vmap(lambda k: jax.random.permutation(k, N))(keys_perm)
    noise = jax.random.uniform(key_noise, shape=(d, N))
    lhs_unit = (perms + noise) / N
    lhs_unit = lhs_unit.T
    return lhs_unit    