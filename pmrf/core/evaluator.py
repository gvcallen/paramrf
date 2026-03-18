import jax.numpy as jnp
import equinox as eqx

from pmrf.core import Model, Frequency

class Evaluator(eqx.Module):
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        raise NotImplementedError