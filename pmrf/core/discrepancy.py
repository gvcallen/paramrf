import jax.numpy as jnp
import parax as prx

class Discrepancy(prx.Module):
    """
    Callable that calculate a discrepancy mean and variance given features x and model prediction y_pred.
    """
    def __call__(self, x: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
