from typing import Callable

import jax.numpy as jnp
import parax as prx

class Loss(prx.Module, prx.Operator):
    """
    Base class for frequentist loss functions.
    
    A loss function accepts (y_true, y_pred) and returns a scalar loss.
    """
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

LossFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
LossLike = str | LossFn | list[LossFn]