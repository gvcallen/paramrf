from typing import Callable

import jax.numpy as jnp
import parax as prx

class Metric(prx.Operator[[jnp.ndarray, jnp.ndarray], jnp.ndarray]):
    """
    Base class for callables that compare two arrays and return a metric.
    """
    def __call__(self, y_true: jnp.ndarray, y_pred: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
MetricLike = str | MetricFn | list[MetricFn]