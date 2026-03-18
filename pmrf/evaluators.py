from typing import Callable, Any
import operator

import jax
import jax.numpy as jnp
import equinox as eqx
from numpyro.distributions import Distribution

from pmrf.core import Model, Frequency, Parameter, Evaluator

class CustomEvaluator(Evaluator):
    fn: Callable[[Model, Frequency], jnp.ndarray] = eqx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(model, freq)

class Method(Evaluator):
    path: str = eqx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        func = operator.attrgetter(self.path)(model)
        return func(freq)

class Index(Evaluator):
    evaluator: Evaluator
    indices: Any = eqx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return data[self.indices]

class Mask(Evaluator):
    evaluator: Evaluator
    mask: jnp.ndarray = eqx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return jax.vmap(lambda m: m[self.mask])(data)

class Mapped(Evaluator):
    evaluator: Evaluator
    fn: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(self.evaluator(model, freq))
        
class Stacked(Evaluator):
    evaluators: list[Evaluator]
    axis: int = eqx.field(default=-1, static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        results = [ev(model, freq) for ev in self.evaluators]
        return jnp.stack(results, axis=self.axis)

class Residual(Evaluator):
    predictor: Evaluator
    target: jnp.ndarray

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.predictor(model, freq) - self.target

class Flatness(Evaluator):
    evaluator: Evaluator

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return jnp.gradient(data, freq.f_scaled, axis=0)

class Likelihood(Evaluator):
    predictor: Evaluator
    target: jnp.ndarray
    distribution_fn: Callable[[jnp.ndarray], Distribution] = eqx.field(static=True)
    parameters: dict[str, Parameter]

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        prediction = self.predictor(model, freq)
        probability_dist = self.distribution_fn(prediction, **self.parameters)
        return probability_dist.log_prob(self.target)
        
def Diagonal(evaluator: Evaluator) -> Mapped:
    return Mapped(evaluator=evaluator, fn=lambda data: jax.vmap(jnp.diag)(data))

def OffDiagonal(evaluator: Evaluator, n_ports: int) -> Mask:
    mask = ~jnp.eye(n_ports, dtype=bool)
    return Mask(evaluator=evaluator, mask=mask)