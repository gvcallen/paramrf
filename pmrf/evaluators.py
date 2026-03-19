from typing import Callable, Any, Sequence
import operator
import re

import jax
import jax.numpy as jnp
import parax as prx
from numpyro.distributions import Distribution
from parax import Parameter

from pmrf.core import Model, Frequency, Evaluator

class Functional(Evaluator):
    fn: Callable[[Model, Frequency], jnp.ndarray] = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(model, freq)

class Method(Evaluator):
    path: str = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        func = operator.attrgetter(self.path)(model)
        return func(freq)

class Index(Evaluator):
    evaluator: Evaluator
    indices: Any = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return data[self.indices]

class Mask(Evaluator):
    evaluator: Evaluator
    mask: jnp.ndarray = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return jax.vmap(lambda m: m[self.mask])(data)

class Mapped(Evaluator):
    evaluator: Evaluator
    fn: Callable[[jnp.ndarray], jnp.ndarray] = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(self.evaluator(model, freq))
        
class Stacked(Evaluator):
    evaluators: list[Evaluator]
    axis: int = prx.field(default=-1, static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        results = [ev(model, freq) for ev in self.evaluators]
        return jnp.stack(results, axis=self.axis)

class Residual(Evaluator):
    predictor: Evaluator
    target: jnp.ndarray

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.predictor(model, freq) - self.target

class Metric(Evaluator):
    predictor: Evaluator
    target: jnp.ndarray
    metric_fn: Callable[[jnp.ndarray, jnp.ndarray]] = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        predicted = self.predictor(model, freq)
        return self.metric_fn(self.target, predicted)

class Flatness(Evaluator):
    evaluator: Evaluator

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return jnp.gradient(data, freq.f_scaled, axis=0)

class Likelihood(Evaluator):
    predictor: Evaluator
    target: jnp.ndarray
    distribution_fn: Callable[[jnp.ndarray], Distribution] = prx.field(static=True)
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

def Alias(alias: str | Sequence[str]) -> Evaluator:
    """
    Factory function to generate Evaluators from string aliases.
    Accepts a single string (e.g., 's11_db', 'amplifier.s_tau') or a list of strings.
    """
    # 1. Handle Sequences (wrap in Stacked)
    if not isinstance(alias, str) and isinstance(alias, Sequence):
        evaluators = [Alias(a) for a in alias]
        return Stacked(evaluators=evaluators, axis=-1)
    
    # 2. Parse submodel paths (e.g., "submodel.s11_db")
    fields = alias.split('.')
    if len(fields) > 1:
        subattrs = ".".join(fields[:-1]) 
        local_alias = fields[-1]
    else:
        subattrs = ""
        local_alias = fields[0]

    # 3. Intercept special aliases: s_gamma (reflection) and s_tau (transmission)
    # By replacing the prefix, "s_gamma_db" elegantly becomes "s_db" for the base method!
    special_type = None
    if local_alias.startswith('s_gamma'):
        base_prop = local_alias.replace('s_gamma', 's', 1)
        special_type = 'gamma'
    elif local_alias.startswith('s_tau'):
        base_prop = local_alias.replace('s_tau', 's', 1)
        special_type = 'tau'
        
    if special_type:
        path = f"{subattrs}.{base_prop}" if subattrs else base_prop
        base_evaluator = Method(path=path)
        
        if special_type == 'gamma':
            # Extract diagonals for all frequencies (shape: n_freqs, n_ports)
            return Mapped(evaluator=base_evaluator, fn=lambda data: jax.vmap(jnp.diag)(data))
        else:
            # Extract off-diagonals dynamically (shape: n_freqs, n_ports^2 - n_ports)
            def extract_off_diag(mat):
                return mat[~jnp.eye(mat.shape[0], dtype=bool)]
            return Mapped(evaluator=base_evaluator, fn=lambda data: jax.vmap(extract_off_diag)(data))

    # 4. Standard regex matching for specific ports (e.g., s11_db, y21_deg)
    match = re.match(r'^([a-zA-Z]+)(\d)?(\d)?(.*)$', local_alias)
    if not match:
        raise ValueError(f"Invalid feature alias format: '{alias}'")

    prop_prefix = match.group(1)
    port1 = match.group(2)
    port2 = match.group(3)
    prop_suffix = match.group(4)
    
    prop = prop_prefix + prop_suffix
    path = f"{subattrs}.{prop}" if subattrs else prop
    
    evaluator = Method(path=path)

    # 5. Map 1-indexed string alias to 0-indexed port array slices
    if port1 is not None and port2 is not None:
        # Translates to [:, port1-1, port2-1]
        indices = (slice(None), int(port1) - 1, int(port2) - 1)
        evaluator = Index(evaluator=evaluator, indices=indices)
        
    return evaluator