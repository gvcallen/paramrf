"""
Callables that extract frequency-dependent features from a model.
"""

from typing import Callable, Any, Sequence, Literal
import operator
import re

import numpy as np
import jax
import jax.numpy as jnp
import parax as prx
from distreqx.distributions import AbstractDistribution
from parax import Parameter

from pmrf.metrics import metric_from_alias
from pmrf.core import Model, Frequency, Evaluator


class Functional(Evaluator):
    """
    Wraps a standard Python or JAX callable.
    
    This is useful for defining quick, custom, on-the-fly objective 
    functions without needing to subclass Evaluator.
    """
    fn: Callable[[Model, Frequency], jnp.ndarray] = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(model, freq)


class Method(Evaluator):
    """
    Dynamically accesses and executes a method on the Model.
    
    This uses `operator.attrgetter` to traverse the PyTree and execute
    the requested method (e.g., 's_db', 'amplifier.s_tau') by passing 
    it the Frequency object.
    """
    path: str = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        func = operator.attrgetter(self.path)(model)
        return func(freq)


class Index(Evaluator):
    """
    Slices or indexes the output of another Evaluator.
    
    Useful for extracting specific ports or frequency ranges from a larger
    N-dimensional response array.
    """
    evaluator: Evaluator
    indices: Any = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return data[self.indices]


class Mask(Evaluator):
    """
    Applies a boolean mask to the final dimension of the data.
    
    Utilizes `jax.vmap` to efficiently broadcast the masking operation across 
    the batch/frequency dimensions.
    """
    evaluator: Evaluator
    mask: jnp.ndarray = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return jax.vmap(lambda m: m[self.mask])(data)


class Map(Evaluator):
    """
    Applies an arbitrary transformation function to the data.
    
    Typically used in conjunction with `jax.vmap` to apply operations like 
    matrix diagonal extraction across an entire frequency sweep.
    """
    evaluator: Evaluator
    fn: Callable[[jnp.ndarray], jnp.ndarray] = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(self.evaluator(model, freq))
        
        
class Stack(Evaluator):
    """
    Combines the outputs of multiple evaluators into a single array.
    
    By default, it stacks the results along the last axis (-1), which is useful 
    for aggregating disparate metrics (e.g., S11 and S21) into a single residual vector.
    """
    evaluators: list[Evaluator]
    axis: int = prx.field(default=-1, static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        results = [ev(model, freq) for ev in self.evaluators]
        return jnp.stack(results, axis=self.axis)


class Sum(Evaluator):
    """
    Sums the outputs of multiple Evaluators.
    
    Useful for creating a composite scalar loss function from multiple 
    independent penalty vectors.
    """
    evaluators: list[Evaluator]

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        results = [ev(model, freq) for ev in self.evaluators]
        return jnp.sum(jnp.array(results))


class Residual(Evaluator):
    """
    Calculates the raw difference between a prediction and a target.
    
    Formula: $e = y_{pred} - y_{target}$
    """
    predictor: Evaluator
    target: jnp.ndarray

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.predictor(model, freq) - self.target


class Metric(Evaluator):
    """
    Applies a standard mathematical metric to the prediction.
    
    The `metric_fn` must have the signature `f(y_true, y_pred)`. Standard ML 
    metrics like MSE, MAE, or RMS should be routed through this component.
    """
    predictor: Evaluator
    target: jnp.ndarray
    metric_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = prx.field(static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        predicted = self.predictor(model, freq)
        return self.metric_fn(self.target, predicted)


class Flatness(Evaluator):
    """
    Computes the numerical derivative of the data with respect 
    to frequency.
    
    Used to penalize ripple or enforce gain flatness across a band. It relies on 
    `freq.f_scaled` to prevent catastrophic numerical instability during JAX 
    gradient calculations caused by raw Hz values (e.g., 1e9).
    """
    evaluator: Evaluator

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return jnp.gradient(data, freq.f_scaled, axis=0)


class Likelihood(Evaluator):
    """
    Calculates the log-probability of the target data given the model.
    
    Crucial for Bayesian inference and MCMC sampling. It wraps a distreqx 
    distribution parameterized by the circuit's predictions and returns the 
    likelihood of observing the `target` data.
    """
    predictor: Evaluator
    target: jnp.ndarray
    distribution_fn: Callable[[jnp.ndarray], AbstractDistribution] = prx.field(static=True)
    params: dict[str, Parameter] = prx.field(transparent=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        prediction = self.predictor(model, freq)
        dist_kwargs = {k: jnp.array(v) for k, v in self.params.items()}
        probability_dist = self.distribution_fn(prediction, **dist_kwargs)
        return jnp.sum(probability_dist.log_prob(self.target))
      
        
def Diagonal(evaluator: Evaluator) -> Map:
    """Extracts the diagonals of N-port scattering matrices."""
    return Map(evaluator=evaluator, fn=lambda data: jax.vmap(jnp.diag)(data))


def OffDiagonal(evaluator: Evaluator, n_ports: int) -> Mask:
    """Extract the off-diagonals (transmission) of N-port matrices."""
    mask = ~jnp.eye(n_ports, dtype=bool)
    return Mask(evaluator=evaluator, mask=mask)


def Alias(alias: str | Sequence[str] | list[Evaluator]) -> Evaluator:
    """
    Generate Evaluators from shorthand string aliases.
    
    Parses regex patterns to automatically route strings like 's11_db' or 
    'amplifier.y21_deg' to the correct Method and Index evaluators.
    
    Parameters
    ----------
    alias : str | Sequence[str]
        A string mapping to a method/port (e.g., 's11_db') or a list of such strings.
        
    Returns
    -------
    Evaluator
        The fully composed evaluator chain mapping to the requested feature.
    """
    if isinstance(alias, list) and isinstance(alias[0], Evaluator):
        return Sum(alias)
    
    # 1. Handle Sequences (wrap in Stacked)
    if not isinstance(alias, str) and isinstance(alias, Sequence):
        evaluators = [Alias(a) for a in alias]
        return Stack(evaluators=evaluators, axis=-1)
    
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
            return Map(evaluator=base_evaluator, fn=lambda data: jax.vmap(jnp.diag)(data))
        else:
            # Extract off-diagonals dynamically (shape: n_freqs, n_ports^2 - n_ports)
            def extract_off_diag(mat):
                return mat[~jnp.eye(mat.shape[0], dtype=bool)]
            return Map(evaluator=base_evaluator, fn=lambda data: jax.vmap(extract_off_diag)(data))

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


def Goal(
    feature: str | Evaluator,
    operator: Literal['<', '<=', '>', '>=', '==', '='] = '==',
    target: float | jnp.ndarray = 0.0,
    weight: float | jnp.ndarray = 1.0,
    mask: jnp.ndarray | None = None,
    metric_fn: str | Callable = 'rms',
) -> Evaluator:
    """
    Define a specific one-sided or strict design goal.
    
    Utilizes a differentiable clamping technique (hinge loss) to ensure the 
    optimizer only experiences a penalty gradient when the constraint is violated.
    
    Parameters
    ----------
    feature : str | Evaluator
        The circuit feature to evaluate, either as an instantiated Evaluator or
        an Alias string (e.g., 's11_db').
    operator : Literal['<', '<=', '>', '>=', '==', '='], default='=='
        The logical constraint operator.
    target : float | jnp.ndarray, default=0.0
        The target value or array the feature must satisfy.
    weight : float | jnp.ndarray, default=1.0
        A scalar or array multiplier to scale the importance of this goal.
    mask : jnp.ndarray | None, default=None
        A boolean array filtering which frequencies apply to this goal.
    metric_fn : str | Callable, default='rms'
        The underlying mathematical metric applied to the constraint residual.
        
    Returns
    -------
    Evaluator
        A composed Metric evaluator representing the constrained goal.
        
    Examples
    --------
    >>> Goal('s11_db', '<', -20)
    >>> Goal('s21_db', '>', -1, weight=10.0, mask=(freq.f > 2e9), metric_fn='mae')
    """
    predictor = Alias(feature) if isinstance(feature, str) else feature
    
    # Note: If Metric.metric_fn is static=True, closing over JAX tracers can break Equinox.
    # We cast to standard numpy arrays here to keep the static closure safe during JIT.
    _weight = np.array(weight) if isinstance(weight, (jnp.ndarray, list)) else weight
    _mask = np.array(mask) if isinstance(mask, (jnp.ndarray, list)) else mask
    
    # 1. Resolve the metric callable securely
    _metric_callable = metric_from_alias(metric_fn) if isinstance(metric_fn, str) else metric_fn

    # 2. Define the clamped metric logic
    def goal_metric(tgt: jnp.ndarray, pred: jnp.ndarray) -> jnp.ndarray:
        # Step A: Clamping (The Hinge)
        # This completely zeros out the error for satisfied constraints by mapping
        # the prediction perfectly onto the target.
        if operator in ['<', '<=']:
            effective_pred = jnp.maximum(pred, tgt)
        elif operator in ['>', '>=']:
            effective_pred = jnp.minimum(pred, tgt)
        elif operator in ['==', '=']:
            effective_pred = pred
        else:
            raise ValueError(f"Unknown Goal operator: '{operator}'")
            
        # Step B: Weighting & Masking in Residual Space
        residual = effective_pred - tgt
        weighted_residual = residual * _weight
        
        if _mask is not None:
            weighted_residual = jnp.where(_mask, weighted_residual, 0.0)
            
        # Step C: Shift back to target space and evaluate!
        final_pred = tgt + weighted_residual
        return _metric_callable(tgt, final_pred)

    return Metric(
        predictor=predictor,
        target=jnp.asarray(target),
        metric_fn=goal_metric
    )