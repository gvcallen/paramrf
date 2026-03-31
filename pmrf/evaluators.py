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
from parax import Parameter

from pmrf.core import Model, Frequency, Evaluator, Metric

# ==============================================================================
# Core evaluators
# ==============================================================================

class Lambda(Evaluator):
    """
    Wraps a standard Python or JAX callable.
    
    This is useful for defining quick, custom, on-the-fly objective 
    functions without needing to subclass Evaluator.
    
    The callable may accept arbitrary parameters set in ``self.params``.
    """
    fn: Callable[[Model, Frequency], jnp.ndarray] = prx.field(static=True)
    params: dict[str, Parameter] = prx.field(default_factory=dict, transparent=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(model, freq, **self.params)


class Constant(Evaluator):
    """
    Returns a fixed constant array or scalar.
    
    Useful for inserting fixed thresholds, reference gold-standard data, 
    or static masks directly into the evaluator tree.
    """
    value: Any = prx.field(static=True)
    
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return jnp.asarray(self.value)


class Binary(Evaluator):
    """
    Calculates a metric between two inputs.
    
    ``fn`` must have the signature ``f(left, right)``, optionally accepting
    additional key-word arguments in ``params``.
    
    Inputs can be other Evaluators, arrays, or scalars.
    """
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = prx.field(static=True)
    left: Evaluator | jnp.ndarray | float
    right: Evaluator | jnp.ndarray | float
    params: dict[str, Parameter] = prx.field(default_factory=dict, transparent=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        # Resolve left branch
        val_left = self.left(model, freq) if isinstance(self.left, Evaluator) else self.left
        
        # Resolve right branch
        val_right = self.right(model, freq) if isinstance(self.right, Evaluator) else self.right
            
        return self.fn(val_left, val_right, **self.params)


class Where(Evaluator):
    """
    A conditional branching node using `jnp.where`.

    Evaluates a boolean condition (from an Evaluator) and returns elements 
    from the `true_branch` or `false_branch` accordingly. 
    
    Useful for applying frequency-dependent logic or piecewise penalty functions.
    """
    condition: Evaluator
    true_branch: Evaluator | jnp.ndarray | float
    false_branch: Evaluator | jnp.ndarray | float

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        cond_val = self.condition(model, freq)
        
        true_val = self.true_branch(model, freq) if isinstance(self.true_branch, Evaluator) else jnp.asarray(self.true_branch)
        false_val = self.false_branch(model, freq) if isinstance(self.false_branch, Evaluator) else jnp.asarray(self.false_branch)
            
        return jnp.where(cond_val, true_val, false_val)


class Method(Evaluator):
    """
    Dynamically accesses and executes a method on the Model.
    
    This uses `operator.attrgetter` to traverse the PyTree and execute
    the requested method (e.g., 's_db', 'amplifier.s_tau') by passing 
    it the Frequency object.
    
    The method may accept arbitrary parameters set in ``self.params``.
    """
    path: str = prx.field(static=True)
    params: dict[str, Parameter] = prx.field(default_factory=dict, transparent=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        func = operator.attrgetter(self.path)(model)
        return func(freq, **self.params)


class Map(Evaluator):
    """
    Applies an arbitrary transformation function to the data.
    
    Typically used in conjunction with `jax.vmap` to apply operations like 
    matrix diagonal extraction or complex-to-magnitude conversions 
    across an entire frequency sweep.
    
    The function may accept arbitrary parameters set in ``self.params``.    
    """
    fn: Callable[[jnp.ndarray], jnp.ndarray] = prx.field(static=True)
    evaluator: Evaluator
    params: dict[str, Parameter] = prx.field(default_factory=dict, transparent=True)    

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(self.evaluator(model, freq), **self.params)


class Reduce(Evaluator):
    """
    Applies a reduction operation (e.g., max, min, mean) over a specific axis.
    
    Useful for extracting worst-case performance metrics across a frequency band
    or calculating the average value of a multi-port response.
    """
    evaluator: Evaluator
    fn: Callable[..., jnp.ndarray] = prx.field(static=True)
    axis: int | tuple[int, ...] | None = prx.field(default=None, static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        return self.fn(data, axis=self.axis)


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


class Derivative(Evaluator):
    """
    Computes the discrete numerical derivative of the data.
    
    Used to penalize ripple, enforce gain flatness, or calculate group delay.
    Supports arbitrary axes, higher-order derivatives, and configurable 
    step sizes via the `step_attr` (e.g., 'f_scaled' to prevent numerical instability).
    """
    evaluator: Evaluator
    axis: int = prx.field(default=0, static=True)
    order: int = prx.field(default=1, static=True)
    step_attr: str = prx.field(default='f_scaled', static=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        data = self.evaluator(model, freq)
        dx = operator.attrgetter(self.step_attr)(freq)
        
        for _ in range(self.order):
            data = jnp.gradient(data, dx, axis=self.axis)
            
        return data


class Objective(Evaluator):
    """
    Computes an objective by comparing an evaluator's prediction to a target using a Metric.
    """
    evaluator: Evaluator
    target: jnp.ndarray
    metric: Metric
    
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        y_pred = self.evaluator(model, freq)
        return self.metric(self.target, y_pred)

# ==============================================================================
# Generic factories
# ==============================================================================

def Index(evaluator: Evaluator, indices: Any) -> Evaluator:
    """
    Slices or indexes the output of another Evaluator.
    
    Useful for extracting specific ports or frequency ranges from a larger
    N-dimensional response array.
    """
    return Map(evaluator=evaluator, fn=lambda data: data[indices])


def Mask(evaluator: Evaluator, mask: jnp.ndarray) -> Evaluator:
    """
    Applies a boolean mask to the final dimension of the data.
    
    Utilizes `jax.vmap` to efficiently broadcast the masking operation across 
    the batch/frequency dimensions.
    """
    return Map(evaluator=evaluator, fn=lambda data: jax.vmap(lambda m: m[mask])(data))


def Sum(evaluators: list[Evaluator]) -> Evaluator:
    """
    Sums the outputs of multiple Evaluators into a single scalar.
    
    Useful for creating a composite scalar loss function from multiple 
    independent penalty vectors.
    """
    return Map(evaluator=Stack(evaluators=evaluators, axis=0), fn=jnp.sum)


def Flatness(evaluator: Evaluator) -> Evaluator:
    """
    Computes the numerical derivative of the data with respect to frequency.
    
    Used to penalize ripple or enforce gain flatness across a band. It relies on 
    `freq.f_scaled` to prevent catastrophic numerical instability.
    """
    return Derivative(evaluator=evaluator, axis=0, order=1, step_attr='f_scaled')


def Diagonal(evaluator: Evaluator) -> Evaluator:
    """Extracts the diagonals of N-port scattering matrices."""
    return Map(evaluator=evaluator, fn=lambda data: jax.vmap(jnp.diag)(data))


def OffDiagonal(evaluator: Evaluator, n_ports: int) -> Evaluator:
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
            return Diagonal(base_evaluator)
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
    loss_fn: str | Callable = 'rms',
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
    loss_fn : str | Callable, default='rms'
        The underlying mathematical metric applied to the constraint residual.
        
    Returns
    -------
    Evaluator
        A composed Metric evaluator representing the constrained goal.
    """
    predictor = Alias(feature) if isinstance(feature, str) else feature
    
    _weight = np.array(weight) if isinstance(weight, (jnp.ndarray, list)) else weight
    _mask = np.array(mask) if isinstance(mask, (jnp.ndarray, list)) else mask
    
    # 1. Resolve the metric callable securely
    from pmrf.losses import loss_from_alias
    _metric_callable = loss_from_alias(loss_fn) if isinstance(loss_fn, str) else loss_fn

    # 2. Define the clamped metric logic (The Hinge)
    def goal_metric(tgt: jnp.ndarray, pred: jnp.ndarray) -> jnp.ndarray:
        if operator in ['<', '<=']:
            effective_pred = jnp.maximum(pred, tgt)
        elif operator in ['>', '>=']:
            effective_pred = jnp.minimum(pred, tgt)
        elif operator in ['==', '=']:
            effective_pred = pred
        else:
            raise ValueError(f"Unknown Goal operator: '{operator}'")
            
        # Weighting & Masking in Residual Space
        residual = effective_pred - tgt
        weighted_residual = residual * _weight
        
        if _mask is not None:
            weighted_residual = jnp.where(_mask, weighted_residual, 0.0)
            
        # Shift back to target space and evaluate!
        final_pred = tgt + weighted_residual
        return _metric_callable(tgt, final_pred)

    return Binary(
        left=predictor,
        right=jnp.asarray(target),
        fn=goal_metric
    )

class DiscrepancyCost(Evaluator):
    evaluator: Evaluator
    discrepancy_model: prx.Module
    target: jnp.ndarray
    loss_fn: Callable
    loss_params: dict[str, prx.Parameter] = prx.field(default_factory=dict, transparent=True)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        # 1. Extract the physical feature (e.g., 's11', 's21_db')
        y_phys = self.evaluator(model, freq)
        
        # 2. Evaluate the discrepancy model
        # Contract: Must return (mean, epistemic_variance)
        y_disc_mean, y_disc_var = self.discrepancy_model(freq)
        
        # 3. Combine predictions in the feature domain
        y_total = y_phys + y_disc_mean
        
        # 4. Evaluate the loss/likelihood
        # We pass the epistemic variance down to the loss function via kwargs so 
        # probabilistic likelihoods can use it, while deterministic ones can ignore it.
        kwargs = {k: jnp.array(v) for k, v in self.loss_params.items()}
        return self.loss_fn(self.target, y_total, epistemic_var=y_disc_var, **kwargs)