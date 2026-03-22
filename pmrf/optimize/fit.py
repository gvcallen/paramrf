from functools import partial
from typing import Callable
import jax.numpy as jnp
import skrf
from distreqx.bijectors import AbstractBijector

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.math_functions import FUNC_LOOKUP
from pmrf.constants import EvaluatorLike, Solver, Aggregation
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.minimize import minimize_problem
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Alias, Metric
from pmrf.metrics import metric_from_alias
from pmrf.constants import MetricFn


def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    solver: Solver = ScipyMinimizer(),
    *,
    frequency: Frequency | None = None,
    features: EvaluatorLike = 's',
    metric_fn: str | MetricFn = 'rms',
    multioutput: Aggregation = 'uniform_average',
    metric_transform_fn: str | Callable[[jnp.ndarray], jnp.ndarray] = 'db',
    transform: AbstractBijector | None = None,
    **kwargs,
) -> OptimizeResult:
    """
    Fits an RF circuit model to empirical measurement data.

    This high-level function handles data format coercion (e.g., extracting arrays 
    from scikit-rf Networks) and automatically composes the necessary evaluator metrics.

    Parameters
    ----------
    model : Model
        The circuit model to optimize.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The target data to fit against. Can be raw JAX arrays or standard Touchstone networks.
    solver : Solver, default=ScipyMinimizer()
        The optimization algorithm backend.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    features : EvaluatorLike, default='s'
        The specific circuit feature to fit (e.g., 's', 's11_db', 'y').
    metric_fn : str | MetricFn, default='rms'
        The mathematical loss metric (e.g., 'mse', 'mae', 'rms') comparing prediction to data.
        See :meth:`pmrf.metric_from_alias`.
    multioutput : Aggregation, default='uniform_average'
        How to aggregate losses across multiple ports/outputs.
    metric_transform_fn : str | MetricFn, default='rms'
        A transform to apply to the output metric after aggregation. See :meth:`pmrf.math_functions`.
    transform : ParameterTransform, default=None
        An invertible transformation to apply to all model parameters before optimization.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    OptimizeResult
        The optimization result containing the newly fitted Model.
    """
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise Exception("Frequency must be passed if Network data is not provided")
    if not isinstance(features, Evaluator):
        features = Alias(features)
    if isinstance(metric_fn, str):
        metric_fn = metric_from_alias(metric_fn, multioutput=multioutput)
    else:
        metric_fn = partial(metric_fn, multioutput=multioutput)
    if isinstance(metric_transform_fn, str):
        metric_transform_fn = FUNC_LOOKUP[metric_transform_fn][1]

    def transformed_metric_fn(y_true, y_pred):
        metric = metric_fn(y_true, y_pred)
        return metric_transform_fn(metric)
    
    # Standardize target data and frequencies
    if isinstance(data, skrf.Network | NetworkCollection):
        if frequency is None:
            if isinstance(data, skrf.Network):
                frequency = Frequency.from_skrf(data.frequency)
            else:
                frequency = Frequency.from_skrf(data.common_frequency())
        target = features(Measured(data), frequency)
    else:
        target = data
    
    metric_evaluator = Metric(features, target, transformed_metric_fn)
    problem = Problem(model, frequency, metric_evaluator)
    
    return minimize_problem(problem, solver, transform=transform, **kwargs)


def fit_sequential(
    model: Model, 
    data: NetworkCollection,
    solver: Solver | dict[str, Solver] = ScipyMinimizer(),
    *,
    frequency: Frequency | dict[str, Frequency] | None = None,
    features: EvaluatorLike | dict[str, EvaluatorLike] = 's',
    metric_fn: str | MetricFn | dict[str, MetricFn | str] = 'rms',
    multioutput: Aggregation | dict[str, Aggregation] = 'uniform_average',
    transform: AbstractBijector | None = None,
    **kwargs,
) -> tuple[Model, dict[str, OptimizeResult]]:
    """
    Sequentially fits sub-modules of a complex circuit cascade.

    Iterates through a collection of networks, extracting matching sub-modules 
    from the main model. Each module is fitted locally, and the main model's state 
    is continuously updated. This ensures downstream components are fitted against 
    the updated physical realities of their upstream neighbors.

    Parameters
    ----------
    model : Model
        The global circuit model.
    data : NetworkCollection
        A collection of network data whose names match the sub-modules in the model.
    solver, frequency, features, metric_fn, multioutput, transform:
        Optimization settings. These can either be single global rules, or dictionaries 
        mapping the sub-module's string name to specific localized rules.
    **kwargs : dict
        Additional kwargs passed to the solvers.

    Returns
    -------
    tuple[Model, dict[str, OptimizeResult]]
        The fully updated global Model, and a dictionary of localized optimization results.
    """
    all_results: dict[str, OptimizeResult] = {}
    
    for ntwk in data:
        name = ntwk.name
        
        # Isolate the free parameters of this specific sub-module for the optimizer
        sub_model = model.with_free_submodules_only(name)
        sub_data = data.filter(lambda n: n.name == name)
        
        # Resolve localized arguments if dicts are provided
        sub_solver = solver[name] if isinstance(solver, dict) else solver
        sub_frequency = frequency[name] if isinstance(frequency, dict) else frequency
        sub_metric_fn = metric_fn[name] if isinstance(metric_fn, dict) else metric_fn
        sub_features_base = features[name] if isinstance(features, dict) else features
        sub_features = f"{name}.{sub_features_base}"
        sub_multioutput = multioutput[name] if isinstance(multioutput, dict) else multioutput
        sub_transform = transform[name] if isinstance(transform, dict) else transform
        
        result_sub = fit(
            sub_model,
            sub_data,
            solver=sub_solver,
            frequency=sub_frequency,
            features=sub_features,
            metric_fn=sub_metric_fn,
            transform=sub_transform,
            multioutput=sub_multioutput,
            **kwargs
        )
        
        # IMMEDIATE UPDATE: Stitch the fitted parameters back into the main state
        model = model.with_params(result_sub.model.params())
        model = model.with_param_groups(result_sub.model.param_groups(explicit_only=True))
        
        all_results[name] = result_sub
    
    return model, all_results