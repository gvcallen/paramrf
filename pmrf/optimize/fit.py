from functools import partial
from typing import Callable
import dataclasses

import jax.numpy as jnp
import skrf
from distreqx.bijectors import AbstractBijector
import parax as prx

from pmrf.core import Model, Frequency, Evaluator
from pmrf.math_functions import FUNC_LOOKUP
from pmrf.constants import EvaluatorLike, Optimizer, AggregationKind
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.minimize import minimize
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Alias, Loss
from pmrf.losses import loss_from_alias
from pmrf.constants import MetricFn


def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Optimizer = ScipyMinimizer(),
    *,
    features: EvaluatorLike = 's',
    loss_fn: str | MetricFn = 'mse',
    multioutput: AggregationKind | None = None,
    loss_params: dict[str, prx.Parameter] = None,
    scale_fn: str | Callable[[jnp.ndarray], jnp.ndarray] = None,
    transform: AbstractBijector | None = None,
    **kwargs,
) -> OptimizeResult:
    """
    Fits an RF model to measured data using frequentist optimization.

    This high-level function handles data format coercion (e.g., extracting arrays 
    from scikit-rf Networks) and automatically composes the necessary evaluator metrics.

    Parameters
    ----------
    model : Model
        The model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The target data to fit against. Can be raw JAX arrays or standard Touchstone networks.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    solver : Solver, default=ScipyMinimizer()
        The optimization algorithm backend.
    features : EvaluatorLike, default='s'
        The specific circuit feature to fit (e.g., 's', 's11_db', 'y').
    loss_fn : str | MetricFn, default='rms'
        The mathematical loss metric (e.g., 'mse', 'mae', 'rms') comparing prediction to data.
        See :meth:`pmrf.losses.loss_from_alias`. Used to construct a loss evaluator
        from :class`pmrf.evalutors.Loss`.
    loss_params : dict[str, parax.Parameter], optional
        Loss parameters to pass to the loss function. This can be useful to define
        hyper-parameters e.g. for regulariziation. Used in :class`pmrf.evalutors.Loss`.
        Defaults to None.
    multioutput : Aggregation, optional
        An additional key-word parameter to optionally pass to ``loss_fn`` indicating
        how to aggregate outputs. For the default of `None`, the argument is not passed.
    scale_fn : str | Callable, default=None
        A scaling to apply to the output metric after aggregation. See :meth:`pmrf.math_functions`.
    transform : ParameterTransform, default=None
        An invertible transformation to apply to all model parameters before optimization.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    OptimizeResult
        The optimization result containing the newly fitted Model.
    """
    if loss_params is None:
        loss_params = {}
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise Exception("Frequency must be passed if Network data is not provided")
    if not isinstance(features, Evaluator):
        features = Alias(features)
    if isinstance(loss_fn, str):
        loss_fn = loss_from_alias(loss_fn)
    if multioutput is not None:
        loss_fn = partial(loss_fn, multioutput=multioutput)
    if isinstance(scale_fn, str):
        scale_fn = FUNC_LOOKUP[scale_fn][1]

    def scaled_loss_fn(y_true, y_pred, **kwargs):
        metric = loss_fn(y_true, y_pred, **kwargs)
        if scale_fn is not None:
            return scale_fn(metric)
        else:
            return metric
    
    # Standardize target data and frequencies
    if isinstance(data, skrf.Network | NetworkCollection):
        if frequency is None:
            if isinstance(data, skrf.Network):
                frequency = Frequency.from_skrf(data.frequency)
            else:
                frequency = Frequency.from_skrf(data.common_frequency())
        
        # Keep a reference to the PyTree-safe Measured model!
        measured_data = Measured(data)
        target = features(measured_data, frequency)
    else:
        measured_data = data
        target = data
    
    cost = Loss(features, target, scaled_loss_fn, params=loss_params)
    result = minimize(cost, model, frequency, solver, transform=transform, **kwargs)
    
    return dataclasses.replace(
        result, 
        data=measured_data,
        frequency=frequency
    )


def fit_sequential(
    model: Model, 
    data: NetworkCollection,
    solver: Optimizer | Callable[[Model, skrf.Network], Optimizer] = ScipyMinimizer(),
    *,
    frequency: Frequency | None = None,
    features: EvaluatorLike = 's',
    loss_fn: str | MetricFn = 'rms',
    loss_params: dict[str, prx.Parameter] | Callable[[Model, skrf.Network], dict[str, prx.Parameter]] = None,
    multioutput: AggregationKind | None = None,
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
    solver, frequency, features, loss_fn, multioutput, transform:
        Optimization settings. These can either be single global rules, or dictionaries 
        mapping the sub-module's string name to specific localized rules.
    **kwargs : dict
        Additional kwargs passed to :meth:`pmrf.fit.fit`.

    Returns
    -------
    tuple[Model, dict[str, OptimizeResult]]
        The fully updated global Model, and a dictionary of localized optimization results.
    """
    if isinstance(features, str):
        features = [features]

    all_results: dict[str, OptimizeResult] = {}
    
    for ntwk in data:
        name = ntwk.name
        
        # Isolate the free parameters of this specific sub-module for the optimizer
        sub_model = model.with_free_submodules_only(name)
        sub_data = data.filter(lambda n: n.name == name)
        
        # Resolve localized arguments
        sub_features = [f"{name}.{feature}" for feature in features]

        # Resolve loss parameters
        sub_loss_params = loss_params(sub_model, ntwk) if callable(loss_params) else loss_params
        
        result_sub = fit(
            sub_model,
            sub_data,
            solver=solver,
            frequency=frequency,
            features=sub_features,
            loss_fn=loss_fn,
            loss_params=sub_loss_params,
            transform=transform,
            multioutput=multioutput,
            **kwargs
        )
        
        model = model.merged(result_sub.model)
        all_results[name] = result_sub
    
    return model, all_results