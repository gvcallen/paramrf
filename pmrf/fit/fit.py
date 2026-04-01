from typing import Any, Callable

import jax.numpy as jnp
import skrf

from pmrf.core import Model, Frequency, EvaluatorLike
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.network_collection import NetworkCollection
from pmrf.optimize import fit as optimize_fit, is_optimizer
from pmrf.infer import condition as infer_condition, is_inferer
from pmrf.optimize.result import OptimizeResult
from pmrf.infer.result import InferResult
from pmrf.fit.result import FitResult
from pmrf.constants import Optimizer, Inferer

def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Optimizer | Inferer = ScipyMinimizer(),
    *,
    features: EvaluatorLike | None = None,    
    **kwargs
) -> FitResult:
    """
    Fit a model to data using either optimization or sampling.

    This is a unified router. The execution path is determined by the
    type of `solver` provided.

    Parameters
    ----------
    model : Model
        The parametric model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The observed data (e.g., S-parameters).
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array.
    solver : Optimizer | Sampler, default=ScipyMinimizer()
        The solver to use. If an optimizer, routes to frequentist minimization
        via :meth:`pmrf.optimize.fit`. If a sampler, routes to Bayesian inference
        via :meth:`pmrf.infer.condition`.
    features : EvaluatorLike | None, default=None
        The specific circuit feature to evaluate. If None, it defers to the 
        native default of the chosen solver backend ('s' for optimization, 
        ('s_re', 's_im') for inference).
    **kwargs : dict
        Additional arguments passed directly to the underlying fit function.

    Returns
    -------
    OptimizeResult | InferResult
        A result object containing the newly fitted model. Depending on the solver, 
        the model contains either optimized point-estimates or empirical posteriors.
    """
    if features is not None:
        kwargs['features'] = features
        
    if frequency is None:
        if isinstance(data, skrf.Network):
            frequency = Frequency.from_skrf(data.frequency)
        elif isinstance(data, NetworkCollection):
            frequency = Frequency.from_skrf(data.common_frequency())        

    if is_optimizer(solver):
        history = optimize_fit(model=model, data=data, frequency=frequency, solver=solver, **kwargs)
    elif is_inferer(solver):
        history = infer_condition(model=model, data=data, frequency=frequency, solver=solver, **kwargs)
    else:
        raise TypeError(
            f"Unrecognized solver type: {type(solver)}. "
            "Solver must be a valid optimizer or inferer."
        )
        
    result = FitResult(
        data=data,
        frequency=frequency,
        solution=history,
    )
    
    return result

def fit_sequential(
    model: Model, 
    data: NetworkCollection,
    *,
    features: EvaluatorLike | dict[str, EvaluatorLike] | None = None,
    dynamic_kwargs: dict[str, dict[str, Any] | Callable[[skrf.Network], Any]] | None = None,
    **kwargs,
) -> tuple[Model, dict[str, FitResult]]:
    """
    Sequentially fits sub-modules of a circuit using either 
    optimization or sampling.
    
    For each network in the network collection, the network's
    name is used as a prefix for the features to fit,
    and :meth:`pmrf.fit.fit` is called.

    Parameters
    ----------
    model : Model
        The global circuit model.
    data : NetworkCollection
        A collection of network data whose names are used as prefixes for sub-model features.
    features : EvaluatorLike | dict | None, default=None
        The circuit feature(s) to evaluate for each sub-model. If None, defers to the backend's defaults.
    dynamic_kwargs : dict[str, dict | Callable[[skrf.Network], Any]] | None, default=None
        A mapping of keyword arguments that should be resolved dynamically per network. 
        If a value is a dict, it is resolved using the network name as the key.
        If a value is a callable, it is resolved by passing the network to the callable.
    **kwargs : dict
        Standard static kwargs passed directly to the underlying sequential fitters for all iterations.

    Returns
    -------
    tuple[Model, dict[str, OptimizeResult | InferenceResult]]
        The fully updated global Model, and a dictionary of localized results.
    """
    if features is None:
        features = 's'

    # Initialize dynamic_kwargs safely
    dynamic_kwargs = dynamic_kwargs or {}
    all_results: dict[str, OptimizeResult] = {}
    
    for ntwk in data:
        name = ntwk.name
        
        # Isolate the free parameters of this specific sub-module for the optimizer
        sub_model = model.with_free_submodules_only(name)
        sub_data = data.filter(lambda n: n.name == name)
        
        # Resolve localized arguments for features
        if isinstance(features, str):
            sub_features = f"{name}.{features}"
        else:
            sub_features = [f"{name}.{feature}" for feature in features]

        # Resolve dynamic kwargs (callables and dicts)
        resolved_dynamics = {}
        
        if sub_features is not None:
            resolved_dynamics['features'] = sub_features
        
        for key, value in dynamic_kwargs.items():
            if callable(value):
                resolved_dynamics[key] = value(ntwk)
            elif isinstance(value, dict):
                if name in value:
                    resolved_dynamics[key] = value[name]
                else:
                    raise KeyError(f"Dynamic kwarg '{key}' is a dict but missing configuration for network '{name}'")
            else:
                # Fallback just in case a static value is accidentally passed here
                resolved_dynamics[key] = value

        # Merge standard kwargs with resolved dynamic kwargs. 
        # dynamic_kwargs will overwrite static kwargs if there is a name collision.
        final_kwargs = {**kwargs, **resolved_dynamics}
        
        # Fit the sub-module
        result_sub = fit(
            sub_model,
            sub_data,
            **final_kwargs,
        )
        
        model = model.merged(result_sub.model)
        all_results[name] = result_sub
    
    return model, all_results