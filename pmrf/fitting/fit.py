from typing import Any, Callable

import jax.numpy as jnp

try:
    import skrf
except ImportError:
    pass

from pmrf.core import Model, Frequency, EvaluatorLike
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.network_collection import NetworkCollection
from pmrf.optimize import is_optimizer, OptimizeResult
from pmrf.infer import is_inferer, InferResult
from pmrf.fitting.result import FitResult
from pmrf.fitting.minimize import fit_minimize
from pmrf.fitting.sample import fit_sample
from pmrf.constants import Optimizer, Inferer

def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Optimizer | Inferer = ScipyMinimizer(),
    *,
    features: EvaluatorLike | None = 's',
    **kwargs
) -> FitResult:
    """
    Fit a model to data using a variety of methods.

    This is a unified router to either :meth:`pmrf.fitting.fit_minimize`
    or :meth:`pmrf.fitting.fit_sample`. The execution path is determined by the
    type of `solver` provided. All key-word arguments are forward appropriately.

    Parameters
    ----------
    model : Model
        The RF model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The data to fit. Can either be a JAX array,
        a :class:`skrf.Network`, or a :class:`pmrf.NetworkCollection`.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array.
    solver : Optimizer | Sampler, default=ScipyMinimizer()
        The solver to use. If an optimizer is passed, routes to frequentist minimization
        via :meth:`pmrf.fitting.fit_minimize`. If a sampler, routes to Bayesian inference
        via :meth:`pmrf.fitting.fit_sample`. Can be either in instance of :class:`pmrf.optimize.ScipyMinimizer`,
        a minimizer from `Optimistix <https://docs.kidger.site/optimistix/api/minimise>`_
        (such as :class:`optimistix.LBFGS`) or a sampler from `Inferix <https://github.com/gvcallen/inferix>`_
        (such as :class:`inferix.PolyChord`).
    features : EvaluatorLike | None, default='s'
        The RF features to fit. Defaults to all S-parameters.
        Can either be an instance of :class:`pmrf.Evaluator` or a string,
        in which case a 'feature' evaluator is created (see :class:`pmrf.evaluators.Feature`).
    **kwargs : dict
        Additional arguments passed to the underlying solver.

    Returns
    -------
    FitResult
        A result object containing the fitted model and backend solution results.
        Frequentist optimizers return a single best model, whereas Bayesian inferers also
        return full posterior distributions on the model.
    """
    if is_optimizer(solver):
        return fit_minimize(model=model, data=data, frequency=frequency, solver=solver, features=features, **kwargs)
    elif is_inferer(solver):
        return fit_sample(model=model, data=data, frequency=frequency, solver=solver, features=features, **kwargs)
    else:
        raise TypeError(
            f"Unrecognized solver type: {type(solver)}. "
            "Solver must be a valid optimizer or inferer."
        )
        
def fit_sequential(
    model: Model, 
    data: NetworkCollection,
    *,
    features: EvaluatorLike | dict[str, EvaluatorLike] | None = 's',
    dynamic_kwargs: dict[str, dict[str, Any] | Callable[[skrf.Network], Any]] | None = None,
    **kwargs,
) -> tuple[Model, dict[str, FitResult]]:
    """
    Sequentially fits sub-modules of a circuit using either 
    optimization or sampling.
    
    For each network in the network collection, the network's
    name is used as a prefix for the features to fit,
    and :meth:`pmrf.fit` is called.

    Parameters
    ----------
    model : Model
        The RF model to fit.
    data : NetworkCollection
        A collection of network data whose names are used as prefixes for sub-model features.
    features : EvaluatorLike | None, default='s'
        The RF features to fit. Defaults to all S-parameters.
        Can either be an instance of :class:`pmrf.Evaluator` or a string,
        in which case a 'feature' evaluator is created (see :class:`pmrf.evaluators.Feature`).
    dynamic_kwargs : dict[str, dict | Callable[[skrf.Network], Any]] | None, default=None
        A mapping of keyword arguments that should be resolved dynamically per network. 
        If a value is a dict, it is resolved using the network name as the key.
        If a value is a callable, it is resolved by passing the network to the callable.
    **kwargs : dict
        Standard kwargs passed to :func:`pmrf.fit`.

    Returns
    -------
    tuple[Model, dict[str, OptimizeResult | InferenceResult]]
        The fully updated global Model, and a dictionary of localized results.
    """
    # Initialize dynamic_kwargs safely
    dynamic_kwargs = dynamic_kwargs or {}
    all_results: dict[str, OptimizeResult | InferResult] = {}
    
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