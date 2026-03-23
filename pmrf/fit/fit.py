import jax.numpy as jnp
import skrf

import optimistix as optx
import inferix as infx

from pmrf.core import Model, Frequency
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.network_collection import NetworkCollection
from pmrf.optimize import fit as optimize_fit
from pmrf.optimize import fit_sequential as optimize_fit_sequential
from pmrf.infer import fit as infer_fit
from pmrf.infer import fit_sequential as infer_fit_sequential
from pmrf.optimize.result import OptimizeResult
from pmrf.infer.result import InferResult
from pmrf.constants import Optimizer, Inferer, EvaluatorLike

def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    engine: Optimizer | Inferer = ScipyMinimizer(),
    *,
    features: EvaluatorLike | None = None,    
    **kwargs
) -> OptimizeResult | InferResult:
    """
    Fit a model to data using either optimization or sampling.

    This is a unified router. The execution path is determined by the
    type of `engine` provided.

    Parameters
    ----------
    model : Model
        The parametric model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The observed data (e.g., S-parameters).
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array.
    engine : Optimizer | Sampler, default=ScipyMinimizer()
        The engine to use. If an optimizer, routes to frequentist minimization
        via :meth:`pmrf.optimize.fit`. If a sampler, routes to Bayesian inference
        via :meth:`pmrf.infer.fit`.
    features : EvaluatorLike | None, default=None
        The specific circuit feature to evaluate. If None, it defers to the 
        native default of the chosen engine backend ('s' for optimization, 
        ('s_re', 's_im') for inference).
    **kwargs : dict
        Additional arguments passed directly to the underlying fit function.

    Returns
    -------
    OptimizeResult | InferenceResult
        A result object containing the newly fitted model. Depending on the engine, 
        the model contains either optimized point-estimates or empirical posteriors.
    """
    if features is not None:
        kwargs['features'] = features

    if isinstance(engine, optx.AbstractMinimiser | ScipyMinimizer):
        return optimize_fit(model=model, data=data, frequency=frequency, solver=engine, **kwargs)
    elif isinstance(engine, infx.AbstractNestedSampler | infx.AbstractHostHypercubeNestedSampler):
        return infer_fit(model=model, data=data, frequency=frequency, sampler=engine, **kwargs)
    else:
        raise TypeError(
            f"Unrecognized engine type: {type(engine)}. "
            "Engine must be a valid solver or sampler."
        )


def fit_sequential(
    model: Model, 
    data: NetworkCollection,
    engine: Optimizer | Inferer | dict[str, Optimizer | Inferer] = ScipyMinimizer(),
    *,
    frequency: Frequency | dict[str, Frequency] | None = None,
    features: EvaluatorLike | dict[str, EvaluatorLike] | None = None,
    **kwargs,
) -> tuple[Model, dict[str, OptimizeResult | InferResult]]:
    """
    Sequentially fits sub-modules of a complex circuit cascade using either 
    optimization or sampling.

    This acts as a unified router for sequential fitting, dynamically dispatching 
    to either Bayesian inference or Frequentist minimization depending on the 
    underlying `engine`.

    Parameters
    ----------
    model : Model
        The global circuit model.
    data : NetworkCollection
        A collection of network data whose names match the sub-modules in the model.
    engine : Optimizer | Sampler | dict, default=ScipyMinimizer()
        The engine to use. If a dictionary is provided, the type of the first 
        engine in the mapping determines the execution path.
    frequency : Frequency | dict | None, default=None
        The frequency sweep(s).
    features : EvaluatorLike | dict | None, default=None
        The specific circuit feature(s) to evaluate. If None, defers to the backend's defaults.
    **kwargs : dict
        Additional kwargs passed to the underlying sequential fitters.

    Returns
    -------
    tuple[Model, dict[str, OptimizeResult | InferenceResult]]
        The fully updated global Model, and a dictionary of localized results.
    """
    if features is not None:
        kwargs['features'] = features

    # Safely determine the engine type, checking the first value if it's a dict
    first_engine = next(iter(engine.values())) if isinstance(engine, dict) else engine

    if isinstance(first_engine, optx.AbstractMinimiser | ScipyMinimizer):
        return optimize_fit_sequential(
            model=model, data=data, solver=engine, frequency=frequency, **kwargs
        )
    elif isinstance(first_engine, infx.AbstractNestedSampler | infx.AbstractHostHypercubeNestedSampler):
        return infer_fit_sequential(
            model=model, data=data, sampler=engine, frequency=frequency, **kwargs
        )
    else:
        raise TypeError(
            f"Unrecognized engine type: {type(first_engine)}. "
            "Engine must be a valid solver or sampler."
        )