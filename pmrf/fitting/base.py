from typing import Any, Callable
from dataclasses import replace

import jax.numpy as jnp
import parax as prx
import skrf

from pmrf.core import Model, Frequency, Evaluator, EvaluatorLike
from pmrf.optimize import is_optimizer, OptimizeResult, ScipyMinimize
from pmrf.infer import is_inferer, InferResult
from pmrf.evaluators import Feature
from pmrf.models import Measured
from pmrf.network_collection import NetworkCollection
from pmrf.constants import Optimizer, Inferer

from pmrf.fitting.minimize import fit_minimize
from pmrf.fitting.sample import fit_sample

class FitResult(prx.Module):
    """
    Standardized return object for a fitting routines.

    Wraps an :class:`pmrf.optimize.OptimizeResult` or :class:`pmrf.infer.InferResult`
    with added information for easy plotting.
    """
    #: The data used for the fit.
    data: Any = None
        
    #: The frequeny used for the fit.
    frequency: Frequency | None = None

    #: The underlying :class:`pmrf.optimize.OptimizeResult` or :class:`pmrf.infer.InferResult` solution.
    solution: OptimizeResult | InferResult | None = None
    
    @property
    def model(self):
        return self.solution.model

    def plot(self, features: str | list[str] | Evaluator = 's', ax=None, **kwargs):
        """
        Plots the best fit.
        
        Parameters
        ----------
        feature : str | Evaluator
            The specific feature to extract and plot (e.g., 's_real', 's_mag_db', 
            or a custom Evaluator instance).
        """
        from pmrf.viz.plots import plot_fit_result
        return plot_fit_result(self, features=features, ax=ax, **kwargs)


    def __getattr__(self, name: str):
        """
        Intercepts calls to undefined methods to dynamically route plot requests.
        Example: `plot_s_db(m=0, n=1)` routes to `plot('s21_db')`.
        """
        if name.startswith('plot_') and name != 'plot':
            # Extract the base feature name (e.g., 's_db' from 'plot_s_db')
            feature_base = name[5:] 
            
            def dynamic_plot(m=None, n=None, ax=None, **kwargs):
                feature_name = feature_base
                
                # If m and n are provided, inject them into the feature string
                if m is not None and n is not None:
                    parts = feature_base.split('_', 1)
                    
                    # Convert 0-based to 1-based indexing. 
                    # The prompt's mapping of (m=0, n=1) to '21' means n comes first.
                    # Adjust {n+1}{m+1} to {m+1}{n+1} here if that was a typo for '12'.
                    port_str = f"{n+1}{m+1}" 
                    
                    if len(parts) == 2:
                        # e.g. parts[0]='s', parts[1]='db' -> 's21_db'
                        feature_name = f"{parts[0]}{port_str}_{parts[1]}"
                    else:
                        # Fallback for things like plot_s(m=0, n=1) -> 's21'
                        feature_name = f"{feature_base}{port_str}" 
                        
                return self.plot(features=feature_name, ax=ax, **kwargs)
            
            return dynamic_plot
            
        # Standard fallback if the missing attribute isn't a plot_ hook
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    

def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Optimizer | Inferer = ScipyMinimize(),
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
    solver : Optimizer | Sampler, default=ScipyMinimize()
        The solver to use.
        If an optimizer is passed, routes to frequentist minimization via :meth:`pmrf.fitting.fit_minimize`.
        If a sampler, routes to Bayesian inference via :meth:`pmrf.fitting.fit_sample`.
        Can be either an instance of :class:`pmrf.optimize.ScipyMinimize`, an instance of :class:`pmrf.infer.PolyChord`,
        a minimizer from `Optimistix <https://docs.kidger.site/optimistix/api/minimise>`_ (such as :class:`optimistix.LBFGS`),
        or a sampler from `Inferix <https://github.com/gvcallen/inferix>`_ (such as :class:`inferix.NUTS`).
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

        if len(sub_data.networks) > 1:
            raise ValueError(f"Multiple sets of data with the same name found in `fit_sequential`. Name: {name}")
        
        sub_ntwk = sub_data.networks[0]
        
        # Resolve dynamic kwargs (callables and dicts)
        resolved_dynamics = {}
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
        frequency = final_kwargs.setdefault('frequency', Frequency.from_skrf(sub_ntwk.frequency))
        features = final_kwargs.pop('features', 's')
        if isinstance(features, str):
            sub_features = f"{name}.{features}"
        else:
            sub_features = [f"{name}.{feature}" for feature in features]        

        sub_array = Feature(features)(Measured(sub_ntwk), frequency)

        try:
            # Fit the sub-module
            result_sub = fit(
                sub_model,
                sub_array,
                features=sub_features,
                **final_kwargs,
            )

            result_sub = replace(result_sub, data=sub_ntwk)

        except Exception as e:
            raise Exception(f"Error fitting {name}: {e}")

        model = model.merged(result_sub.model)
        all_results[name] = result_sub
    
    return model, all_results


def fit_joint(
    model: Model, 
    data: NetworkCollection,
    **kwargs,
) -> FitResult:
    """
    Jointly fits all sub-modules of a circuit using the same features with either 
    optimization or sampling.
    
    This function processes the entire network collection at once. It uses
    the networks' names as prefixes for the features to fit, mapping the 
    global model parameters to the entire dataset in a single solve.

    Parameters
    ----------
    model : Model
        The RF model to fit.
    data : NetworkCollection
        A collection of network data whose names are used as prefixes for sub-model features.
    **kwargs : dict
        Standard kwargs passed to :func:`pmrf.fit`.

    Returns
    -------
    FitResult
        A single result object containing the globally fitted model and backend 
        solution results. (The updated model can be accessed via `result.model`).
    """
    # 1. Prevent feature collision
    # Ensure all network names are unique so the solver doesn't overwrite features
    names = [ntwk.name for ntwk in data]
    if len(names) != len(set(names)):
        raise ValueError(
            "Multiple networks with the same name found in `data`. "
            "Names must be unique for joint fitting."
        )

    # 2. Resolve frequency
    # We assume a shared frequency sweep across the NetworkCollection for joint 
    # fitting. If one isn't explicitly provided, extract it from the first network.
    if 'frequency' not in kwargs and hasattr(data, 'networks') and data.networks:
        kwargs['frequency'] = Frequency.from_skrf(data.networks[0].frequency)
        
    # 3. Resolve sub-feature indexing across the entire collection
    features = kwargs.pop('features', 's')
    joint_features = []
    
    for ntwk in data:
        name = ntwk.name
        if isinstance(features, str):
            joint_features.append(f"{name}.{features}")
        else:
            # Handle iterable of feature strings (e.g., ['s11', 's21'])
            joint_features.extend([f"{name}.{feature}" for feature in features])

    # 4. Execute the simultaneous fit
    # We pass the global model and the full data collection to the base fit function
    return fit(
        model=model,
        data=data,
        features=joint_features,
        **kwargs,
    )