from typing import Callable

import jax.numpy as jnp
import skrf
import distreqx.distributions as dist
import parax as prx
from inferix import PolyChord

from pmrf.core import Model, Frequency, Evaluator
from pmrf.constants import EvaluatorLike, Inferer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Alias, Likelihood
from pmrf.infer.result import InferResult
from pmrf.infer.sample import sample


def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    sampler: Inferer = PolyChord(),
    *,
    features: EvaluatorLike = ('s_re', 's_im'),
    distribution_fn: Callable[..., dist.AbstractDistribution] = dist.Normal,
    distribution_params: dict[str, prx.Parameter] = None,
    **kwargs,
) -> InferResult:
    """
    Fits an RF model to measured data using Bayesian inference.
    
    This high-level function handles data format coercion (e.g., extracting arrays 
    from scikit-rf Networks) and automatically composes the necessary evaluator metrics
    to compute the log-likelihood over the parameter space.

    Parameters
    ----------
    model : Model
        The parametric model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The target data to fit against. Can be raw JAX arrays or standard Touchstone networks.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    sampler : Sampler, default=PolyChord()
        The Bayesian sampling algorithm backend (e.g., PolyChord, MultiNest).
    features : EvaluatorLike, default=('s_re', 's_im')
        The specific circuit feature(s) to compute the likelihood against. 
        Usually passed as a tuple of real and imaginary parts for Bayesian analysis.
    distribution_fn : Callable, default=dist.Normal
        The distreqx distribution class representing the likelihood noise model.
    distribution_params : dict[str, prx.Parameter], optional
        The parameters characterizing the likelihood noise model. Defaults to a uniform 
        scale parameter if None.
    **kwargs : dict
        Additional keyword arguments passed to the underlying sampler.

    Returns
    -------
    InferenceResult
        The result containing the model loaded with empirical posterior distributions.
    """
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise ValueError("Frequency must be passed if Network data is not provided")
    if not isinstance(features, Evaluator):
        features = Alias(features)
    if distribution_params is None:
        distribution_params = {'scale': prx.Uniform(0.0, 20.0, scale=1e-3)}
    
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
    
    likelihood = Likelihood(features, target, distribution_fn, distribution_params)
    return sample(likelihood, model, frequency, sampler=sampler, **kwargs)


def fit_sequential(
    model: Model, 
    data: NetworkCollection,
    sampler: Inferer | dict[str, Inferer] = PolyChord(),
    *,
    frequency: Frequency | dict[str, Frequency] | None = None,
    features: EvaluatorLike | dict[str, EvaluatorLike] = ('s_re', 's_im'),
    distribution_fn: Callable[..., dist.AbstractDistribution] | dict[str, Callable[..., dist.AbstractDistribution]] = dist.Normal,
    distribution_params: dict[str, prx.Parameter] | dict[str, dict[str, prx.Parameter]] = None,
    **kwargs,
) -> tuple[Model, dict[str, InferResult]]:
    """
    Sequentially fits sub-modules of a complex circuit cascade using Bayesian inference.

    Iterates through a collection of networks, extracting matching sub-modules 
    from the main model. Each module is sampled locally, and the main model's state 
    is continuously updated with the posterior distributions. This ensures downstream 
    components are evaluated against the inferred physical realities of their upstream neighbors.

    Parameters
    ----------
    model : Model
        The global circuit model.
    data : NetworkCollection
        A collection of network data whose names match the sub-modules in the model.
    sampler, frequency, features, distribution_fn, distribution_params:
        Inference settings. These can either be single global rules, or dictionaries 
        mapping the sub-module's string name to specific localized rules.
    **kwargs : dict
        Additional kwargs passed to the samplers.

    Returns
    -------
    tuple[Model, dict[str, InferenceResult]]
        The fully updated global Model (loaded with empirical posteriors), 
        and a dictionary of localized inference results.
    """
    all_results: dict[str, InferResult] = {}
    
    for ntwk in data:
        name = ntwk.name
        
        # Isolate the free parameters of this specific sub-module for the sampler
        sub_model = model.with_free_submodules_only(name)
        sub_data = data.filter(lambda n: n.name == name)
        
        # Resolve localized arguments if dicts are provided
        sub_sampler = sampler[name] if isinstance(sampler, dict) else sampler
        sub_frequency = frequency[name] if isinstance(frequency, dict) else frequency
        
        # Resolve features, appending the sub-module prefix safely
        sub_features_base = features[name] if isinstance(features, dict) else features
        if isinstance(sub_features_base, tuple):
            sub_features = tuple(f"{name}.{f}" if isinstance(f, str) else f for f in sub_features_base)
        elif isinstance(sub_features_base, str):
            sub_features = f"{name}.{sub_features_base}"
        else:
            sub_features = sub_features_base

        sub_dist_fn = distribution_fn[name] if isinstance(distribution_fn, dict) else distribution_fn
        sub_dist_params = distribution_params[name] if isinstance(distribution_params, dict) and name in distribution_params else distribution_params
        
        result_sub = fit(
            sub_model,
            sub_data,
            frequency=sub_frequency,
            sampler=sub_sampler,
            features=sub_features,
            distribution_fn=sub_dist_fn,
            distribution_params=sub_dist_params,
            **kwargs
        )
        
        # Update the global model with the empirical posteriors of the sub-module
        model = model.merged(result_sub.model)
        all_results[name] = result_sub
    
    return model, all_results