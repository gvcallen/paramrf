"""
Conditioning a model on data using Bayesian inference.
"""

from typing import Callable

import jax.numpy as jnp
import distreqx.distributions as dist
import parax as prx
from inferix import PolyChord

try:
    import skrf
except ImportError:
    pass

from pmrf.core import Model, Frequency
from pmrf.constants import Inferer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Feature, MarginalLogLikelihood
from pmrf.likelihoods import GaussianLikelihood
from pmrf.infer import InferResult, sample
from pmrf.fitting.result import FitResult

def fit_sample(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Inferer = PolyChord(),
    *,
    features: str | list[str] | Callable = 's',
    likelihood_fn: Callable[[jnp.ndarray], dist.AbstractDistribution] | list[Callable[[jnp.ndarray], dist.AbstractDistribution]] = None,
    noise: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray] = None,
    discrepancy_fn: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None,
    **kwargs,
) -> FitResult:
    """
    Conditions an RF model on measured data using Bayesian sampling.
    
    This high-level function handles data format formatting (e.g., extracting arrays 
    from scikit-rf Networks) and forwards to :func:`pmrf.infer.sample`.

    Parameters
    ----------
    model : Model
        The RF model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The data to condition on. Can either be a JAX array,
        a :class:`skrf.Network`, or a :class:`pmrf.NetworkCollection`.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    solver : Solver, default=PolyChord()
        The sampler to use. Currently, only :class:`inferix.PolyChord` from
        `Inferix <https://github.com/gvcallen/inferix>`_ is supported.
    features : str | list[str] | Callable[[Model, Frequency], jnp.ndarray], default='s'
        The RF features to condition on.
        Can either be function, a callable PyTree with optional parameters, or a string,
        in which case a 'feature' evaluator is created (see :class:`pmrf.evaluators.Feature`).
        Defaults to all S-parameters.
    likelihood_fn : Callable[[jnp.ndarray], dist.AbstractDistribution], optional
        The likelihood function, which accepts a model prediction (in event space)
        and returns a distribution representing the probability of observing the data.
        Can be a function or a callable PyTree with optional parameters.
        See :mod:`pmrf.likelihoods` for common likelihoods.
        Defaults to `None`, in which case :class:`pmrf.likelihoods.GaussianLikelihood` is used.
    noise : prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray], optional
        Gaussian likelihood noise, either a fixed parameter, or a callable that accepts
        a model prediction (in event space) and returns noise parameters
        for a Gaussian likelihood. Mutually exclusive with `likelihood_fn`.
        For the function case, can be a callable PyTree with optional parameters.
        See :mod:`pmrf.likelihoods.noise_models` for built-in noise models.
        Defaults to `None`, in which case uniform variance from 0.0 to 0.1 is constructed internally.
    discrepancy_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray | dist.AbstractDistribution], optional
        A discrepancy function, which models the discrepancy between the model and measured data.
        Can either be a function, or a callable PyTree with optional parameters.
        To use a Gaussian process as a discrepancy model,
        see :class:`pmrf.discrepancy_models.GaussianProcess`.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    FitResult
        The result containing the model maximum likelikhood estimate model with an empirical posterior.
    """
    # Error checking
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise ValueError("Frequency must be passed if Network data is not provided")
    if likelihood_fn is not None and noise is not None:
        raise Exception("Cannot pass both `noise` and `likelihood_fn`.")

    # Resolve the features and data
    if not isinstance(features, Callable):
        features = Feature(features)
    if isinstance(data, skrf.Network | NetworkCollection):
        if frequency is None:
            if isinstance(data, skrf.Network):
                frequency = Frequency.from_skrf(data.frequency)
            else:
                frequency = Frequency.from_skrf(data.common_frequency())
        target = features(Measured(data), frequency)
    else:
        target = data
        
    # Resolve the likelihood model
    if likelihood_fn is None:
        if noise is None:
            noise = prx.Uniform(0.0, 0.1)
        likelihood_fn = GaussianLikelihood(noise=noise)
    
    log_likelihood_fn = MarginalLogLikelihood(predictor=features, observed=target, likelihood=likelihood_fn, discrepancy=discrepancy_fn)
    infer_result = sample(log_likelihood_fn, model, frequency, solver=solver, **kwargs)

    return FitResult(
        data=data,
        frequency=frequency,
        solution=infer_result,
    )    