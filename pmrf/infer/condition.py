"""
Conditioning a model on data using Bayesian inference.
"""

from typing import Callable

import jax.numpy as jnp
import skrf
import distreqx.distributions as dist
import parax as prx
from inferix import PolyChord

from pmrf.core import Model, Frequency
from pmrf.constants import Inferer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Feature, MarginalLogLikelihood

from pmrf.likelihoods import GaussianLikelihood
from pmrf.infer.result import InferResult
from pmrf.infer.sample import sample

def condition(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Inferer = PolyChord(),
    *,
    features: str | list[str] | Callable = 's',
    likelihood_fn: Callable[[jnp.ndarray], dist.AbstractDistribution] | list[Callable[[jnp.ndarray], dist.AbstractDistribution]] = None,
    discrepancy_fn: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None,
    **kwargs,
) -> InferResult:
    """
    Conditions an RF model on measured data using Bayesian inference.
    
    This high-level function handles data format coercion (e.g., extracting arrays 
    from scikit-rf Networks) and automatically composes the necessary evaluator metrics
    to compute the log-likelihood over the parameter space.

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
    discrepancy_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray | dist.AbstractDistribution], optional
        A discrepancy function, which models the discrepancy between the model and measured data.
        Can either be a function, or a callable PyTree with optional parameters.
        To use a Gaussian process as a discrepancy model,
        see :class:`pmrf.discrepancy_models.GaussianProcess`.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    InferResult
        The result containing the model maximum likelikhood estimate model with an empirical posterior.
    """
    # Error checking
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise ValueError("Frequency must be passed if Network data is not provided")

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
        likelihood_fn = GaussianLikelihood(sigma=prx.Uniform(0.0, 100.0, scale=1e-3))
    
    log_likelihood_fn = MarginalLogLikelihood(predictor=features, data=target, likelihood=likelihood_fn, discrepancy=discrepancy_fn)
    
    # Run the sampling
    return sample(log_likelihood_fn, model, frequency, solver=solver, **kwargs)