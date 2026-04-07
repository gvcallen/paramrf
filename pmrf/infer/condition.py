"""
Conditioning a model on data using Bayesian inference.
"""

from typing import Callable
from functools import partial

import jax.numpy as jnp
import skrf
import distreqx.distributions as dist
import parax as prx
from inferix import PolyChord

from pmrf.core import Model, Frequency
from pmrf.math import CONVERSION_LOOKUP, LOSS_LOOKUP
from pmrf.constants import Inferer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Feature, DataLikelihood

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
        The target data to fit against. Can be raw JAX arrays or standard Touchstone networks.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    solver : Solver, default=PolyChord()
        The Bayesian sampling algorithm backend (e.g., PolyChord, MultiNest).
    features : EvaluatorLike, default='s'
        The specific circuit feature(s) to compute the likelihood against. 
        Usually passed as a tuple of real and imaginary parts for Bayesian analysis.
    likelihood_fn : Callable[[jnp.ndarray], dist.AbstractDistribution], optional
        The likelihood function that accepts a model prediction and returns a distribution
        representing the probability of observing data given that prediction.
        Can be a function or a callable PyTree. See :mod:``pmrf.likelihoods`` for common likelihoods.
        Defaults to `None`, in which case :class:``pmrf.likelihoods.GaussianLikelihood``
        is constructed internally with a symmetric noise model.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    InferenceResult
        The result containing the model loaded with empirical posterior distributions.
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
    
    # if isinstance(likelihood_fn, list):
        
    
    log_likelihood_fn = DataLikelihood(predictor=features, data=target, likelihood=likelihood_fn)  
    
    # Run the sampling
    return sample(log_likelihood_fn, model, frequency, solver=solver, **kwargs)