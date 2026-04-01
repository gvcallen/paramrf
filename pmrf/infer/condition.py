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
from pmrf.evaluators import Alias, Objective

from pmrf.likelihoods import SymmetricGaussianLikelihood
from pmrf.infer.result import InferResult
from pmrf.infer.sample import sample

def condition(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Inferer = PolyChord(),
    *,
    features: str | list[str] | Callable = 's',
    log_likelihood_fn: str | Callable = None,
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
    log_likelihood_fn : str | Callable, optional
        The log likelihood function between the model prediction and the data.
        Can be a callable taking (y_true, y_pred), or a callable PyTree.
        See :mod:``pmrf.likelihoods`` for common likelihoods.
        Defaults to `None`, in which case :class:``pmrf.likelihoods.SymmetricGaussianLikelihood``
        is constructed internally.
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
        features = Alias(features)
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
    if log_likelihood_fn is None:
        log_likelihood_fn = SymmetricGaussianLikelihood(sigma=prx.Uniform(0.0, 100.0, scale=1e-3))
    
    objective_fn = Objective(metric=log_likelihood_fn, predictor=features, target=target)  
    
    # Run the sampling
    return sample(objective_fn, model, frequency, solver=solver, **kwargs)