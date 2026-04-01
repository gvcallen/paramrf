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

from pmrf.core import Model, Frequency, Operator
from pmrf.constants import EvaluatorLike, Inferer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Feature, Binary
from pmrf.likelihoods import distribution_log_likelihood, symmetric_gaussian_log_likelihood
from pmrf.infer.result import InferResult
from pmrf.infer.sample import sample

def condition(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Inferer = PolyChord(),
    *,
    features: EvaluatorLike = 's',
    log_likelihood_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = None,
    log_likelihood_params: dict[str, prx.Parameter] = None,
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
    log_likelihood_fn: Callable, optional
        A likelihood function to use, which takes the true and predicted features as
        first and second arguments, `log_likelihood_params` as key-word arguments,
        and returns the log likelihood. If not provided, defaults to :func:`pmrf.likelihoods.symmetric_gaussian_likelihood`.
        Used to create a Binary evaluator via :class:`pmrf.evaluators.Binary`.
    log_likelihood_params : dict[str, prx.Parameter], optional
        Additional parameters to pass to ``log_likelihood_fn``. Defaults to a uniform 
        scale parameter if None. Passed to :class:`pmrf.evaluators.Binary`.
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
    if not isinstance(features, Operator):
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
    if log_likelihood_params is None:
        log_likelihood_params = {}        
    if log_likelihood_fn is None:
        log_likelihood_fn = symmetric_gaussian_log_likelihood
        log_likelihood_params = {'sigma': prx.Uniform(0.0, 100.0, scale=1e-3)}
    log_likelihood_model = Binary(fn=log_likelihood_fn, left=target, right=features, params=log_likelihood_params)
    
    # Run the sampling
    return sample(log_likelihood_model, model, frequency, solver=solver, **kwargs)