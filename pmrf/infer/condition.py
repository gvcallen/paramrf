"""
Conditioning a model on data using Bayesian inference.
"""

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
from pmrf.evaluators import Alias, Metric
from pmrf.infer.result import InferResult
from pmrf.infer.sample import sample


def condition(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Inferer = PolyChord(),
    *,
    features: EvaluatorLike = ('s_re', 's_im'),
    distribution_fn: Callable[..., dist.AbstractDistribution] = None,
    log_likelihood_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = None,
    likelihood_params: dict[str, prx.Parameter] = None,
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
        The parametric model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The target data to fit against. Can be raw JAX arrays or standard Touchstone networks.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    solver : Solver, default=PolyChord()
        The Bayesian sampling algorithm backend (e.g., PolyChord, MultiNest).
    features : EvaluatorLike, default=('s_re', 's_im')
        The specific circuit feature(s) to compute the likelihood against. 
        Usually passed as a tuple of real and imaginary parts for Bayesian analysis.
    distribution_fn : Callable, optional
        The distreqx distribution class representing the likelihood model.
        Must accept the model prediction as its first argument, `likelihood_params`
        as key-word arguments, and implement `log_prob` which accepts the data.
        Used to create a Metric evaluator from :class:`pmrf.evaluators.Metric`.
        Mutually exclusive with `log_likelihood_fn`.
    log_likelihood_fn: Callable, optional
        A likelihood function to use, which takes the true and predicted features as
        first and second arguments, `likelihood_params` as key-word arguments,
        and returns the log likelihood. Used to create a Metric evaluator from :class:`pmrf.evaluators.Metric`.
        Mutually exclusive with `distribution_fn`.
    likelihood_params : dict[str, prx.Parameter], optional
        Additional parameters characterizing the likelihood model. Defaults to a uniform 
        scale parameter if None. Passed to :class:`pmrf.evaluators.Metric`.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    InferenceResult
        The result containing the model loaded with empirical posterior distributions.
    """
    if distribution_fn is None and log_likelihood_fn is None:
        distribution_fn = dist.Normal
        likelihood_params = {'sigma': prx.Uniform(0.0, 100.0, scale=1e-3)}
    elif distribution_fn is not None and log_likelihood_fn is not None:
        raise Exception("Cannot pass both `distribution_fn` and `likelihood_params`")
    
    if distribution_fn:
        def dist_log_likelihood_fn(y_true, y_pred, **params):
            return jnp.sum(distribution_fn(y_true, **params).log_prob(y_pred))
        log_likelihood_fn = dist_log_likelihood_fn
        distribution_fn = None
    
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise ValueError("Frequency must be passed if Network data is not provided")
    if not isinstance(features, Evaluator):
        features = Alias(features)
    
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
    
    likelihood = Metric(evaluator=features, reference=target, fn=log_likelihood_fn, params=likelihood_params)
    return sample(likelihood, model, frequency, solver=solver, **kwargs)