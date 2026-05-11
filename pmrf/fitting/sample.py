"""
Conditioning a model on data using Bayesian inference.
"""

from typing import Callable

import jax.numpy as jnp
import distreqx.distributions as dist
import parax as prx
from pmrf.infer import PolyChord

try:
    import skrf
except ImportError:
    pass

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.constants import Solver
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Feature, MarginalLogLikelihood, GibbsMarginalLogLikelihood
from pmrf.likelihoods import GaussianLikelihood
from pmrf.infer import sample
from pmrf.fitting.result import FitResult
from pmrf.parameters import Param, Random
from pmrf.distributions import Normal

def fit_sample(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Solver = PolyChord(),
    *,
    features: str | list[str] | Callable = 's',
    likelihood: Callable[[jnp.ndarray], dist.AbstractDistribution] | list[Callable[[jnp.ndarray], dist.AbstractDistribution]] = None,
    noise: Param | Callable[[jnp.ndarray], jnp.ndarray] = None,
    loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = None,
    discrepancy: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None,
    temperature: float = None,
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
        The sampler to use.
        Can be any sampler from :mod:`pmrf.infer` (such as :mod:`pmrf.infer.PolyChord).
    features : str | list[str] | Callable[[Model, Frequency], jnp.ndarray], default='s'
        The RF features to condition on.
        Can either be function, a callable PyTree with optional parameters, or a string,
        in which case a 'feature' evaluator is created (see :class:`pmrf.evaluators.Feature`).
        Defaults to all S-parameters.
    likelihood : Callable[[jnp.ndarray], dist.AbstractDistribution], optional
        The likelihood model, which accepts a model prediction (in event space)
        and returns a distribution representing the probability of observing the data.
        Can be a function or a callable PyTree with optional parameters.
        See :mod:`pmrf.likelihoods` for common likelihoods.
        Mutually exclusive with `loss`.
    noise : prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray], optional
        Likelihood noise, either a fixed parameter, or a callable that accepts
        a model prediction (in event space) and returns noise parameters
        for a Gaussian likelihood. Mutually exclusive with `likelihood` and `loss`.
        For the function case, can be a callable PyTree with optional parameters.
        See :mod:`pmrf.noise_models` for built-in noise models.
        Defaults to `None`, in which case uniform variance from 0.0 to 0.1 is constructed internally.
    loss : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], optional
        A loss function between the model prediction and the data to construct a Gibbs measure.
        Can be a function or a callable PyTree with optional parameters.
        Mutually exclusive with `likelihood` and `noise`. If neither `loss` nor `likelihood` 
        is passed, a :class:`pmrf.likelihoods.GaussianLikelihood` is constructed.
    discrepancy : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray | dist.AbstractDistribution], optional
        A discrepancy model, which caters for the discrepancy between the model and measured data.
        Can either be a function, or a callable PyTree with optional parameters.
        To use a Gaussian process as a discrepancy model,
        see :class:`pmrf.discrepancys.GaussianProcess`.
    temperature : float, optional
        The temperature value for generalized Bayesian optimization.
        Only used when `loss` is not None. Defaults to 1.0 internally.
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
    if likelihood is not None and noise is not None:
        raise ValueError("Cannot pass both `noise` and `likelihood`.")
    if loss is not None and likelihood is not None:
        raise ValueError("Only one of either `loss` or `likelihood` can be passed to `fit_sample`.")
    if loss is not None and noise is not None:
        raise ValueError("Cannot pass `noise` when using a `loss` function for Generalized Bayesian Inference.")

    # Resolve the features and data
    if not isinstance(features, Callable):
        features = Feature(features)
    if isinstance(data, skrf.Network | NetworkCollection):
        if frequency is None:
            if isinstance(data, skrf.Network):
                frequency = Frequency.from_skrf(data.frequency)
            else:
                frequency = Frequency.from_skrf(data.common_frequency())
        observed = features(Measured(data), frequency)
    else:
        observed = data
        
    # Resolve the likelihood or loss model
    if loss is None and likelihood is None:
        if noise is None:
            noise = Random(Normal(0.0, 0.01))
        likelihood = GaussianLikelihood(noise=noise)
    
    if likelihood is not None:
        loglikelihood = MarginalLogLikelihood(
            predictor=features, 
            observed=observed, 
            likelihood=likelihood, 
            discrepancy=discrepancy
        )
    else:
        temperature = temperature if temperature is not None else 1.0
        loglikelihood = GibbsMarginalLogLikelihood(
            predictor=features, 
            observed=observed, 
            loss=loss, 
            discrepancy=discrepancy, 
            temperature=temperature
        )

    infer_result = sample(loglikelihood, model, frequency, solver=solver, **kwargs)

    return FitResult(
        data=data,
        frequency=frequency,
        solution=infer_result,
    )