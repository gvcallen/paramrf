from typing import Callable

import jax.numpy as jnp
import skrf
import distreqx.distributions as dist
import parax as prx
from inferix import PolyChord

from pmrf.core import Model, Frequency, Evaluator
from pmrf.constants import EvaluatorLike, Sampler
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Alias, Likelihood
from pmrf.infer.result import InferenceResult
from pmrf.infer.sample import sample


def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    sampler: Sampler = PolyChord(),
    *,
    features: EvaluatorLike = ('s_re', 's_im'),
    distribution_fn: Callable[..., dist.AbstractDistribution] = dist.Normal,
    distribution_params: dict[str, prx.Parameter] = None,
    **kwargs,
) -> InferenceResult:
    """
    Fits an RF model to measured data using Bayesian inference.
    
    This high-level function handles data format coercion (e.g., extracting arrays 
    from scikit-rf Networks) and automatically composes the necessary evaluator metrics.
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