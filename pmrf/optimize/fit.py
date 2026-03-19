from typing import Callable

import jax.numpy as jnp

import optimistix as optx
from parax.transforms import ParameterTransform, IdentityTransform
import skrf

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.minimize import minimize_problem
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Alias, Metric
from pmrf.metrics import metric_from_alias
from pmrf.constants import MetricFn


def fit_data(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    solver: optx.AbstractMinimiser | Callable = ScipyMinimizer(),
    *,
    frequency: Frequency | None = None,
    features: str | list[str] | Evaluator = 's',
    metric_fn: str | MetricFn = 'rms',
    transform: ParameterTransform = IdentityTransform(),
    **kwargs,
) -> OptimizeResult:
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise Exception("Frequency must be passed if Network data is not provided")
    if isinstance(features, (str, list)):
        features = Alias(features)
    if isinstance(metric_fn, str):
        metric_fn = metric_from_alias(metric_fn)
    
    # Make sure data and frequency are in the right format
    if isinstance(data, skrf.Network | NetworkCollection):
        if frequency is None:
            if isinstance(data, skrf.Network):
                frequency = Frequency.from_skrf(data.frequency)
            else:
                frequency = Frequency.from_skrf(data.common_frequency())
        target = features(Measured(data), frequency)
    else:
        target = data
    
    metric_evaluator = Metric(features, target, metric_fn)
    problem = Problem(model, frequency, metric_evaluator)
    
    return minimize_problem(problem, solver, transform=transform, **kwargs)