from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp

import optimistix as optx

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.minimize import minimize_problem
from pmrf.network_collection import NetworkCollection
from pmrf.metrics import root_mean_squared_error
from pmrf.models import Measured
from pmrf.evaluators import Method, Alias, Metric
from pmrf.constants import MetricFn
from pmrf.constants import SolverSpace

if TYPE_CHECKING:
    import skrf

def fit_data(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    solver: optx.AbstractMinimiser | None = None,
    *,
    frequency: Frequency | None = None,
    features: Evaluator | str | list[str] = Method('s'),
    metric_fn: MetricFn = root_mean_squared_error,
    space: SolverSpace = None,
    **kwargs,
) -> OptimizeResult:
    if frequency is None and isinstance(data, jnp.ndarray):
        raise Exception("Frequency must be passed if Network data is not provided")
    
    if isinstance(features, (str, list)):
        features = Alias(features)
    
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
    
    return minimize_problem(problem, solver, space=space, **kwargs)