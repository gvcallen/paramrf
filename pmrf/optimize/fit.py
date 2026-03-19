from functools import partial
import jax.numpy as jnp

from parax.transforms import ParameterTransform, IdentityTransform
import skrf

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.constants import EvaluatorLike, Solver, Aggregation
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.minimize import minimize_problem
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Alias, Metric
from pmrf.metrics import metric_from_alias
from pmrf.constants import MetricFn


def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    solver: Solver = ScipyMinimizer(),
    *,
    frequency: Frequency | None = None,
    features: EvaluatorLike = 's',
    metric_fn: str | MetricFn = 'rms',
    multioutput: Aggregation = 'uniform_average',
    transform: ParameterTransform = IdentityTransform(),
    **kwargs,
) -> OptimizeResult:
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise Exception("Frequency must be passed if Network data is not provided")
    if not isinstance(features, Evaluator):
        features = Alias(features)
    if isinstance(metric_fn, str):
        metric_fn = metric_from_alias(metric_fn, multioutput=multioutput)
    else:
        metric_fn = partial(metric_fn, multioutput=multioutput)
    
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

def fit_sequential(
    model: Model, 
    data: NetworkCollection,
    solver: Solver | dict[Solver] = ScipyMinimizer(),
    *,
    frequency: Frequency | dict[Frequency] | None = None,
    features: EvaluatorLike | dict[str, EvaluatorLike] = 's',
    metric_fn: str | MetricFn | dict[str | MetricFn] = 'rms',
    multioutput: Aggregation | dict[Aggregation] = 'uniform_average',
    transform: ParameterTransform | dict[ParameterTransform] = IdentityTransform(),
    **kwargs,
) -> tuple[Model, dict[str, OptimizeResult]]:
    all_results: dict[str, OptimizeResult] = {}
    
    for ntwk in data:
        name = ntwk.name
        
        sub_model = model.with_free_submodules_only(name)
        sub_data = data.filter(lambda n: n.name == name)
        sub_solver = solver[name] if isinstance(solver, dict) else solver
        sub_frequency = frequency[name] if isinstance(frequency, dict) else frequency
        sub_metric_fn = metric_fn[name] if isinstance(metric_fn, dict) else metric_fn
        sub_features = features[name] if isinstance(features, dict) else features
        sub_multioutput = multioutput[name] if isinstance(multioutput, dict) else multioutput
        sub_transform = transform[name] if isinstance(transform, dict) else transform
        
        result_sub = fit(
            sub_model,
            sub_data,
            solver=sub_solver,
            frequency=sub_frequency,
            features=sub_features,
            metric_fn=sub_metric_fn,
            transform=sub_transform,
            multioutpt=sub_multioutput,
            **kwargs
        )
        
        model = model.with_params(result_sub.model.params())
        model = model.with_param_groups(result_sub.model.param_groups(explicit_only=True))
        
        all_results[name] = result_sub
    
    return model, all_results