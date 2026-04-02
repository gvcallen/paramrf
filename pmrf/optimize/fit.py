from functools import partial
from typing import Callable

import jax.numpy as jnp
import skrf
from distreqx.bijectors import AbstractBijector
import parax as prx

from pmrf.core import Model, Frequency
from pmrf.math import CONVERSION_LOOKUP, LOSS_LOOKUP
from pmrf.constants import Optimizer, AggregationKind
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Alias, Objective
from pmrf.losses import LogMSELoss

from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.minimize import minimize
from pmrf.optimize.solvers import ScipyMinimizer

def fit(
    model: Model,
    data: jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Optimizer = ScipyMinimizer(),
    *,
    features: str | list[str] | Callable = 's',
    loss_fn: str | Callable = LogMSELoss(),
    multioutput: AggregationKind | None = None,
    scale_fn: str | Callable | None = None,
    transform: AbstractBijector | None = None,
    **kwargs,
) -> OptimizeResult:
    """
    Fits an RF model to measured data using frequentist optimization.

    This high-level function handles data format coercion (e.g., extracting arrays 
    from scikit-rf Networks) and automatically composes the necessary evaluator metrics.

    Parameters
    ----------
    model : Model
        The model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The target data to fit against. Can be raw JAX arrays or standard Touchstone networks.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    solver : Solver, default=ScipyMinimizer()
        The optimization algorithm backend.
    features : str | list[str] | Callable, default='s'
        The specific circuit feature to fit (e.g., 's', 's11_db', 'y').
    loss_fn : str | Callable, default=LogMSELoss()
        The loss function between the model predictionand the data.
        Can be a string for a lookup into :data:``pmrf.math.LOSS_LOOKUP``        
        (e.g., 'mse', 'mae', 'rmse'), a callable taking (y_true, y_pred),
        or a callable PyTree. See :mod:``pmrf.losses`` for common losses.
    multioutput : Aggregation, optional
        An additional key-word parameter to optionally pass to ``loss_fn`` indicating
        how to aggregate outputs. For the default of `None`, the argument is not passed.
    scale_fn : str | Callable, default=None
        A scaling to apply to the output metric after aggregation.
        Can be a string for a lookup into :data:``pmrf.math.CONVERSION_LOOKUP``.
    transform : ParameterTransform, default=None
        An invertible transformation to apply to all model parameters before optimization.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    OptimizeResult
        The optimization result containing the newly fitted Model.
    """
    # Error checking
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise Exception("Frequency must be passed if Network data is not provided")
    
    # Resolve data and features
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
    
    # Resolve the loss model    
    if isinstance(loss_fn, str):
        loss_fn = LOSS_LOOKUP[loss_fn][1]
    if multioutput is not None:
        loss_fn = partial(loss_fn, multioutput=multioutput)
    cost_fn = Objective(metric=loss_fn, evaluator=features, target=target)
    
    # Append an optional scale function
    if isinstance(scale_fn, str):
        scale_fn = CONVERSION_LOOKUP[scale_fn][1]
    if scale_fn is not None:
        scaled_cost_fn = prx.op.Map(scale_fn, cost_fn)
    else:
        scaled_cost_fn = cost_fn

    # Run the optimizer
    return minimize(scaled_cost_fn, model, frequency, solver, transform=transform, **kwargs)