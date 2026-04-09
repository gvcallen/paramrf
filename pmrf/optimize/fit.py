from functools import partial
from typing import Callable

import jax.numpy as jnp
try:
    import skrf
except ImportError:
    pass
import parax as prx

from pmrf.core import Model, Frequency
from pmrf.math import CONVERSION_LOOKUP, LOSS_LOOKUP
from pmrf.constants import Optimizer, AggregationKind
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Feature, TargetLoss
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
    loss_fn: Callable = LogMSELoss(),
    multioutput: AggregationKind | None = None,
    scale_fn: str | Callable | None = None,
    **kwargs,
) -> OptimizeResult:
    """
    Fits an RF model to measured data using frequentist optimization.

    This high-level function handles data format coercion (e.g., extracting arrays 
    from scikit-rf Networks) and automatically composes the necessary evaluator metrics.
    
    Parameters
    ----------
    model : Model
        The RF model to fit.
    data : jnp.ndarray | skrf.Network | NetworkCollection
        The data to fit to. Can either be a JAX array,
        a :class:`skrf.Network`, or a :class:`pmrf.NetworkCollection`.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    solver : Solver, default=ScipyMinimizer()
        The optimizer to use. Can be either in instance of :class:`pmrf.optimize.ScipyMinimizer`
        or a minimizer from `Optimistix <https://docs.kidger.site/optimistix/api/minimise>`_
        (such as :class:`optimistix.LBFGS`).
    features : str | list[str] | Callable[[Model, Frequency], jnp.ndarray], default='s'
        The RF features to fit.
        Can either be function, a callable PyTree with optional parameters, or a string,
        in which case a feature evaluator is created (see :class:`pmrf.evaluators.Feature`).
        Defaults to all S-parameters.
    loss_fn : str | Callable, default=LogMSELoss()
        The loss function between the model prediction and the data.
        Can be a function, a callable PyTree with optional parameters, or a string
        for a lookup into :data:`pmrf.math.LOSS_LOOKUP`
        See :mod:`pmrf.losses` for common losses.
        Defaults to `None`, in which case :class:`pmrf.losses.LogMSELoss` is used.
    multioutput : Aggregation, optional
        An additional key-word parameter to optionally pass to `loss_fn` indicating
        how to aggregate outputs. For the default of None, the argument is not passed.
    scale_fn : str | Callable, default=None
        A scaling to apply to the output metric after aggregation.
        Can be a string for a lookup into :data:`pmrf.math.CONVERSION_LOOKUP`.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    OptimizeResult
        The optimization result containing the fitted Model.
    """
    # Error checking
    if isinstance(data, jnp.ndarray) and frequency is None:
        raise Exception("Frequency must be passed if Network data is not provided")
    
    # Resolve data and features
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
    
    # Resolve the loss model    
    if isinstance(loss_fn, str):
        loss_fn = LOSS_LOOKUP[loss_fn][1]
    if multioutput is not None:
        loss_fn = partial(loss_fn, multioutput=multioutput)
    cost_fn = TargetLoss(loss=loss_fn, predictor=features, target=target)
    
    # Append an optional scale function
    if isinstance(scale_fn, str):
        scale_fn = CONVERSION_LOOKUP[scale_fn][1]
    if scale_fn is not None:
        scaled_cost_fn = prx.op.Map(scale_fn, cost_fn)
    else:
        scaled_cost_fn = cost_fn

    # Run the optimizer
    return minimize(scaled_cost_fn, model, frequency, solver, **kwargs)