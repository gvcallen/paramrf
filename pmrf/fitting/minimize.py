from typing import Callable

import jax.numpy as jnp
try:
    import skrf
except ImportError:
    pass

import parax as prx
import numpy as np
import distreqx.distributions as dist

from pmrf.core import Model, Frequency
from pmrf.constants import Optimizer
from pmrf.network_collection import NetworkCollection
from pmrf.models import Measured
from pmrf.evaluators import Feature, TargetLoss, MarginalLogLikelihood, NegativeLogLikelihood, NegativeLogPosterior
from pmrf.likelihoods import GaussianLikelihood
from pmrf.losses import RMSELoss

from pmrf.optimize.minimize import minimize
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.fitting.result import FitResult

def fit_minimize(
    model: Model,
    data: np.ndarray | jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: Optimizer = ScipyMinimizer(),
    *,
    features: str | list[str] | Callable = 's',
    inference: str = 'frequentist',
    loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = None,
    likelihood: Callable[[jnp.ndarray], dist.AbstractDistribution] = None,
    noise: prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray] = None,
    discrepancy_model: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None,    
    **kwargs,
) -> FitResult:
    """
    Fits an RF model to measured data using non-linear optimization.

    This high-level function handles data formatting (e.g., extracting arrays 
    from the scikit-rf Networks) and forwards to :func:`pmrf.optimize.minimize`.
    
    Parameters
    ----------
    model : Model
        The RF model to fit.
    data :np.ndarray | jnp.ndarray | skrf.Network | NetworkCollection
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
    inference : str
        The type of inference to use, either 'frequentist' or 'bayesian'.
        See `loss` and `likelihood` for more information.
    loss : str | Callable, optional
        A loss function between the model prediction and the data.
        Can be a function or a callable PyTree with optional parameters.
        Used to internally create a :class:`pmrf.evaluators.TargetLoss` evaluator.
        Mutually exclusive with `likelihood`. If neither `loss` nor `likelihood` is passed,
        :class:`pmrf.losses.RMSELoss` is used for `loss` if `inference` is 'frequentist',
        otherwise :class:`pmrf.likelihoods.GaussianLikelihood` is used for `likelihood`.
        See :mod:`pmrf.losses` for common losses.
    likelihood : str | Callable, optional
        A likelihood model representing the probability of observing the data.
        Can be a function or a callable PyTree with optional parameters.
        Used to internally create a :class:`pmrf.evaluators.NegativeLogLikelihood`
        or :class:`pmrf.evaluators.NegativeLogPosterior` evaluator.
        Mutually exclusive with `loss`. If neither `loss` nor `likelihood` is passed,
        :class:`pmrf.losses.RMSELoss` is used for `loss` if `inference` is 'frequentist',
        otherwise :class:`pmrf.likelihoods.GaussianLikelihood` is used for `likelihood`.
        See :mod:`pmrf.losses` for common losses.
    noise : prx.Parameter | Callable[[jnp.ndarray], jnp.ndarray], optional
        Likelihood noise, either a fixed parameter, or a callable that accepts
        a model prediction (in event space) and returns noise parameters
        for a Gaussian likelihood. Mutually exclusive with `likelihood`.
        For the function case, can be a callable PyTree with optional parameters.
        See :mod:`pmrf.noise_models` for built-in noise models.
        Defaults to `None`, in which case uniform variance from 0.0 to 0.1 is constructed internally.
        Only used if a `likelihood` is passed or `inference` is 'bayesian'.
    discrepancy_model : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray | dist.AbstractDistribution], optional
        A discrepancy model, which caters for the discrepancy between the model and measured data.
        Can either be a function, or a callable PyTree with optional parameters.
        To use a Gaussian process as a discrepancy model,
        see :class:`pmrf.discrepancy_models.GaussianProcess`.
        Only used if a `likelihood` is passed or `inference` is 'bayesian'.
    **kwargs : dict
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    FitResult
        The optimization result containing the fitted Model.
    """
    # Error checking
    if isinstance(data, np.ndarray | jnp.ndarray) and frequency is None:
        raise ValueError("Frequency must be passed if Network data is not provided")
    if loss is not None and likelihood is not None:
        raise ValueError("Only one of either `loss` or `likelihood` can be past to `fit_minimize`")
    if loss is not None and inference == 'bayesian':
        raise Exception("Generalized bayesian inference not yet supported in `fit_minimize`")
    
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

    # Resolve defaults e.g. loss vs MLE vs MAP optimization
    if loss is None and likelihood is None:
        if inference == 'frequentist':
            loss = RMSELoss()
        else:
            if noise is None:
                noise = prx.Uniform(0.0, 0.01)
            likelihood = GaussianLikelihood(noise)
        
    if loss is not None:
        objective_fn = TargetLoss(predictor=features, target=target, loss=loss)
    else:
        mll = MarginalLogLikelihood(predictor=features, observed=target, likelihood=likelihood, discrepancy=discrepancy_model)
        if inference == 'frequentist':
            objective_fn = NegativeLogLikelihood(mll)
        else:
            objective_fn = NegativeLogPosterior(mll)

    # Run the optimizer
    optimize_result = minimize(objective_fn, model, frequency, solver, **kwargs)

    return FitResult(
        data=data,
        frequency=frequency,
        solution=optimize_result,
    )