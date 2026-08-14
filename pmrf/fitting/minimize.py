from typing import Callable, TypeVar

import jax.numpy as jnp
try:
    import skrf
except ImportError:
    pass

import numpy as np
import distreqx.distributions as dist

from pmrf.modules.base import Module
from pmrf.frequency import Frequency
from pmrf.network_collection import NetworkCollection
from pmrf.evaluators import TargetLoss, MarginalLogLikelihood, GibbsMarginalLogLikelihood, NegativeLogLikelihood
from pmrf.problems import SummedTerms, PriorPenalized
from pmrf.terms import as_terms
from pmrf.fitting.targets import resolve_datasets, union_frequency
from pmrf.likelihoods import GaussianLikelihood
from pmrf.losses import RMSELoss
from pmrf.parameters import Random
from pmrf.distributions import Uniform

from pmrf.optimize.minimize import minimize, AbstractMinimizer
from pmrf.fitting.result import FitResult
from pmrf.parameters import Param

ModuleT = TypeVar('ModuleT', bound=Module)

def fit_minimize(
    model: ModuleT,
    data: np.ndarray | jnp.ndarray | skrf.Network | NetworkCollection,
    frequency: Frequency | None = None,
    solver: AbstractMinimizer | None = None,
    *,
    features: str | list[str] | Callable = 's',
    inference: str = 'frequentist',
    loss: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = None,
    likelihood: Callable[[jnp.ndarray], dist.AbstractDistribution] = None,
    noise: Param | Callable[[jnp.ndarray], jnp.ndarray] = None,
    discrepancy: Callable[[jnp.ndarray, jnp.ndarray], dist.AbstractDistribution] | None = None,    
    temperature: float = None,
    **kwargs,
) -> FitResult[ModuleT]:
    """
    Fits a parameter-aware module to measured data using non-linear optimization.

    This high-level function handles data formatting (e.g., extracting arrays 
    from the scikit-rf Networks) and forwards to :func:`pmrf.optimize.minimize`.
    
    Parameters
    ----------
    model : Module
        The parameter-aware module to fit.
    data :np.ndarray | jnp.ndarray | skrf.Network | NetworkCollection
        The data to fit to. Can either be a JAX array,
        a :class:`skrf.Network`, or a :class:`pmrf.NetworkCollection`.
    frequency : Frequency | None, default=None
        The frequency sweep. Required if `data` is a raw array; otherwise automatically 
        extracted from the Network object.
    solver : AbstractMinimizer, optional
        The optimizer to use.
        See :mod:`pmrf.optimize` for available solvers.
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
        evaluator.
        Mutually exclusive with `loss`. If neither `loss` nor `likelihood` is passed,
        :class:`pmrf.losses.RMSELoss` is used for `loss` if `inference` is 'frequentist',
        otherwise :class:`pmrf.likelihoods.GaussianLikelihood` is used for `likelihood`.
        See :mod:`pmrf.losses` for common losses.
    noise : prf.Param | Callable[[jnp.ndarray], jnp.ndarray], optional
        Likelihood noise (variance), either a fixed parameter, or a callable that accepts
        a model prediction (in event space) and returns noise parameters
        for a Gaussian likelihood. Mutually exclusive with `likelihood`.
        For the function case, can be a callable PyTree with optional parameters.
        See :mod:`pmrf.noise_models` for built-in noise models.
        Defaults to `None`, in which case uniform variance from 0.0 to 0.1 is constructed internally.
        Only allowed if `likelihood` is passed and/or `inference` is 'bayesian'.

        When `data` is a collection, each network is fitted by its own likelihood, so
        each receives its own copy of this noise, free to take a different value and
        carrying its own prior. This suits datasets of differing quality, which is the
        common case for a collection. To share one noise across them, pass a
        `likelihood` built around a single parameter instead.
    discrepancy : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray | dist.AbstractDistribution], optional
        A discrepancy model, which caters for the discrepancy between the model and measured data.
        Can either be a function, or a callable PyTree with optional parameters.
        To use a Gaussian process as a discrepancy model,
        see :class:`pmrf.discrepancy_models.GaussianProcess`.
        Only allowed if `likelihood` is passed and/or `inference` is 'bayesian'.
    temperature : float, optional
        The temperature value for generalized Bayesian optimization.
        Only allowed if `inference` is 'bayesian' and `loss` is not None.
        Defaults to 1.0 internally.
    **kwargs : dict
        Additional keyword arguments passed to :func:`pmrf.optimize.minimize`
        and then underlying solver.

    Returns
    -------
    FitResult
        The optimization result containing the fitted Model.
    """
    # Error checking
    if inference != 'frequentist' and inference != 'bayesian':
        raise ValueError(f"`inference` must be either 'frequentist' or 'bayesian'. Got {inference}")
    if isinstance(data, np.ndarray | jnp.ndarray) and frequency is None:
        raise ValueError("Frequency must be passed if Network data is not provided")
    if loss is not None and likelihood is not None:
        raise ValueError("Only one of either `loss` or `likelihood` can be past to `fit_minimize`")
    if discrepancy is not None and not (likelihood is not None or inference == 'bayesian'):
        raise ValueError("Discrepancy models can only be passed if `likelihood` is passed or `inference` is 'bayesian'`")
    
    # Resolve data and features
    datasets = resolve_datasets(features, data, frequency)

    # Resolve defaults e.g. loss vs MLE vs MAP optimization
    if loss is None and likelihood is None:
        if inference == 'frequentist':
            loss = RMSELoss()
        else:
            if noise is None:
                # Lower bound kept strictly positive: at noise=0 the Gaussian likelihood
                # becomes infinitely peaked, producing huge gradients that can drive other
                # parameters to their own bounds within a couple of solver steps.
                noise = Random(Uniform(1e-6, 0.1))
            likelihood = GaussianLikelihood(noise)

    objective = []
    for dataset in datasets:
        if inference == 'frequentist' and loss is not None:
            evaluator = TargetLoss(predictor=dataset.predictor, target=dataset.target, loss=loss)
        else:
            if likelihood is not None:
                mll = MarginalLogLikelihood(predictor=dataset.predictor, observed=dataset.target, likelihood=likelihood, discrepancy=discrepancy)
            else:
                temperature = temperature if temperature is not None else 1.0
                mll = GibbsMarginalLogLikelihood(predictor=dataset.predictor, observed=dataset.target, loss=loss, discrepancy=discrepancy, temperature=temperature)
            evaluator = NegativeLogLikelihood(mll)
        objective.append((evaluator, dataset.frequency))

    # Run the optimizer
    if solver is not None:
        kwargs['solver'] = solver

    problem = SummedTerms(model=model, terms=as_terms(objective))
    if inference == 'bayesian':
        problem = PriorPenalized(problem)
    optimize_result = minimize(problem, **kwargs)

    return FitResult(
        data=data,
        frequency=union_frequency(datasets),
        solution=optimize_result,
    )
