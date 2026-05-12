import logging
from typing import Callable, Any, Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import eqxpress as ex
import parax as prx

from pmrf.models import Model, validate
from pmrf.frequency import Frequency
from pmrf.problem import Problem
from pmrf.infer.base import AbstractSampler, sample as base_sample
from pmrf.infer.result import InferResult
from pmrf.utils.random import generate_key

def sample(
    loglikelihood: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: AbstractSampler,
    *,
    key: jnp.ndarray | None = None,
    max_steps: int | None = None,
    **kwargs,
) -> InferResult:
    """
    Samples a given log likelihood function for a model over a frequency range.
    
    The log likelihood function can have its own hyper-parameters, and is returned in `result.loglikelihood`.

    Parameters
    ----------
    loglikelihood : Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
        The log likelihood function to sample. Can be a function or a callable PyTree
        with optional parameters. If a list of log likelihoods is provided,
        they are automatically summed.
    model : Model
        The RF model containing the parameters to be sample.
    frequency : Frequency
        The frequency sweep over which the log likelihood should be evaluated.
    solver : pmrf.infer.AbstractSampler
        The sampler to use (e.g., MCMC, Nested Sampling, etc.).
        See :mod:`pmrf.infer` for available solvers.
    key : jnp.ndarray, optional
        The random JAX key.
        Automatically generated if not passed.
    options : dict
        Additional options passed to the underlying solver backend.
    max_steps : int | None, default=None
        The maximum number of sampling steps to take. Defaults to None i.e. no limit.
    **kwargs
        Additional runtime arguments forwarded to the solver backend.
    Returns
    -------
    InferResult
        A structured result containing the sampled model and solver statistics.
    """
    if isinstance(loglikelihood, list):
        loglikelihood = ex.Sum([c if isinstance(c, eqx.Module) else prx.Static(ex.Lambda(c)) for c in loglikelihood])
    else:
        loglikelihood = loglikelihood if isinstance(loglikelihood, eqx.Module) else prx.Static(ex.Lambda(loglikelihood))
    
    problem = Problem(model=model, frequency=frequency, evaluator=loglikelihood)
    
    if key is None:
        key = generate_key()
        
    validate(problem)
        
    sampled_problem, static_problem, payload, metrics = base_sample(
        loglikelihood_fn=lambda p, _args: p(),
        y0=problem,
        solver=solver,
        key=key,
        max_steps=max_steps,
        **kwargs
    )

    # Extract batched components
    sampled_model, static_model = sampled_problem.model, static_problem.model
    sampled_loglikelihood, static_loglikelihood = sampled_problem.evaluator, static_problem.evaluator

    # Extract MAP/MLE parameters using the best evaluated function value
    best_idx = jnp.argmax(payload.fn_values)
    best_sampled_problem = jax.tree.map(lambda x: x[best_idx], sampled_problem)
    best_problem = eqx.combine(best_sampled_problem, static_problem)
    
    best_model = best_problem.model
    best_loglikelihood = best_problem.evaluator

    return InferResult(
        best_model=best_model,
        best_loglikelihood=best_loglikelihood,
        sampled_model=sampled_model,
        static_model=static_model,
        sampled_loglikelihood=sampled_loglikelihood,
        static_loglikelihood=static_loglikelihood,
        fn_values=payload.fn_values,
        weights=payload.weights,
        metrics=metrics,
    )