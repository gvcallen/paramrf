from typing import Callable

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
from pmrf.utils.tree import infer_batch_axes

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
    
    This function uses Bayesian sampling algorithms to represent the full posterior
    distribution of a model as a collection of parameter samples. This is in contrast
    to classical minimization techniques in :mod:`pmrf.optimize`.
    
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
        
    batched_problem, payload, metrics = base_sample(
        loglikelihood_fn=lambda p, _args: p(),
        model=problem,
        solver=solver,
        key=key,
        max_steps=max_steps,
        **kwargs
    )

    # Extract MAP/MLE parameters using the best evaluated function value
    problem_batch_axes = infer_batch_axes(batched_problem, problem)
    best_idx = jnp.argmax(payload.fn_values)

    def extract_best(leaf, axis):
        if axis is None:
            return leaf
        return jnp.take(leaf, best_idx, axis=axis)
    best_problem = jax.tree.map(extract_best, batched_problem, problem_batch_axes)
    
    best_model = best_problem.model
    best_loglikelihood = best_problem.evaluator

    return InferResult(
        best_model=best_model,
        best_loglikelihood=best_loglikelihood,
        sampled_model=batched_problem.model,
        sampled_loglikelihood=batched_problem.evaluator,
        fn_values=payload.fn_values,
        weights=payload.weights,
        metrics=metrics,
    )