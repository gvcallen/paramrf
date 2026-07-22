from typing import Sequence, TypeVar

import jax
import jax.numpy as jnp

from pmrf.models import Model, validate
from pmrf.frequency import Frequency
from pmrf.problems import AbstractProblem, SummedTerms
from pmrf.terms import TermLike, as_terms
from pmrf.infer.base import AbstractSampler, run_sampler
from pmrf.infer.result import InferResult
from pmrf.utils.random import generate_key
from pmrf.utils.tree import batch_axes

ModelT = TypeVar('ModelT', bound=Model)

def sample(
    loglikelihood: TermLike | Sequence[TermLike],
    model: ModelT | None = None,
    frequency: Frequency | None = None,
    solver: AbstractSampler = None,
    *,
    key: jnp.ndarray | None = None,
    max_steps: int | None = None,
    **kwargs,
) -> InferResult[ModelT]:
    """
    Samples a given log likelihood function for a model over a frequency range.
    
    This function uses Bayesian sampling algorithms to represent the full posterior
    distribution of a model as a collection of parameter samples. This is in contrast
    to classical minimization techniques in :mod:`pmrf.optimize`.
    
    The log likelihood function can have its own hyper-parameters, and is returned in `result.loglikelihood`.

    Parameters
    ----------
    loglikelihood : TermLike | Sequence[TermLike] | pmrf.AbstractProblem
        An already-built problem, or the log likelihood function to sample. Can be a function or a callable PyTree
        with optional parameters. If a sequence of log likelihoods is provided,
        they are automatically summed.

        Each may instead be a ``(loglikelihood, frequency)`` pair, or a
        :class:`pmrf.Term`, binding it to its own frequency sweep rather than the
        shared one. This allows a single parameter set to be sampled against
        several datasets on their own grids at once.
    model : Model | None, default=None
        The RF model containing the parameters to be sampled. Omitted when an
        already-built problem is passed.
    frequency : Frequency | None, default=None
        The frequency sweep over which the log likelihood should be evaluated. May be
        omitted only if every log likelihood already carries its own frequency.
    solver : pmrf.infer.AbstractSampler
        The sampler to use (e.g., MCMC, Nested Sampling, etc.).
        See :mod:`pmrf.infer` for available solvers.
    key : jnp.ndarray, optional
        The random JAX key.
        Automatically generated if not passed.
    options : dict
        Additional options passed to the underlying solver backend.
    max_steps : int | None, default=None
        The maximum number of sampling steps to take. Defaults to None which does not pass the argument.
    **kwargs
        Additional runtime arguments forwarded to the solver backend.
    Returns
    -------
    InferResult
        A structured result containing the sampled model and solver statistics.
    """
    if solver is None:
        raise ValueError("A sampler must be passed to `solver`. See `pmrf.infer` for available solvers.")

    if isinstance(loglikelihood, AbstractProblem):
        problem = loglikelihood
    else:
        problem = SummedTerms(model=model, terms=as_terms(loglikelihood, frequency))

    if key is None:
        key = generate_key()
        
    validate(problem)
    
    if max_steps is not None:
        kwargs['max_steps'] = max_steps
        
    batched_problem, results = run_sampler(
        loglikelihood_fn=lambda p, _args: p(),
        model=problem,
        solver=solver,
        key=key,
        **kwargs
    )

    # Extract MAP/MLE parameters using the best evaluated function value
    problem_batch_axes = batch_axes(batched_problem, problem)
    best_idx = jnp.argmax(results.fn_values)

    def extract_best(leaf, axis):
        if axis is None:
            return leaf
        return jnp.take(leaf, best_idx, axis=axis)
    best_problem = jax.tree.map(extract_best, batched_problem, problem_batch_axes)
    
    return InferResult(
        best_problem=best_problem,
        sampled_problem=batched_problem,
        fn_values=results.fn_values,
        weights=results.weights,
        metrics=results.metrics,
    )