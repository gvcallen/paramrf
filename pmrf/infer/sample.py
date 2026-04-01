from dataclasses import replace
from typing import Callable

import jax
import jax.numpy as jnp

import equinox as eqx
import parax as prx
import inferix as infx

from pmrf.core import Model, Frequency, Problem
from pmrf.evaluators import Sum
from pmrf.distributions import Empirical
from pmrf.infer.result import InferResult
from pmrf.utils import generate_key

def is_inferer(x):
    """
    Returns if a solver is suitable for Bayesian inference in :mod:`pmrf.infer`.

    Returns ``True`` for ``infx.AbstractNestedSampler`` and :class:``infx.AbstractHostHypercubeNestedSampler``.
    """    
    return isinstance(x, infx.AbstractNestedSampler | infx.AbstractHostHypercubeNestedSampler)

def sample(
    log_likelihood_fn: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: infx.AbstractNestedSampler = infx.PolyChord(),
    *,
    nlive_factor: int = 25,
    key = None,
    **kwargs,
) -> InferResult:
    """
    Samples a given likelihood function for a model over a frequency range.
    
    The likelihood function can have its own hyper-parameters, and is returned in ``result.likelihood``.

    Parameters
    ----------
    log_likelihood_fn : Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
        The log likelihood function to sample. Must be a callable or PyTree with signature
        (model, freq) -> jnp.ndarray. If a list of costs is provided, they are automatically summed.
    model : Model
        The RF model containing the parameters to be sample.
    frequency : Frequency
        The frequency sweep over which the cost should be evaluated.
    solver : infx.AbstractNestedSampler, default=infx.PolyChord()
        The backend to use. Defaults to ``infx.PolyChord()``.
    transform : distreqx.bijectors.AbstractBijector, default=None
        An invertible transformation to apply to all model parameters before sampling.
    nlive_factor : int, default=25
        The number of live points to use as a factor of the number of parameters.
        Only used for nested sampling.
    key : jnp.ndarray
        The random JAX key.
    **kwargs : dict
        Additional options passed to the underlying solver backend.

    Returns
    -------
    InferResult
        A structured result containing the sampled model and solver statistics.
    """    
    if isinstance(log_likelihood_fn, list):
        log_likelihood_fn = prx.op.Sum([c if isinstance(c, eqx.Module) else prx.op.Lambda(c) for c in log_likelihood_fn])
    else:
        log_likelihood_fn = log_likelihood_fn if isinstance(log_likelihood_fn, eqx.Module) else prx.op.Lambda(log_likelihood_fn)
    
    problem = Problem(model=model, frequency=frequency, evaluator=log_likelihood_fn)
    
    if solver is None:
        solver = infx.PolyChord()
    if key is None:
        key = generate_key()
        
    params, static = prx.partition(problem)

    def internal_log_likelihood_fn(params, _args) -> jnp.ndarray:
        problem = eqx.combine(params, static)
        return problem()
    
    def prior_transform_fn(u_problem, _args) -> Problem:
        full_u_problem = eqx.combine(u_problem, static)
        
        def map_param(x):
            if isinstance(x, prx.Parameter):
                value = jnp.array(x.value, dtype=jnp.float64)
                return x.with_value(x.distribution.icdf(value))
            return x
        full_physical_problem = jax.tree.map(map_param, full_u_problem, is_leaf=prx.is_free_param)
        params_physical_problem, static_physical_problem = prx.partition(full_physical_problem)
        return params_physical_problem

    infx_result = infx.nested(
        internal_log_likelihood_fn,
        key=key,
        sampler=solver,
        y0=params,
        prior_transform_fn=prior_transform_fn,
        nlive=nlive_factor*problem.num_flat_params,
        **kwargs
    )
    
    # 1. Reconstruct the batched Problem and extract sub-components
    batched_problem = eqx.combine(infx_result.samples, static)
    model_samples = batched_problem.model
    likelihood_samples = batched_problem.evaluator

    # 2. Extract MLE parameters using the log_likelihoods array
    best_idx = jnp.argmax(infx_result.log_likelihoods) 
    mle_problem_params = jax.tree_util.tree_map(lambda x: x[best_idx], infx_result.samples)
    mle_problem = eqx.combine(mle_problem_params, static)
    mle_model = mle_problem.model
    mle_likelihood = mle_problem.evaluator

    # 3. Create the flattened Joint Posterior Distribution for the model
    # Parax distributions expect flat arrays, so we must map ravel_pytree across the batch axis
    def flatten_model_params(m):
        flat, _ = jax.flatten_util.ravel_pytree(m)
        return flat
    
    flat_model_samples = jax.vmap(flatten_model_params)(infx_result.samples.model)
    
    # Strip the samples so we dont store them twice
    infx_result = replace(infx_result, samples=None)
    
    posterior_dist = Empirical(samples=flat_model_samples, log_likelihoods=infx_result.log_likelihoods, weights=infx_result.weights)

    # 4. Inject the posterior into the MLE model
    # We clear any independent prior groups and replace them with the global joint posterior
    posterior_group = prx.ParameterGroup(
        param_names=mle_model.flat_param_names(),
        distribution=posterior_dist
    )
    
    mle_model = mle_model.with_param_groups([posterior_group])

    return InferResult(
        model=mle_model,
        likelihood=mle_likelihood,
        sampled_models=model_samples,
        sampled_likelihoods=likelihood_samples,
        log_likelihoods=infx_result.log_likelihoods, 
        weights=infx_result.weights,
        stats=infx_result,
    )