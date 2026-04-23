"""
Sampling-based design space exploration.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import parax as prx

from pmrf.core import Model, Frequency, Problem
from pmrf.evaluators import Feature
from pmrf.explore.base import AbstractAdaptiveSampler, ExploreResult
from pmrf.explore.oneshot import AbstractOneShotSampler
from pmrf.utils.module import hypercube_to_physical

def sample_oneshot(
    response: Callable[[Model, Frequency], jnp.ndarray] | str | list,
    model: Model,
    frequency: Frequency,
    solver: AbstractOneShotSampler,
    num_samples: int,
    key: jax.Array,
    **kwargs
) -> ExploreResult:
    """
    Explore the parameter space of a model using a specified oneshot sampling engine.
    
    The parameter space is explored in the hypercube.
    
    Parameters
    ----------
    response : Callable[[Model, Frequency], jnp.ndarray]
        The response function to sample. Can be a function or a callable PyTree,
        or a string to create a feature response using :meth:`pmrf.evaluators.Feature`.
        See :meth:`pmrf.evaluators.Feature` for an easy way to define RF features.    
    model : Model
        The parametric model to sample.
        The parameters bounds/distributions are used to set the sampling space.
    frequency : Frequency | None
        The frequency sweep to evaluate the response at.
    solver : AbstractOneShotSampler
        The sampling algorithm to use.
    num_samples : int
        The number of samples to generate.
    key : jax.Array
        A JAX random key. For example, pass `jax.random.key(0)`.
    **kwargs
        Additional arguments passed to the underlying sampler.

    Returns
    -------
    ExploreResult
        The result object containing the resultant samples.
    """
    # Error checking
    if not isinstance(solver, AbstractOneShotSampler):
        raise Exception(f"Expected a one-shot sampler. Got: {solver}")
    
    # Variable expansion
    if isinstance(response, str):
        response = Feature(response)
    elif isinstance(response, list):
        for r in response:
            if isinstance(r, prx.Module) and r.num_flat_params > 0:
                raise Exception("Cannot pass a list of responses that include parameters.")        
        response = prx.op.Sum([c if isinstance(c, eqx.Module) else prx.op.Lambda(c) for c in response])
    else:
        response = response if isinstance(response, eqx.Module) else prx.op.Lambda(response)
        
    problem = Problem(model=model, frequency=frequency, evaluator=response)
    if problem.num_flat_params == 0:
        raise Exception("Received no free parameters in `pmrf.explore.sample_oneshot`") 
    problem.validate_params()    
    
    if key is None:
        key = generate_key()
        
    params, static = prx.partition(problem)    

    d = model.num_flat_params
    
    def internal_response(params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        problem = eqx.combine(params, static)
        return problem()
        
    def prior_transform_fn(u_problem, _args) -> Problem:
        full_u_problem = eqx.combine(u_problem, static)
        full_physical_problem = hypercube_to_physical(full_u_problem)
        params_physical_problem, _ = prx.partition(full_physical_problem)
        return params_physical_problem    

    key, init_key = jr.split(key)
    state = solver.init(internal_response, d, init_key, options={'num_samples': num_samples})
    
    # 4. Package the array of parameters back into a cleanly batched JAX PyTree
    batched_models = jax.vmap(model.with_params)(thetas)
    
    return ExploreResult(
        model=model, # Leave the original continuous model untouched
        frequency=frequency,
        sampled_models=batched_models, 
        sampled_features=extracted_features,
        history=state.backend_state
    )


def sample_adaptive(
    response: Callable[[Model, Frequency], jnp.ndarray] | str,
    model: Model,
    frequency: Frequency,
    sampler: AbstractAdaptiveSampler,
    max_samples: int | None = None,
    key: jax.Array | None = None,
    **kwargs
) -> ExploreResult:
    """
    Explore the parameter space of a model using a specified adaptive sampling engine.
    
    Parameters
    ----------
    response : Callable[[Model, Frequency], jnp.ndarray]
        The response function to sample. Can be a function or a callable PyTree,
        or a string to create a feature response using :meth:`pmrf.evaluators.Feature`.
        See :meth:`pmrf.evaluators.Feature` for an easy way to define RF features.    
    model : Model
        The parametric model to sample.
        The parameters bounds/distributions are used to set the sampling space..
    frequency : Frequency | None
        The frequency sweep to evaluate the response at.
    sampler : AbstractSampler
        The sampling algorithm to use.
    max_samples : int | None, default=None
        The maximum number of samples to generate. For one-shot samplers, this 
        is the exact number generated. For adaptive samplers, this acts as a 
        computational budget. If None, adaptive samplers run until convergence.
    features : EvaluatorLike | None, default=None
        The specific circuit features to extract.
    key : jax.Array | None, default=None
        JAX PRNG key for stochastic samplers.
    **kwargs
        Additional arguments passed to the underlying sampler.

    Returns
    -------
    ExploreResult
        The comprehensive result object containing the original continuous model 
        and batched execution states.
    """
    if key is None:
        key = jr.PRNGKey(0)
        
    if not isinstance(response, Callable):
        response = Feature(response)

    d = model.num_flat_params
    
    def eval_fn(U_batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Closure to map hypercube proposals to physical params and evaluate."""
        return _evaluate_batch(model, U_batch, frequency, response, **kwargs)

    # 1. Initialize the solver state
    options = {"max_samples": max_samples}
    key, init_key = jr.split(key)
    state = sampler.init(eval_fn, d, init_key, options)
    
    # 2. Standardized Execution Loop
    while not sampler.terminate(state, max_samples):
        key, step_key = jr.split(key)
        state = sampler.step(eval_fn, d, state, step_key, options)

    # 3. Truncate if batching pushed us slightly over the budget
    thetas = state.params
    extracted_features = state.response
    if max_samples is not None and len(thetas) > max_samples:
        thetas = thetas[:max_samples]
        extracted_features = extracted_features[:max_samples]

    # 4. Package the array of parameters back into a cleanly batched JAX PyTree
    batched_models = jax.vmap(model.with_params)(thetas)
    
    return ExploreResult(
        model=model, # Leave the original continuous model untouched
        frequency=frequency,
        sampled_models=batched_models, 
        sampled_features=extracted_features,
        history=state.backend_state
    )


def _evaluate_batch(model: Model, U: jnp.ndarray, frequency: Frequency | None, features: Callable, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Helper to map hypercube proposals to physical params and evaluate features."""
    def eval_single(u):
        flat_params = model.flat_params()
        theta = jnp.array([p.distribution.icdf(u_i) for p, u_i in zip(flat_params, u)])
        
        m_sampled = model.with_params(theta)
        feat_val = features(m_sampled, frequency, **kwargs) if frequency else features(m_sampled, **kwargs)
        return theta, feat_val

    return jax.vmap(eval_single)(U)