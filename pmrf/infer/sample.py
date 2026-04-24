import logging
from dataclasses import replace
from typing import Callable, Any

import jax
import jax.numpy as jnp
import equinox as eqx
import parax as prx

from pmrf.core import Model, Frequency, Problem
from pmrf.infer.base import is_inferer, InferResult, NestedSamplingResult, AbstractMCMCSampler, AbstractNestedSampler
from pmrf.infer.polychord import PolyChord
from pmrf.utils.random import generate_key

from pmrf.utils.module import hypercube_to_physical, physical_to_hypercube

def sample(
    loglikelihood: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: AbstractMCMCSampler | AbstractNestedSampler = PolyChord(),
    *,
    key: jnp.ndarray | None = None,
    options: dict[str, Any] = None,
) -> InferResult:
    """
    Samples a given log likelihood function for a model over a frequency range.
    
    The log likelihood function can have its own hyper-parameters, and is returned in `result.log_likelihood`.

    Parameters
    ----------
    loglikelihood : Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
        The log likelihood function to sample. Can be a function or a callable PyTree
        with optional parameters. If a list of log likelihoods is provided,
        they are automatically summed, however the inner likelihoods may not
        have hyperparameters in this case.
    model : Model
        The RF model containing the parameters to be sample.
    frequency : Frequency
        The frequency sweep over which the log likelihood should be evaluated.
    solver : AbstractBackendMCMCSampler | AbstractBackendNestedSampler, default=PolyChord()
        The sampler to use. Can be either a MCMC or nested sampler in :mod:`pmrf.infer`.
    key : jnp.ndarray, optional
        The random JAX key.
        Automatically generated if not passed.
    options : dict
        Additional options passed to the underlying solver backend.

    Returns
    -------
    InferResult
        A structured result containing the sampled model and solver statistics.
    """
    try:
        from distreqx.distributions import WeightedEmpirical, Empirical
    except ImportError as e:
        logging.info(
            f"Could not loaded `WeightedEmpirical` or `Empirical` distribution class. "
            "Make sure that your version of distreqx supports Empirical distributions. "
            "You may need a custom fork at https://github.com/gvcallen/distreqx.git"
        )

    if not is_inferer(solver):
        raise Exception(f"Expected an inference solver. Got: {solver}")
    
    if isinstance(loglikelihood, list):
        for logl in loglikelihood:
            if isinstance(logl, prx.Module) and logl.num_flat_params > 0:
                raise Exception("Cannot pass a list of likelihoods that include parameters.")        
        loglikelihood = prx.op.Sum([c if isinstance(c, eqx.Module) else prx.op.Lambda(c) for c in loglikelihood])
    else:
        loglikelihood = loglikelihood if isinstance(loglikelihood, eqx.Module) else prx.op.Lambda(loglikelihood)
    
    problem = Problem(model=model, frequency=frequency, evaluator=loglikelihood)
    if problem.num_flat_params == 0:
        raise Exception("Received no free parameters in `pmrf.infer.sample`") 
    problem.validate_params()    
    
    if key is None:
        key = generate_key()
        
    params, static = prx.partition(problem)

    def internal_log_likelihood(params, _args) -> jnp.ndarray:
        problem = eqx.combine(params, static)
        return problem()
    
    def prior_transform_fn(u_problem, _args) -> Problem:
        full_u_problem = eqx.combine(u_problem, static)
        full_physical_problem = hypercube_to_physical(full_u_problem)
        params_physical_problem, _ = prx.partition(full_physical_problem)
        return params_physical_problem
        
    if isinstance(solver, PolyChord):
        solver_results = solver(internal_log_likelihood, prior_transform_fn, y0=params, init_samples=None, key=key, args=None, options=options)
    else:
        raise ValueError("Got unexpected solver")
    
    # 1. Extract components of the batched problem
    batched_problem = solver_results.samples
    batched_model = batched_problem.model
    batched_loglikelihoods = batched_problem.evaluator

    # 2. Extract MLE parameters using the log_likelihoods array
    best_idx = jnp.argmax(solver_results.loglikelihoods) 
    mle_problem_params = jax.tree_util.tree_map(lambda x: x[best_idx], solver_results.samples)
    mle_problem = eqx.combine(mle_problem_params, static)
    mle_model: Model = mle_problem.model
    mle_loglikelihood = mle_problem.evaluator

    # 3. Create the flattened Joint Posterior Distribution for the model
    # Parax distributions expect flat arrays, so we must map ravel_pytree across the batch axis
    def flatten_model_params(m):
        flat, _ = jax.flatten_util.ravel_pytree(m)
        return flat
    
    flat_model_samples = jax.vmap(flatten_model_params)(solver_results.samples.model)

    if isinstance(solver_results, NestedSamplingResult):
        from distreqx.distributions import WeightedEmpirical
        weights = solver_results.weights
        posterior_dist = WeightedEmpirical(samples=flat_model_samples, weights=weights)
    else:
        from distreqx.distributions import Empirical
        weights = None
        posterior_dist = Empirical(samples=flat_model_samples)

    posterior_group = prx.ParameterGroup(
        param_names=mle_model.flat_param_names(),
        distribution=posterior_dist
    )
    
    mle_model = mle_model.with_param_groups([posterior_group]).with_demoted_param_groups()
    
    # Strip the samples, log_likelihoods and weights so we dont store them twice
    loglikelihood_values = solver_results.loglikelihoods
    solver_results = replace(solver_results, samples=None, log_likelihoods=None, weights=None)

    return InferResult(
        model=mle_model,
        loglikelihood=mle_loglikelihood,
        sampled_models=batched_model,
        sampled_loglikelihoods=batched_loglikelihoods,
        loglikelihood_values=loglikelihood_values,
        weights=weights,
        solver_results=solver_results,
    )