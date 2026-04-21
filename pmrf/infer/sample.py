import logging
from dataclasses import replace
from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import parax as prx
import inferix as infx

from pmrf.core import Model, Frequency, Problem
from pmrf.infer.result import InferResult
from pmrf.infer.solvers import is_inferer
from pmrf.utils import generate_key


def sample(
    log_likelihood: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: infx.AbstractNestedSampler = infx.PolyChord(),
    *,
    key: jnp.ndarray | None = None,
    **kwargs,
) -> InferResult:
    """
    Samples a given log likelihood function for a model over a frequency range.
    
    The log likelihood function can have its own hyper-parameters, and is returned in `result.log_likelihood`.

    Parameters
    ----------
    log_likelihood : Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
        The log likelihood function to sample. Can be a function or a callable PyTree
        with optional parameters. If a list of log likelihoods is provided,
        they are automatically summed, however the inner likelihoods may not
        have hyperparameters in this case.
    model : Model
        The RF model containing the parameters to be sample.
    frequency : Frequency
        The frequency sweep over which the log likelihood should be evaluated.
    solver : infx.AbstractNestedSampler, default=infx.PolyChord()
        The sampler to use. Can be any sampler from `Inferix <https://github.com/gvcallen/inferix>`_,
        such as :class:`inferix.PolyChord`.
    key : jnp.ndarray, optional
        The random JAX key. Not required for all samplers.
        Automatically generated if not passed.
    **kwargs : dict
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
        raise Exception(f"Expected an Inferix solver. Got: {solver}")
    
    if isinstance(log_likelihood, list):
        for logl in log_likelihood:
            if isinstance(logl, prx.Module) and logl.num_flat_params > 0:
                raise Exception("Cannot pass a list of likelihoods that include parameters.")        
        log_likelihood = prx.op.Sum([c if isinstance(c, eqx.Module) else prx.op.Lambda(c) for c in log_likelihood])
    else:
        log_likelihood = log_likelihood if isinstance(log_likelihood, eqx.Module) else prx.op.Lambda(log_likelihood)
    
    problem = Problem(model=model, frequency=frequency, evaluator=log_likelihood)

    if problem.num_flat_params == 0:
        raise Exception("Received no free parameters in `pmrf.optimize.minimize`") 

    problem.validate_params()    
    
    if solver is None:
        solver = infx.PolyChord()
    if key is None:
        key = generate_key()
        
    params, static = prx.partition(problem)

    def internal_log_likelihood(params, _args) -> jnp.ndarray:
        problem = eqx.combine(params, static)
        return problem()
    
    def prior_transform_fn(u_problem, _args) -> Problem:
        full_u_problem = eqx.combine(u_problem, static)
        
        # 1. Extract unit values and the joint distribution structurally matched
        unit_vals = full_u_problem.grouped_param_values()
        joint_dist = full_u_problem.grouped_distribution()
        
        # 2. Transform from unit hypercube to physical space using the joint inverse CDF
        physical_vals = joint_dist.icdf(unit_vals)
        
        # 3. Unpack the structural dictionary back to flat parameter updates
        named_grouped = full_u_problem.named_grouped_params()
        flat_updates = {}
        
        for group_key, param_dict in named_grouped.items():
            phys_val = physical_vals[group_key]
            
            # Handle scalar/univariate vs stacked/multivariate distributions
            # matching the logic inside your grouped_param_values() method
            if len(param_dict) == 1:
                name = list(param_dict.keys())[0]
                flat_updates[name] = phys_val
            else:
                for i, name in enumerate(param_dict.keys()):
                    flat_updates[name] = phys_val[i]
                    
        # 4. Inject the physical values back into the problem tree
        full_physical_problem = full_u_problem.with_params(flat_updates)
        
        # 5. Repartition to return just the dynamic parameters for Inferix
        params_physical_problem, _ = prx.partition(full_physical_problem)
        return params_physical_problem    
    
    # def prior_transform_fn(u_problem, _args) -> Problem:
    #     full_u_problem = eqx.combine(u_problem, static)
        
    #     def map_param(x):
    #         if isinstance(x, prx.Parameter):
    #             value = jnp.array(x.value, dtype=jnp.float64)
    #             return x.with_value(x.distribution.icdf(value))
    #         return x
    #     full_physical_problem = jax.tree.map(map_param, full_u_problem, is_leaf=prx.is_free_param)
    #     params_physical_problem, static_physical_problem = prx.partition(full_physical_problem)
    #     return params_physical_problem

    if isinstance(solver, infx.AbstractNestedSampler | infx.AbstractHostHypercubeNestedSampler | infx.AbstractHostPhysicalNestedSampler):    
        nested_sampler = True
        infx_result = infx.nested(
            internal_log_likelihood,
            key=key,
            sampler=solver,
            y0=params,
            prior_transform_fn=prior_transform_fn,
            **kwargs
        )
    else:
        nested_sampler = False
        infx_result = infx.mcmc(
            internal_log_likelihood,
            key=key,
            sampler=solver,
            y0=params,
            **kwargs
        )        
    
    # 1. Reconstruct the batched Problem and extract sub-components
    batched_problem = eqx.combine(infx_result.samples, static)
    batched_model = batched_problem.model
    batched_log_likelihoods = batched_problem.evaluator

    # 2. Extract MLE parameters using the log_likelihoods array
    best_idx = jnp.argmax(infx_result.log_likelihoods) 
    mle_problem_params = jax.tree_util.tree_map(lambda x: x[best_idx], infx_result.samples)
    mle_problem = eqx.combine(mle_problem_params, static)
    mle_model: Model = mle_problem.model
    mle_log_likelihood = mle_problem.evaluator

    # 3. Create the flattened Joint Posterior Distribution for the model
    # Parax distributions expect flat arrays, so we must map ravel_pytree across the batch axis
    def flatten_model_params(m):
        flat, _ = jax.flatten_util.ravel_pytree(m)
        return flat
    
    flat_model_samples = jax.vmap(flatten_model_params)(infx_result.samples.model)
    
    if nested_sampler:
        from distreqx.distributions import WeightedEmpirical
        weights = infx_result.weights
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
    log_likelihood_values = infx_result.log_likelihoods
    infx_result = replace(infx_result, samples=None, log_likelihoods=None, weights=None)

    return InferResult(
        model=mle_model,
        log_likelihood=mle_log_likelihood,
        sampled_models=batched_model,
        sampled_log_likelihoods=batched_log_likelihoods,
        log_likelihood_values=log_likelihood_values,
        weights=weights,
        solver_results=infx_result,
    )