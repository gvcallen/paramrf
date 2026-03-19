import jax
import jax.numpy as jnp

import equinox as eqx
import parax as prx
import inferix as infx

from pmrf.core import Model, Frequency, Evaluator, Problem
from pmrf.infer.result import InferResult
from pmrf.utils import generate_key

def sample(
    model: Model,
    frequency: Frequency,
    likelihood: Evaluator,
    sampler: infx.AbstractSampler = None,
    *,
    nlive_factor: int = 25,
    key = None,
    **kwargs,
) -> InferResult:
    problem = Problem(model, frequency, likelihood)
    
    if sampler is None:
        nlive = nlive_factor * problem.num_flat_params
        sampler = infx.PolyChord(nlive)
    if key is None:
        key = generate_key()
        
    params, static = prx.partition(problem)

    def log_likelihood_fn(params, _args) -> jnp.ndarray:
        problem = eqx.combine(params, static)
        return problem()
    
    hypercube_transform = prx.transforms.HypercubeTransform()
    def prior_transform_fn(u_problem, _args) -> Problem:
        return jax.tree.map(hypercube_transform.inv, u_problem, prx.is_valid_param)

    if isinstance(sampler, infx.AbstractNestedSampler):
        ndims = problem.num_flat_params
        infx_result = infx.nested_sample(
            log_likelihood_fn,
            key=key,
            sampler=sampler,
            y0=params,
            prior_transform_fn=prior_transform_fn,
            nlive=nlive_factor*ndims,
            ndims=ndims,
            **kwargs
        )
    else:
        raise Exception("Only nested samplers are currently support in pmrf.infer.sample")

    model, likelihood = infx_result.samples.model, infx_result.samples.likelihood

    return InferResult(
        model=model,
        likelihood=likelihood,
        value=infx_result.samples.logZ,
        history=infx_result.stats,
        success=infx_result.result == infx.RESULTS.successful,
    )