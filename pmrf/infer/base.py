"""
Base inference functions and classes.
"""

from typing import Callable, Any, Optional, TypeVar
import abc

import jax.numpy as jnp
import jax
from jaxtyping import Array, PyTree, Scalar
import equinox as eqx
import parax as prx


class SampleResults(eqx.Module):
    """The core mathematical payload of a sampling run."""
    #: Stacked array samples
    samples: PyTree[Array]
    
    #: Stacked log-likelihood or log-posterior function values
    fn_values: Array
    
    #: Weights associated with the samples (mostly for Nested/Importance sampling)
    weights: Array | None = None
    
    #: Log of the evidence
    logevidence: Array | None = None
    
    #: Error of the log of the evidence
    logevidence_error: Array | None = None
    

class AbstractJointSampler(eqx.Module):
    """Interface for samplers exploring the joint log-posterior (e.g. MCMC-based NUTS or HMC)."""
    @abc.abstractmethod
    def run(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SampleResults, Any]:
        raise NotImplementedError


class AbstractSplitSampler(eqx.Module):
    """Interface for samplers needing separate likelihood and prior densities (e.g., modern Nested Sampling)."""
    @abc.abstractmethod
    def run(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        logprior_fn: Callable[[PyTree], Scalar],
        y0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SampleResults, Any]:
        raise NotImplementedError


class AbstractHypercubeSampler(eqx.Module):
    """
    Interface for samplers operating in a unit hypercube (e.g., classical Nested Sampling).
    
    All inputs (`u0`, `init_cube_samples` etc.) must be in the unit hypercube,
    whereas any outputs (e.g. `samples` in `SampleResults`) must be in physical space.
    """
    @abc.abstractmethod
    def run(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        prior_transform_fn: Callable[[PyTree], PyTree],
        u0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_cube_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SampleResults, Any]:
        raise NotImplementedError
    
AbstractSampler = AbstractJointSampler | AbstractSplitSampler | AbstractHypercubeSampler


def is_sampler(x):
    """
    Returns if a solver is suitable for Bayesian sampling in :mod:`pmrf.infer.sample`.

    Returns `True` for :class:`pmrf.infer.AbstractSampler`.
    """    
    return isinstance(x, AbstractSampler)
    

def is_inferer(x):
    """
    Returns if a solver is suitable for Bayesian inference in :mod:`pmrf.infer`.

    Returns `True` for :class:`pmrf.infer.AbstractSampler`.
    """    
    return is_sampler(x)
    
T = TypeVar('T')

def sample(
    loglikelihood_fn: Callable[[T, Any], Scalar],
    y0: T,
    solver: AbstractSampler,
    key: Array,
    args: Optional[Any] = None,
    init_samples: Optional[T] = None,
    max_steps: Optional[int] = None,
    **kwargs
) -> tuple[T, T, SampleResults, Any]:
    """
    Samples a general PyTree potentially containing Parax probabilistic parameters
    using a joint, split, or hypercube Bayesian sampler.

    Parameters
    ----------
    loglikelihood_fn : callable
        The log-likelihood function taking `(unwrapped_y0, args)`.
        Prior calculations are handled automatically via Parax.
    y0 : PyTree
        The initial parameter guess / model state.
    args : Any
        Args to pass to `loglikelihood_fn`.
    solver : AbstractSampler
        The instantiated sampler to run.
    key : Array
        JAX PRNG key.
    init_samples : PyTree, optional
        Optional batched PyTree of initial states. 
    max_steps: int, optional
        Maximum sampling steps.
    **kwargs
        Runtime arguments forwarded to the solver backend.

    Returns
    -------
    tuple
        A tuple of `(samples, static, payload, metrics)`.
    """

    if isinstance(solver, AbstractJointSampler | AbstractSplitSampler):
        # Extraction
        init_constrained = prx.unwrap(y0, only_if=prx.is_probabilistic)
        unconstrained_prior_all = prx.probabilistic.tree_unconstrained_distribution(y0)
        bijector_all = prx.probabilistic.tree_leafwise_bijector(y0)

        # Partitioning/filtering
        init_params, static = eqx.partition(init_constrained, eqx.is_inexact_array, is_leaf=prx.is_constant)
        unconstrained_prior = prx.remove(unconstrained_prior_all, prx.is_constant, stop_if=prx.is_distribution)
        bijector = prx.remove(bijector_all, prx.is_constant, stop_if=prx.is_bijector)

        # Space transformation
        def logprior_wrapper(unconstrained_params: PyTree) -> Scalar:
            return unconstrained_prior.log_prob(unconstrained_params)

        def loglikelihood_wrapper(unconstrained_params: PyTree, args: Any) -> Scalar:
            constrained_params = bijector.forward(unconstrained_params)
            y_unwrapped = prx.unwrap(eqx.combine(constrained_params, static))
            return loglikelihood_fn(y_unwrapped, args)

        def logposterior_wrapper(params_unconstrained: PyTree, args: Any) -> Scalar:
            log_prior = logprior_wrapper(params_unconstrained)
            log_likelihood = loglikelihood_wrapper(params_unconstrained, args)
            return log_prior + log_likelihood

        init_unconstrained_params = bijector.inverse(init_params)
        
        init_unconstrained_samples = None
        if init_samples is not None:
            init_sampled_constrained = prx.unwrap(init_samples, only_if=prx.is_probabilistic)
            init_samples_filtered = eqx.filter(init_sampled_constrained, eqx.is_inexact_array, is_leaf=prx.is_constant)
            init_unconstrained_samples = eqx.filter_vmap(bijector.inverse)(init_samples_filtered)

        if isinstance(solver, AbstractJointSampler):
            results, metrics = solver.run(
                logposterior_fn=logposterior_wrapper,
                y0=init_unconstrained_params, args=args, key=key,
                init_samples=init_unconstrained_samples, max_steps=max_steps, **kwargs
            )
        else:
            results, metrics = solver.run(
                loglikelihood_fn=loglikelihood_wrapper,
                logprior_fn=logprior_wrapper,
                y0=init_unconstrained_params, args=args, key=key,
                init_samples=init_unconstrained_samples, max_steps=max_steps, **kwargs
            )
        
        # Post-process back to original parameter space
        sampled_params = eqx.filter_vmap(bijector.forward)(results.samples)
        return sampled_params, static, results, metrics

    elif isinstance(solver, AbstractHypercubeSampler):
        # Extraction
        init_constrained = prx.unwrap(y0, only_if=prx.is_probabilistic)
        distributions_all = prx.probabilistic.tree_distributions(y0)

        # Partitioning/filtering
        init_params, static = eqx.partition(init_constrained, eqx.is_inexact_array, is_leaf=prx.is_constant)
        distributions = prx.remove(distributions_all, prx.is_constant, stop_if=prx.is_distribution)

        # Space transformations
        def params_to_cube(params):
            return jax.tree.map(lambda d, b: d.cdf(b), distributions, params, is_leaf=prx.is_distribution)

        def cube_to_params(cube_params):
            eps = jnp.finfo(jnp.float32).eps
            safe_cube = jax.tree.map(lambda x: jnp.clip(x, eps, 1.0 - eps), cube_params)
            return jax.tree.map(lambda d, u: d.icdf(u), distributions, safe_cube, is_leaf=prx.is_distribution)
        
        init_cube_params = params_to_cube(init_params)
        
        init_cube_samples = None
        if init_samples is not None:
            init_samples_constrained = prx.unwrap(init_samples, only_if=prx.is_probabilistic)
            init_samples_filtered = eqx.filter(init_samples_constrained, eqx.is_inexact_array, is_leaf=prx.is_constant)
            init_cube_samples = eqx.filter_vmap(params_to_cube)(init_samples_filtered)

        # Likelihood wrapper and sampler running
        def loglikelihood_wrapper(params, args):
            unwrapped = prx.unwrap(eqx.combine(params, static))
            return loglikelihood_fn(unwrapped, args)
        
        results, metrics = solver.run(
            loglikelihood_fn=loglikelihood_wrapper,
            prior_transform_fn=cube_to_params,
            u0=init_cube_params, args=args, key=key,
            init_cube_samples=init_cube_samples, max_steps=max_steps, **kwargs
        )

        return results.samples, static, results, metrics

    else:
        raise TypeError(f"Provided solver {type(solver)} is not a recognized AbstractSampler.")