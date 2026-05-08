"""
Base inference functions and classes.
"""

from typing import Callable, Any, Optional
import abc

import jax.numpy as jnp
import jax
from jaxtyping import Array, PyTree, Scalar
import equinox as eqx
import parax as prx

import parax.probabilistic as prxp
from pmrf.jax_utils import partition


class SamplerPayload(eqx.Module):
    """The core mathematical payload of a sampling run."""
    #: Stacked latent/unscaled arrays (the samples)
    samples: PyTree[Array]
    
    #: Stacked log-likelihoods or log-posteriors
    fn_values: Array
    
    #: Statistical weights (mostly for Nested/Importance sampling)
    weights: Array | None = None
    
    #: Stacked auxiliary data
    auxes: PyTree[Array] | None = None


class AbstractJointSampler(eqx.Module):
    """Interface for samplers exploring the joint log-posterior (e.g. MCMC-based NUTS or HMC)."""
    @abc.abstractmethod
    def sample(
        self,
        logposterior_fn: Callable[[PyTree, Any], Any],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
        raise NotImplementedError


class AbstractSplitSampler(eqx.Module):
    """Interface for samplers needing separate likelihood and prior densities (e.g., modern Nested Sampling)."""
    @abc.abstractmethod
    def sample(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        logprior_fn: Callable[[PyTree], Scalar],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
        raise NotImplementedError


class AbstractHypercubeSampler(eqx.Module):
    """Interface for samplers operating in a unit hypercube (e.g., classical Nested Sampling)."""
    @abc.abstractmethod
    def sample(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        prior_transform_fn: Callable[[PyTree], PyTree],
        y0: PyTree,
        key: Array,
        args: PyTree[Any] = None,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SamplerPayload, PyTree]:
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
    

def sample(
    loglikelihood_fn: Callable[[PyTree, Any], Scalar],
    y0: PyTree,
    solver: AbstractSampler,
    key: Array,
    args: Any = None,
    init_samples: Optional[PyTree] = None,
    max_steps: Optional[int] = None,
    **kwargs
) -> tuple[PyTree, SamplerPayload, PyTree]:
    """
    Samples a general PyTree potentially containing Parax probabilistic parameters
    using a joint, split, or hypercube Bayesian sampler.

    Parameters
    ----------
    loglikelihood_fn : callable
        The log-likelihood function taking `(unwrapped_model, args)`. Prior 
        calculations are handled automatically via Parax.
    y0 : PyTree
        The initial parameter guess / model state.
    solver : AbstractSampler
        The instantiated sampler to run.
    key : Array
        JAX PRNG key.
    args : Any
        Args to pass to `loglikelihood_fn`.
    init_samples : PyTree, optional
        Optional batched PyTree of initial states. 
    max_steps: int, optional
        Maximum sampling steps.
    **kwargs
        Runtime arguments forwarded to the solver backend.

    Returns
    -------
    tuple
        A tuple of `(model_samples, SamplerPayload, metrics)`.
    """
    
    if isinstance(solver, AbstractJointSampler):
        # 1. Unconstrained setup for Joint MCMC Samplers
        unconstrained_prior = prxp.tree_unconstrained_distribution(y0)
        bijector = prxp.tree_leafwise_bijector(y0)
        
        y0_constrained = prx.unwrap(y0, only_if=prx.is_probabilistic)
        y0_unconstrained = bijector.inverse(y0_constrained)
        params, static = partition(y0_unconstrained)
        
        # Prepare batched initial samples if provided
        init_params = None
        if init_samples is not None:
            c_init = eqx.filter_vmap(lambda m: prx.unwrap(m, only_if=prx.is_probabilistic))(init_samples)
            u_init = eqx.filter_vmap(bijector.inverse)(c_init)
            init_params, _ = partition(u_init)

        def logposterior_wrapper(p: PyTree, p_args: Any) -> Scalar:
            unconstrained = eqx.combine(p, static)
            log_prior = unconstrained_prior.log_prob(unconstrained)
            constrained = bijector.forward(unconstrained)
            unwrapped = prx.unwrap(constrained)
            log_likelihood = loglikelihood_fn(unwrapped, p_args)
            return log_prior + log_likelihood

        payload, metrics = solver.sample(
            logposterior_fn=logposterior_wrapper,
            y0=params, key=key, args=args, init_samples=init_params, max_steps=max_steps, **kwargs
        )
        
        # Post-process back to original parameter space
        unconstrained_samples = eqx.filter_vmap(lambda p: eqx.combine(p, static))(payload.samples)
        constrained_samples = eqx.filter_vmap(bijector.forward)(unconstrained_samples)
        
    elif isinstance(solver, AbstractSplitSampler):
        # 2. Constrained setup for Split Samplers
        y0_constrained = prx.unwrap(y0, only_if=prx.is_probabilistic)
        joint_prior = prxp.tree_joint_distribution(y0)
        params, static = partition(y0_constrained)
        
        init_params = None
        if init_samples is not None:
            c_init = eqx.filter_vmap(lambda m: prx.unwrap(m, only_if=prx.is_probabilistic))(init_samples)
            init_params, _ = partition(c_init)

        def loglikelihood_wrapper(p: PyTree, p_args: Any) -> Scalar:
            constrained = eqx.combine(p, static)
            unwrapped = prx.unwrap(constrained)
            return loglikelihood_fn(unwrapped, p_args)

        def logprior_wrapper(p: PyTree) -> Scalar:
            constrained = eqx.combine(p, static)
            return joint_prior.log_prob(constrained)

        payload, metrics = solver.sample(
            loglikelihood_fn=loglikelihood_wrapper,
            logprior_fn=logprior_wrapper,
            y0=params, key=key, args=args, init_samples=init_params, max_steps=max_steps, **kwargs
        )
        
        # Post-process back to original parameter space
        constrained_samples = eqx.filter_vmap(lambda p: eqx.combine(p, static))(payload.samples)

    elif isinstance(solver, AbstractHypercubeSampler):
        # 3. Unit Hypercube setup for Nested/Hypercube Samplers
        y0_constrained = prx.unwrap(y0, only_if=prx.is_probabilistic)
        base_distributions = prxp.tree_distributions(y0)
        
        y0_cube = jax.tree.map(
            lambda d, b: d.cdf(b), 
            base_distributions, y0_constrained, is_leaf=prx.is_distribution
        )
        params, static = partition(y0_cube)
        
        init_params = None
        if init_samples is not None:
            c_init = eqx.filter_vmap(lambda m: prx.unwrap(m, only_if=prx.is_probabilistic))(init_samples)
            cube_init = eqx.filter_vmap(lambda c: jax.tree.map(
                lambda d, b: d.cdf(b), base_distributions, c, is_leaf=prx.is_distribution
            ))(c_init)
            init_params, _ = partition(cube_init)

        def prior_transform_wrapper(p_cube: PyTree) -> PyTree:
            cube_model = eqx.combine(p_cube, static)
            eps = jnp.finfo(jnp.float32).eps
            safe_cube = jax.tree.map(lambda x: jnp.clip(x, eps, 1.0 - eps), cube_model)
            constrained = jax.tree.map(
                lambda d, u: d.icdf(u), 
                base_distributions, safe_cube, is_leaf=prx.is_distribution
            )
            c_params, _ = partition(constrained)
            return c_params

        def loglikelihood_wrapper(c_params: PyTree, p_args: Any) -> Scalar:
            constrained = eqx.combine(c_params, static)
            unwrapped = prx.unwrap(constrained)
            return loglikelihood_fn(unwrapped, p_args)

        payload, metrics = solver.sample(
            loglikelihood_fn=loglikelihood_wrapper,
            prior_transform_fn=prior_transform_wrapper,
            y0=params, key=key, args=args, init_samples=init_params, max_steps=max_steps, **kwargs
        )
        
        # Transform hypercube samples back to constrained space
        c_param_samples = eqx.filter_vmap(prior_transform_wrapper)(payload.samples)
        constrained_samples = eqx.filter_vmap(lambda p: eqx.combine(p, static))(c_param_samples)

    else:
        raise TypeError(f"Provided solver {type(solver)} is not a recognized AbstractSampler.")

    # Reconstruct the final stacked batched models via Parax
    final_model_samples = eqx.filter_vmap(
        lambda s: prx.wrap(y0, s, only_if=prx.is_probabilistic)
    )(constrained_samples)

    return final_model_samples, payload, metrics