"""
Base inference functions and classes.
"""

from typing import Callable, Any, Optional, TypeVar, TypeAlias
import abc

import jax.numpy as jnp
import jax
from jaxtyping import Array, PyTree, Scalar
import equinox as eqx
import parax as prx


T = TypeVar('T')


class SampleResult(eqx.Module):
    """Lower-level solver result returning from a sampling run."""
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
    ) -> tuple[SampleResult, Any]:
        """
        Execute the sampling algorithm.

        Parameters
        ----------
        logposterior_fn : callable
            A function taking the parameters and args as input and returning the log-posterior probability.
        y0 : PyTree
            The initial parameters, either for shape reference or as a starting point.
        args : Any
            Args to pass to `fn`.
        key : Array
            A random JAX key.
        init_samples : PyTree, optional
            An optional batched PyTree the same structure as `y0` with initial samples to warm-start the algorithm.
        max_steps: int, optional
            The maximum number of sampling steps to take. If None, implies there should be no limit.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        tuple
            A tuple of (:class:`pmrf.infer.SampleResult`, metrics)`.
        """        
        raise NotImplementedError


class AbstractSplitSampler(eqx.Module):
    """Interface for samplers needing separate likelihood and prior densities (e.g., modern Nested Sampling)."""
    @abc.abstractmethod
    def run(
        self,
        loglikelihood_fn: Callable[[PyTree, Any], Any],
        logprior_fn: Callable[[PyTree, Any], Scalar],
        y0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SampleResult, Any]:
        """
        Execute the sampling algorithm.

        Parameters
        ----------
        loglikelihood_fn : callable
            A function taking the parameters and args as input and returning the log-likelihood.
        logprior_fn : callable
            A function taking the parameters and args as input and returning the log prior probability.
        y0 : PyTree
            The initial parameters, either for shape reference or as a starting point.
        args : Any
            Args to pass to `fn`.
        key : Array
            A random JAX key.
        init_samples : PyTree, optional
            An optional batched PyTree the same structure as `y0` with initial samples to warm-start the algorithm.
        max_steps: int, optional
            The maximum number of sampling steps to take. If None, implies there should be no limit.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        tuple
            A tuple of (:class:`pmrf.infer.SampleResult`, metrics)`.
        """              
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
        prior_transform_fn: Callable[[PyTree, Any], PyTree],
        u0: PyTree,
        args: PyTree[Any],
        key: Array,
        init_cube_samples: Optional[PyTree] = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> tuple[SampleResult, Any]:
        """
        Execute the sampling algorithm.

        Parameters
        ----------
        loglikelihood_fn : callable
            A function taking the physical parameters and args as input and returning the log-likelihood.
        prior_transform_fn : callable
            A function taking the hypercube parameters and args as input and returning the physical parameters.
        u0 : PyTree
            The initial parameters in the unit hypercube, either for shape reference or as a starting point.
        args : Any
            Args to pass to `fn`.
        key : Array
            A random JAX key.
        init_cube_samples : PyTree, optional
            An optional batched PyTree the same structure as `u0` with initial hypercube samples to warm-start the algorithm.
        max_steps: int, optional
            The maximum number of sampling steps to take. If None, implies there should be no limit.
        **kwargs
            Runtime arguments forward to the solver backend.

        Returns
        -------
        tuple
            A tuple of (:class:`pmrf.infer.SampleResult`, metrics)`.
        """               
        raise NotImplementedError
    

#: A type-hint for a sampler in :mod:`pmrf.infer`. Either :class:`pmrf.infer.AbstractJointSampler`, :class:`pmrf.infer.AbstractSplitSampler` or :class:`pmrf.infer.AbstractHypercubeSampler`.
AbstractSampler: TypeAlias = AbstractJointSampler | AbstractSplitSampler | AbstractHypercubeSampler


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
    loglikelihood_fn: Callable[[T, Any], Scalar],
    model: T,
    solver: AbstractSampler,
    key: Array,
    args: Optional[Any] = None,
    init_samples: Optional[T] = None,
    max_steps: Optional[int] = None,
    **kwargs
) -> tuple[T, SampleResult, Any]:
    """
    Samples a general PyTree potentially containing Parax probabilistic parameters
    using a joint, split, or hypercube Bayesian sampler.

    The solver can be any solver of type :type:`pmrf.infer.AbstractSampler`.

    Performs Equinox partitioning and Parax unwrrapping/extraction,
    as well as delegation to the relevant solver interface.

    Note that all Parax unwrappables (such a Parax variables)
    MUST be re-wrappable for this interface.

    Parameters
    ----------
    loglikelihood_fn : callable
        The log-likelihood function taking `(unwrapped_y0, args)`.
        Prior calculations are handled automatically via Parax.
    model : PyTree
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
        A tuple of `(batched_model, payload, metrics)`.
    """
    # Filtering/unwrapping
    is_dynamic, is_leaf = prx.probability.is_dynamic, prx.probability.is_leaf
    dynamic, static = eqx.partition(model, is_dynamic, is_leaf=is_leaf)
    params = prx.unwrap(dynamic, only_if=prx.is_probabilistic)

    try:
        eqx.combine(params, static)
    except Exception as e:
        raise Exception(f"Error re-combining params and static. Error: {e}")
    
    batched_params = None
    if init_samples is not None:
        batched_dynamic = eqx.filter(init_samples, is_dynamic, is_leaf=is_leaf)
        # TODO in general we shouldn't assume the user's unwrap is natively broadcastable,
        # though in practice all built-in variables in Parax are
        batched_params = prx.unwrap(batched_dynamic, only_if=prx.is_probabilistic)

    if isinstance(solver, AbstractJointSampler | AbstractSplitSampler):
        # Extraction
        unconstrained_prior = prx.probability.tree_unconstrained_distribution(dynamic)
        bijector_to_constrained = prx.constraints.tree_leafwise_bijector(dynamic)

        # Internal functions
        def _logprior_fn(unconstrained_params: PyTree, _args: Any) -> Scalar:
            return unconstrained_prior.log_prob(unconstrained_params)

        def _loglikelihood_fn(unconstrained_params: PyTree, args: Any) -> Scalar:
            params = bijector_to_constrained.forward(unconstrained_params)
            y_unwrapped = prx.unwrap(eqx.combine(params, static, is_leaf=is_leaf))
            return loglikelihood_fn(y_unwrapped, args)

        def _logposterior_fn(unconstrained_params: PyTree, args: Any) -> Scalar:
            log_prior = _logprior_fn(unconstrained_params, args)
            log_likelihood = _loglikelihood_fn(unconstrained_params, args)
            return log_prior + log_likelihood

        # Space conversions
        unconstrained_params = bijector_to_constrained.inverse(params)
        batched_unconstrained_params = None
        if batched_params is not None:
            batched_unconstrained_params = eqx.filter_vmap(bijector_to_constrained.inverse)(batched_params)

        # Run the sampler
        if isinstance(solver, AbstractJointSampler):
            results, metrics = solver.run(
                logposterior_fn=_logposterior_fn,
                y0=unconstrained_params, args=args, key=key,
                init_samples=batched_unconstrained_params, max_steps=max_steps, **kwargs
            )
        else:
            results, metrics = solver.run(
                loglikelihood_fn=_loglikelihood_fn,
                logprior_fn=_logprior_fn,
                y0=unconstrained_params, args=args, key=key,
                init_samples=batched_unconstrained_params, max_steps=max_steps, **kwargs
            )
        
        # Post-process back to original parameter space and re-wrap
        batched_params_unwrapped = eqx.filter_vmap(bijector_to_constrained.forward)(results.samples)
        batched_params = prx.wrap(dynamic, batched_params_unwrapped, only_if=prx.is_probabilistic)
        return eqx.combine(batched_params, static, is_leaf=is_leaf), results, metrics

    elif isinstance(solver, AbstractHypercubeSampler):
        # Extraction
        distributions = prx.probability.tree_distributions(dynamic)

        # Internal functions
        def _params_to_cube(params):
            return jax.tree.map(lambda d, b: d.cdf(b), distributions, params, is_leaf=prx.is_distribution)

        def _cube_to_params(cube_params: PyTree, _args: Any):
            eps = jnp.finfo(jnp.float32).eps
            safe_cube = jax.tree.map(lambda x: jnp.clip(x, eps, 1.0 - eps), cube_params)
            return jax.tree.map(lambda d, u: d.icdf(u), distributions, safe_cube, is_leaf=prx.is_distribution)
        
        def _loglikelihood_fn(params: PyTree, args: Any):
            unwrapped = prx.unwrap(eqx.combine(params, static, is_leaf=is_leaf))
            return loglikelihood_fn(unwrapped, args)
        
        # Space conversions
        cube_params = _params_to_cube(params)
        batched_cube_params = None
        if batched_params is not None:
            batched_cube_params = eqx.filter_vmap(_params_to_cube)(batched_dynamic)
        
        results, metrics = solver.run(
            loglikelihood_fn=_loglikelihood_fn,
            prior_transform_fn=_cube_to_params,
            u0=cube_params, args=args, key=key,
            init_cube_samples=batched_cube_params, max_steps=max_steps, **kwargs
        )

        batched_params = prx.wrap(dynamic, results.samples, only_if=prx.is_probabilistic)
        return eqx.combine(batched_params, static, is_leaf=is_leaf), results, metrics

    else:
        raise TypeError(f"Provided solver {type(solver)} is not a recognized AbstractSampler.")