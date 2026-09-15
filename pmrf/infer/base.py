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

from pmrf._raw_space import RawSpace
from pmrf.parameters import is_param, param_values, params


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
    
    #: Metrics from the underlying solver.
    metrics: Any = None
    

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
    ) -> SampleResult:
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
        results
            An instance of :class:`pmrf.infer.SampleResult`.
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
    ) -> SampleResult:
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
        results
            An instance of :class:`pmrf.infer.SampleResult`.
        """              
        raise NotImplementedError


class AbstractHypercubeSampler(eqx.Module):
    """
    Interface for samplers operating in a unit hypercube (e.g., classical Nested Sampling).
    
    All inputs (`u0`, `init_cube_samples` etc.) must be in the unit hypercube,
    whereas any outputs (e.g. `samples` in `SampleResults`) must be declared values, as `prior_transform_fn` returns.
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
    ) -> SampleResult:
        """
        Execute the sampling algorithm.

        Parameters
        ----------
        loglikelihood_fn : callable
            A function taking declared parameter values and args as input and returning the log-likelihood.
        prior_transform_fn : callable
            A function taking the hypercube parameters and args as input and returning declared parameter values.
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
        results
            An instance of :class:`pmrf.infer.SampleResult`.
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
    

def run_sampler(
    loglikelihood_fn: Callable[[T, Any], Scalar],
    model: T,
    solver: AbstractSampler,
    key: Array,
    args: Optional[Any] = None,
    init_samples: Optional[T] = None,
    max_steps: Optional[int] = None,
    **kwargs
) -> tuple[T, SampleResult]:
    """
    Samples the free parameters of a model, or any collection of models and parameters,
    using a joint, split, or hypercube Bayesian sampler.

    The solver can be any solver of type :type:`pmrf.infer.AbstractSampler`.

    Joint and split samplers move through the free parameters in raw space, as a
    name-keyed dict from ``prf.param_values(model, free_only=True, space='raw')``,
    and score them with ``prf.log_prior(model, space='raw')``. Hypercube samplers map
    the unit cube through each free parameter's prior to its declared value, so every
    free parameter needs one. Samples are written back with :func:`pmrf.update`, so
    fixed parameters, names, scales and priors are unchanged, and only free
    parameters are batched.

    Parameters
    ----------
    loglikelihood_fn : callable
        The log-likelihood function taking `(unwrapped_model, args)`.
        The prior is added automatically.
    model : PyTree
        The initial model.
    solver : AbstractSampler
        The instantiated sampler to run.
    key : Array
        JAX PRNG key.
    args : Any
        Args to pass to `loglikelihood_fn`.
    init_samples : PyTree, optional
        Optional batched model of initial states, with the same parameter names as
        `model`.
    max_steps: int, optional
        Maximum sampling steps.
    **kwargs
        Runtime arguments forwarded to the solver backend.

    Returns
    -------
    tuple
        A tuple of `(batched_model, sample_results)`.

    Raises
    ------
    ValueError
        If `model` has no free parameters, or a hypercube sampler is given a free
        parameter without a prior.
    """
    if max_steps is not None:
        kwargs['max_steps'] = max_steps

    raw = RawSpace(model, 'sample')

    if isinstance(solver, AbstractJointSampler | AbstractSplitSampler):
        batched_raw = None if init_samples is None else raw.read(init_samples)

        if isinstance(solver, AbstractJointSampler):
            results = solver.run(
                logposterior_fn=raw.log_posterior(loglikelihood_fn),
                y0=raw.y0, args=args, key=key,
                init_samples=batched_raw,
                **kwargs
            )
        else:
            results = solver.run(
                loglikelihood_fn=raw.objective(loglikelihood_fn),
                logprior_fn=raw.log_prior,
                y0=raw.y0, args=args, key=key,
                init_samples=batched_raw,
                **kwargs
            )
        return raw.updated(results.samples), results

    elif isinstance(solver, AbstractHypercubeSampler):
        names = list(raw.y0)
        free = params(model, names)
        missing = [name for name, node in free.items() if not is_param(node) or node.distribution is None]
        if missing:
            raise ValueError(
                "A hypercube sampler needs a prior of its own on every free parameter, but "
                f"these have none: {', '.join(repr(name) for name in missing)}. Give them a "
                "prior with `prf.Random`, or fix them with `prf.update(model, names, fixed=True)`. "
                "A joint prior (`prf.modules.Probabilistic`) has no per-parameter CDF, so use "
                "a joint or split sampler for it."
            )
        # Priors are authored in declared space, so the cube maps to declared values.
        distributions = {name: prx.as_unwrapped(node.distribution) for name, node in free.items()}

        def _to_cube(values: dict) -> dict:
            return {name: d.cdf(values[name]) for name, d in distributions.items()}

        def _cube_to_params(cube: dict, _args: Any) -> dict:
            eps = jnp.finfo(jnp.float32).eps
            return {name: d.icdf(jnp.clip(cube[name], eps, 1.0 - eps)) for name, d in distributions.items()}


        batched_cube = None
        if init_samples is not None:
            batched_cube = _to_cube(param_values(init_samples, names))

        results = solver.run(
            loglikelihood_fn=raw.objective(loglikelihood_fn, space='declared'),
            prior_transform_fn=_cube_to_params,
            u0=_to_cube(param_values(model, names)), args=args, key=key,
            init_cube_samples=batched_cube,
            **kwargs
        )
        return raw.updated(results.samples, space='declared'), results

    else:
        raise TypeError(f"Provided solver {type(solver)} is not a recognized AbstractSampler.")
