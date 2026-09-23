"""
Base inference functions and classes.
"""

from typing import Callable, Any, Optional, TypeVar, TypeAlias
import abc

import jax.numpy as jnp
import jax
from jax.scipy.stats import norm
from parax.bijectors import Identity
from parax.distributions import (
    Independent,
    MultivariateNormalDiag,
    MultivariateNormalFullCovariance,
    MultivariateNormalTri,
    Normal,
    Transformed,
)
from jaxtyping import Array, PyTree, Scalar
import equinox as eqx
import parax as prx

from pmrf._solver_view import SolverView
from pmrf.parameters import (
    _joint_blocks, _whitening, is_param, values as _values, params, tree_param_paths, update,
)


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
    name-keyed dict from ``prf.values(model, free_only=True, space='raw')``,
    and score them with ``prf.log_prior(model, space='raw')``. Hypercube samplers map
    the unit cube through each free parameter's prior to its declared value, so every
    free parameter needs one. A prior mapped from raw or physical space by
    :func:`pmrf.prior` goes through its base's inverse CDF and then its map. A joint
    prior's parameters go through the standard normal's inverse CDF to its whitened raw
    values (scaled to its base when that is an independent normal other than the
    standard one), and from there through the whitening and their raw-to-declared
    maps, so the joint prior needs a bijector over an independent normal base, as a
    flow has, or must be a multivariate normal. Samples are written back with
    :func:`pmrf.update`, so fixed parameters, names, scales and priors are unchanged,
    and only free parameters are batched.

    A parameter starting exactly on one of its bounds has an infinite raw value and
    could not move, so a joint or split sampler raises; start it inside its bounds.
    A hypercube sampler works in declared space and accepts it.

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
        If `model` has no free parameters, a joint or split sampler is given a
        parameter starting on a bound, `init_samples` is missing a free parameter, or
        a hypercube sampler is given a free parameter without a prior, a prior with no
        inverse CDF, or a joint prior with no independent normal base.
    """
    if max_steps is not None:
        kwargs['max_steps'] = max_steps

    view = SolverView(model, 'sample')

    if isinstance(solver, AbstractJointSampler | AbstractSplitSampler):
        # These move through raw space, so every starting raw value has to be movable.
        view.check_finite()
        batched_raw = None if init_samples is None else view.read(init_samples)

        if isinstance(solver, AbstractJointSampler):
            results = solver.run(
                logposterior_fn=view.log_posterior(loglikelihood_fn),
                y0=view.y0, args=args, key=key,
                init_samples=batched_raw,
                **kwargs
            )
        else:
            results = solver.run(
                loglikelihood_fn=view.objective(loglikelihood_fn),
                logprior_fn=view.log_prior,
                y0=view.y0, args=args, key=key,
                init_samples=batched_raw,
                **kwargs
            )
        return view.updated(results.samples), results

    elif isinstance(solver, AbstractHypercubeSampler):
        names = list(view.y0)
        free = params(model, names)
        blocks = _cube_blocks(model)
        in_blocks = {name for block_names, _, _ in blocks for name in block_names}
        missing = [
            name for name, node in free.items()
            if name not in in_blocks and (not is_param(node) or node.distribution is None)
        ]
        if missing:
            raise ValueError(
                "A hypercube sampler needs a prior on every free parameter, but "
                f"these have none: {', '.join(repr(name) for name in missing)}. Give them a "
                "prior with `prf.Random` or `prf.prior`, or fix them with "
                "`prf.update(model, names, fixed=True)`."
            )
        # Priors are authored in declared space, so the cube maps to declared values.
        own = {name: _scalar_quantiles(name, node.distribution) for name, node in free.items() if name not in in_blocks}

        def _to_cube(tree) -> dict:
            declared = _values(tree, names)
            cube = {name: cdf(declared[name]) for name, (_, cdf) in own.items()}
            if blocks:
                raw = _values(tree, names, space='raw')
                for block_names, loc, scale in blocks:
                    cube |= {name: norm.cdf((raw[name] - loc[i]) / scale[i]) for i, name in enumerate(block_names)}
            return cube

        def _cube_to_params(cube: dict, _args: Any) -> dict:
            eps = jnp.finfo(jnp.float32).eps
            cube = {name: jnp.clip(u, eps, 1.0 - eps) for name, u in cube.items()}
            declared = {name: icdf(cube[name]) for name, (icdf, _) in own.items()}
            if blocks:
                # A joint prior's block goes from the cube to its whitened raw values, and
                # through the whitening and the old raw-to-declared maps to declared values.
                raw = {
                    name: loc[i] + scale[i] * norm.ppf(cube[name])
                    for block_names, loc, scale in blocks for i, name in enumerate(block_names)
                }
                declared |= _values(update(model, raw, space='raw'), list(raw))
            return declared

        batched_cube = None
        if init_samples is not None:
            batched_cube = _to_cube(init_samples)

        results = solver.run(
            loglikelihood_fn=view.objective(loglikelihood_fn, space='declared'),
            prior_transform_fn=_cube_to_params,
            u0=_to_cube(model), args=args, key=key,
            init_cube_samples=batched_cube,
            **kwargs
        )
        return view.updated(results.samples, space='declared'), results

    else:
        raise TypeError(f"Provided solver {type(solver)} is not a recognized AbstractSampler.")


def _flow_base(distribution) -> tuple[list, Any]:
    """Returns the bijectors of the nested flows `distribution` is, outermost first, and
    the base under them all."""
    bijectors = []
    while isinstance(distribution, Transformed):
        bijectors.append(distribution.bijector)
        distribution = distribution.distribution
    return bijectors, distribution


def _normal_base(distribution) -> tuple[Array, Array] | None:
    """Returns the loc and scale of the independent normal a joint prior's whitened space
    follows, or None when it is not one.

    The whitening of a flow is its bijector after its base's whitening, so the whitened
    space is its base's. A multivariate normal is whitened by its Cholesky factor to a
    standard normal. An independent normal has no whitening, so its whitened space is
    itself, whose loc and scale a trained flow's base moves off the standard normal.
    """
    _, base = _flow_base(distribution)
    n = distribution.event_shape[-1]
    if isinstance(base, MultivariateNormalDiag | MultivariateNormalTri | MultivariateNormalFullCovariance):
        return jnp.zeros(n), jnp.ones(n)
    if isinstance(base, Independent) and isinstance(base.distribution, Normal) and isinstance(_whitening(base), Identity):
        return (jnp.broadcast_to(base.distribution.loc, (n,)), jnp.broadcast_to(base.distribution.scale, (n,)))
    return None


def _cube_blocks(model) -> list[tuple[list[str], Array, Array]]:
    """Returns every joint prior in `model` as its parameters' names, in the order of its
    vector, with the loc and scale of the independent normal its whitened space follows.

    Raises
    ------
    ValueError
        If a joint prior's whitened space is not an independent normal.
    """
    blocks = []
    names = {path: name for name, (path, _) in tree_param_paths(model).items()}
    for joint, paths, _ in _joint_blocks(model):
        block_names = [names[path] for path in paths]
        distribution = prx.as_unwrapped(joint.distribution)
        base = _normal_base(distribution)
        if base is None:
            raise ValueError(
                "A hypercube sampler maps the cube through a joint prior's whitening, which "
                "must start from an independent normal, but the joint prior over "
                f"{', '.join(repr(name) for name in block_names)} is a "
                f"{type(distribution).__name__} with no such base. Give it the structure of a "
                "flow, a bijector over an independent normal base (`Transformed(Independent(Normal(...)), "
                "bijector)`) or a multivariate normal."
            )
        blocks.append((block_names, *base))
    return blocks


def _scalar_quantiles(name: str, distribution) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    """Returns the inverse CDF of a parameter's declared-space prior and its CDF.

    A prior mapped from raw or physical space by `prf.prior` is a flow with no inverse
    CDF of its own; it goes through its base's inverse CDF and then its bijector.

    Raises
    ------
    ValueError
        If the prior, or the base of a mapped one, has no inverse CDF.
    """
    bijectors, base = _flow_base(prx.as_unwrapped(distribution))
    try:
        base.icdf(jnp.asarray(0.5))
    except NotImplementedError:
        if bijectors:
            which = (
                f"is mapped to declared space from another space, as `prf.prior` does with "
                f"space='raw' or space='physical' on a scaled parameter, and its base, a "
                f"{type(base).__name__}, has no inverse CDF"
            )
        else:
            which = f"is a {type(base).__name__}, which has no inverse CDF"
        raise ValueError(
            f"A hypercube sampler maps the cube through each prior's inverse CDF, but the prior "
            f"on {name!r} {which}. Give it a prior whose base has one, or use a joint or split sampler."
        ) from None

    def icdf(u: Array) -> Array:
        x = base.icdf(u)
        for bijector in reversed(bijectors):
            x = bijector.forward(x)
        return x

    def cdf(x: Array) -> Array:
        for bijector in bijectors:
            x = bijector.inverse(x)
        return base.cdf(x)

    return icdf, cdf
