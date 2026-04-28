from typing import Callable, TypeVar, Union, Any, Literal
import functools
import dataclasses

from jaxtyping import Array
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import parax as prx

from pmrf.core import Model, Frequency, Problem

from pmrf.optimize.base import OptimizeResult, AbstractCallableMinimizer, is_minimizer
from pmrf.optimize.scipy import ScipyMinimize

JaxoptSolver = Union[TypeVar('JaxoptSolver'), functools.partial]

def minimize(
    objective: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: optx.AbstractMinimiser | AbstractCallableMinimizer = ScipyMinimize(),
    search_space: Literal['latent', 'hypercube'] = 'latent',
    use_bounds: bool = True,
    icdf_bounds: float = 0.001,
    options: dict[str, Any] = None,
    max_steps: int | None = 1024,
    **kwargs,
) -> OptimizeResult:
    """
    Minimizes a given objective function for a model over a frequency range.
    
    The objective function can have its own hyper-parameters, and is returned in `result.objective`.

    Parameters
    ----------
    objective : Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
        The objective function to minimize. Can be a function or a callable PyTree
        with optional parameters. If a list of objectives is provided,
        they are automatically summed, however the objectives may not have hyper-parameters
        in this case.
        See :meth:`pmrf.evaluators.Goal` for an easy way to define goal-based objectives.
    model : Model
        The RF model containing the parameters to be optimized.
        If the parameters contain bounds, they are used in a bounded optimization unless `use_bounds` is False.
        If the parameters do not contain bounds, their limits are set to infinity.
    frequency : Frequency
        The frequency sweep over which the objective should be evaluated.
    solver : optx.AbstractMinimiser | AbstractBackendMinimizer, default=ScipyMinimize()
        The optimizer to use. Can be either an instance of :class:`pmrf.optimize.ScipyMinimize`,
        a minimizer from Optimistix, or a jaxopt solver class (or a functools.partial of a jaxopt solver class).
    search_space : str, default='physical'
        The parameter space in which the optimization operates.
        - 'latent': The latent space. If parameter bounds are specifed and the parameter
        has no transform, a scaled sigmoid transform is used internally.
        - 'hypercube': The unit hypercube. Requires all parameters to have probability distributions
        with an `icdf` implementation. In this case, icdf bounds can be specified using `icdf_bounds`.
    use_bounds : bool, optional
        Use bounds specified in parameters. If the solver supports bounded optimization, this is
        used directly. Otherwise, either for bounded optimization algorithms
        If True, all parameters must either have no latent transform or a bijective transform.
        Not compatible with `search_space='hypercube'`.
    icdf_bounds : float | None, default=0.001
        The lower inverse CDF value to use for hypercube-space bounds.
        Only used when `search_space` is "hypercube".
    options : dict 
        Problem-specific runtime options passed to the underlying solver routine.
    max_steps : int
        The maximum number of iterations to take. 
    **kwargs
        Additional arguments forwarded to the solver backend.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    if isinstance(objective, list):
        for obj in objective:
            if isinstance(obj, prx.Module) and obj.num_flat_params > 0:
                raise Exception("Cannot pass a list of objectives that include parameters.")
        objective = prx.op.Sum([c if isinstance(c, eqx.Module) else prx.op.Lambda(c) for c in objective])
    else:
        objective = objective if isinstance(objective, eqx.Module) else prx.op.Lambda(objective)
    
    # Create and validate the problem
    problem = Problem(model=model, frequency=frequency, evaluator=objective)   
    if problem.num_flat_params == 0:
        raise Exception("Received no free parameters in `pmrf.optimize.minimize`") 
    problem.validate_params()
    
    # Run the optimization
    optimized_problem, solver_results = minimize_parax(
        lambda p, _args: p(),
        problem,
        solver=solver,
        supports_bounded=solver.supports_bounds if isinstance(solver, AbstractCallableMinimizer) else False,
        search_space=search_space,
        use_bounds=use_bounds,
        icdf_bounds=icdf_bounds,
        options=options,
        max_steps=max_steps,
        **kwargs,
    )
    
    # Return the results
    results = OptimizeResult(
        model=optimized_problem.model,
        objective=optimized_problem.evaluator,
        objective_value=optimized_problem(),
        solver_results=solver_results,
    )
    return results


def minimize_parax(
    fn: Callable[[eqx.Module], Array],
    module: eqx.Module,
    solver: optx.AbstractMinimiser | Callable,
    search_space: Literal['latent', 'hypercube'] = 'latent',
    supports_bounded: bool = False,
    use_bounds: bool = True,
    icdf_bounds: float = 0.001,
    options: dict[str, Any] = None,
    max_steps: int | None = 1024,
    filter_spec: Any = None,
    **kwargs,
) -> tuple[eqx.Module, Any]:
    """
    Minimize an Equinox module containing Parax parameters.

    This wrapper automatically handles parameter partitioning, boundary enforcement, 
    and spatial transformations (e.g., mapping to the unit hypercube) before 
    passing the purified arrays to the objective function.

    Parameters
    ----------
    fn : Callable[[eqx.Module], Array]
        The objective function to minimize. It should accept a fully unwrapped 
        version of the module (containing pure jax.Arrays) and return a scalar loss.
    module : eqx.Module
        The Equinox module containing `parax.Parameter` objects to optimize.
    solver : optimistix.AbstractMinimiser or Callable
        The optimizer to use. Can be an optimistix class or a compatible callable.
    search_space : {'latent', 'hypercube'}, default='latent'
        The domain in which the optimizer operates. 'latent' optimizes in the 
        unconstrained bijector space. 'hypercube' maps bounded variables to [0, 1].
    supports_bounded : bool, default=False
        Set to True if the underlying solver natively accepts 'lower' and 'upper' 
        bounds in its options (e.g., L-BFGS-B). If False, bounds are enforced 
        via auto-generated bijector transforms.
    use_bounds : bool, default True
        Whether to enforce parameter bounds during optimization.
    icdf_bounds : float, default=0.001
        The epsilon margin used to prevent numerical overflow at the edges of 
        distributions when operating in the hypercube (restricts domain to 
        [eps, 1.0 - eps]).
    options : dict, optional
        Additional options passed to the solver.
    max_steps : int or None, default=1024
        The maximum number of optimization steps.
    filter_spec : Any, optional
        An Equinox filter specification dictating which parameters to optimize. 
        If None, defaults to all non-fixed free parameters.
    **kwargs
        Additional keyword arguments passed directly to the solver.

    Returns
    -------
    tuple of (eqx.Module, Any)
        A tuple containing the optimized module (with parameters mapped back to 
        their physical space) and the raw solver results object.
    """
    # Error checking
    if search_space not in ('latent', 'hypercube'):
        raise ValueError(f"search_space must be either 'latent' or 'hypercube', got '{search_space}'")
    if not (isinstance(solver, optx.AbstractMinimiser) or callable(solver)):
        raise TypeError(f"Unsupported solver passed to `minimize`. Got type {type(solver)}")
        
    options = options or {}

    # Defaults and standardization
    if filter_spec is None:
        filter_spec = prx.where_free_param_value(module)

    if search_space == 'hypercube':
        module = prx.physical_to_hypercube(module)
        
        def _clip_to_eps(x: prx.Parameter):
            if not prx.is_free_param(x): 
                return x
            eps = jnp.finfo(x.latent_value.dtype).eps
            # Clip the hypercube state directly, avoiding physical inversion
            clipped_u = jnp.clip(x.latent_value, eps, 1.0 - eps)
            return dataclasses.replace(x, latent_value=clipped_u)
            
        module = jax.tree.map(_clip_to_eps, module, is_leaf=prx.is_free_param)
    
    # Initialize the bounds
    if use_bounds:
        if supports_bounded:
            if search_space == 'hypercube':
                lower, upper = prx.make_bounds(module, icdf_bounds, 1.0 - icdf_bounds)
            else:
                lower, upper = prx.make_bounds(module)

            lower, _ = eqx.partition(lower, filter_spec=filter_spec)
            upper, _ = eqx.partition(upper, filter_spec=filter_spec)

            options.setdefault('lower', lower)
            options.setdefault('upper', upper)
        else:
            module = prx.enforce_bounds(module, search_space, icdf_bounds)

    # Setup the parameters and objective
    params, static = eqx.partition(module, filter_spec=filter_spec)
        
    def obj_fn(params, _args=None):
        mod = eqx.combine(params, static)
        
        # 1. Map from computational space back to physical wrappers
        if search_space == 'hypercube':
            mod = prx.hypercube_to_physical(mod)
            
        # 2. Unwrap Parameters to pure jax.Arrays for safe and fast objective evaluation
        mod = prx.unwrap(mod) 
        
        return fn(mod)
            
    # Run the solver
    if isinstance(solver, optx.AbstractMinimiser):
        solver_results = optx.minimise(obj_fn, solver, params, args=None, options=options, max_steps=max_steps, **kwargs)
    else:
        solver_results = solver(obj_fn, params, args=None, options=options, max_steps=max_steps, **kwargs)
    
    # Reconstruct final module
    final_params = solver_results.value
    optimized_mod = eqx.combine(final_params, static)
    
    if search_space == 'hypercube':
        optimized_mod = prx.hypercube_to_physical(optimized_mod)

    return optimized_mod, solver_results