from typing import Callable, TypeVar, Union
import dataclasses
import inspect
import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import parax as prx

from pmrf.core import Model, Frequency, Problem
from pmrf.utils.module import hypercube_to_physical, physical_to_hypercube

from pmrf.optimize.base import OptimizeResult
from pmrf.optimize.scipy import ScipyMinimize

JaxoptSolver = Union[TypeVar('JaxoptSolver'), functools.partial]

def minimize(
    objective: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: ScipyMinimize | optx.AbstractMinimiser | JaxoptSolver = ScipyMinimize(),
    use_bounds: bool | None = None,
    search_space: str = "physical",
    icdf_bounds: float = 0.001,
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
    solver : ScipyMinimize | optx.AbstractMinimiser | JaxoptSolver, default=ScipyMinimize()
        The optimizer to use. Can be either an instance of :class:`pmrf.optimize.ScipyMinimize`,
        a minimizer from Optimistix, or a jaxopt solver class (or a functools.partial of a jaxopt solver class).
    use_bounds : bool, optional
        Use bounds from the model for optimization algorithms that support it.
        Note that only individual parameter bounds are used (parameter groups are ignored).
    search_space : str, default="physical"
        The parameter space in which the optimization operates. Can be "physical"
        or "hypercube". If "hypercube", the initial parameter values are transformed to the 
        [0, 1] range and clipped by machine epsilon to avoid edge instabilities.
    icdf_bounds : float | None, default=0.001
        The lower inverse CDF value to use for hypercube-space bounds.
        Only used when `search_space` is "hypercube".
    **kwargs : dict
        Additional options passed to the underlying solver backend.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    if search_space not in ("physical", "hypercube"):
        raise ValueError(f"search_space must be either 'physical' or 'hypercube', got '{search_space}'")

    # Identify if the solver is from jaxopt
    is_jaxopt = False
    if hasattr(solver, 'func') and hasattr(solver.func, '__module__') and 'jaxopt' in solver.func.__module__:
        is_jaxopt = True
    elif inspect.isclass(solver) and hasattr(solver, '__module__') and 'jaxopt' in solver.__module__:
        is_jaxopt = True

    # Error checking for supported solver type
    if not (isinstance(solver, (optx.AbstractMinimiser, ScipyMinimize)) or is_jaxopt):
        raise TypeError(f"Unsupported solver passed to `minimize`. Got type {type(solver)}")
    
    # Input standardization
    use_bounds_explicitly_passed = False
    if use_bounds is None:
        use_bounds = True if (isinstance(solver, ScipyMinimize) or is_jaxopt) else False
    else:
        use_bounds_explicitly_passed = True
        
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
    
    if search_space == "hypercube":
        problem = physical_to_hypercube(problem)
        
        def _clip_to_eps(x: prx.Parameter):
            if not prx.is_free_param(x): return x
            eps = jnp.finfo(x.value.dtype).eps
            return x.with_value(jnp.clip(x.value, eps, 1.0 - eps))
            
        problem = jax.tree.map(_clip_to_eps, problem, is_leaf=prx.is_free_param)
    
    # Setup the parameters and objective
    params, static = prx.partition(problem)
        
    def obj_fn(params, _args=None):
        problem = eqx.combine(params, static)
        if search_space == "hypercube":
            problem = hypercube_to_physical(problem)
        return problem()    

    # Initialize the bounds    
    bounds_tuple = None
    if use_bounds:
        if 'lower' in kwargs and 'upper' in kwargs:
            # Extract explicit bounds if user passed them in kwargs
            lower = kwargs.pop('lower')
            upper = kwargs.pop('upper')
            bounds_tuple = (lower, upper)
        else:
            if search_space == "hypercube":
                def lower_fn(x: prx.Parameter):
                    if not prx.is_free_param(x): 
                        return x
                    return x.with_value(jnp.full_like(x.value, icdf_bounds))
                def upper_fn(x: prx.Parameter):
                    if not prx.is_free_param(x): 
                        return x
                    return x.with_value(jnp.full_like(x.value, 1.0-icdf_bounds))
                
                lower_problem = jax.tree.map(lower_fn, problem, is_leaf=prx.is_free_param)
                upper_problem = jax.tree.map(upper_fn, problem, is_leaf=prx.is_free_param)
                
                # 4. Partition to extract just the bounding parameters for the solver
                (lower, upper), _ = prx.partition((lower_problem, upper_problem))
                bounds_tuple = (lower, upper)
            else:
                # Generate physical bounds from model parameters
                def lower_fn(x: prx.Parameter):
                    if not prx.is_free_param(x): return x
                    if x.bounds is not None:
                        return x.with_value(x.bounds[..., 0])
                    if x.distribution is not None and hasattr(x.distribution, 'icdf'):
                        return x.with_value(x.distribution.icdf(jnp.full_like(x.value, icdf_bounds))) # TODO deprecate
                    return x.with_value(jnp.full_like(x.value, -jnp.inf))
                    
                def upper_fn(x: prx.Parameter):
                    if not prx.is_free_param(x): return x
                    if x.bounds is not None:
                        return x.with_value(x.bounds[..., 1])
                    if x.distribution is not None and hasattr(x.distribution, 'icdf'):
                        return x.with_value(x.distribution.icdf(jnp.full_like(x.value, 1.0-icdf_bounds))) # TODO deprecate
                    return x.with_value(jnp.full_like(x.value, jnp.inf))
                
                lower_tree = jax.tree.map(lower_fn, problem, is_leaf=prx.is_free_param)
                upper_tree = jax.tree.map(upper_fn, problem, is_leaf=prx.is_free_param)
                
                (lower, upper), _ = prx.partition((lower_tree, upper_tree))
                bounds_tuple = (lower, upper)
            
    # Run the solver
    if isinstance(solver, ScipyMinimize):
        if bounds_tuple is not None:
            options = kwargs.get('options', {})
            options.setdefault('lower', bounds_tuple[0])
            options.setdefault('upper', bounds_tuple[1])
            kwargs['options'] = options
        solver_results = solver(obj_fn, params, **kwargs)
    elif isinstance(solver, optx.AbstractMinimiser):
        if bounds_tuple is not None:
             raise Exception("Bounds are not supported for standard `optimistix` solvers.")
        solver_results = optx.minimise(obj_fn, solver, params, **kwargs)
    elif is_jaxopt:
        jaxopt_solver = solver(fun=obj_fn, **kwargs)
        
        # Now that the solver is instantiated, we can safely check its signature
        supports_bounds = False
        if hasattr(jaxopt_solver, "init_state") and "bounds" in inspect.signature(jaxopt_solver.init_state).parameters:
            supports_bounds = True
        elif hasattr(jaxopt_solver, "run") and "bounds" in inspect.signature(jaxopt_solver.run).parameters:
            supports_bounds = True
            
        if not supports_bounds and not use_bounds_explicitly_passed:
            use_bounds = False
            bounds_tuple = None
            
        if bounds_tuple is not None:
            if not supports_bounds:
                raise ValueError(
                    f"The provided jaxopt solver '{jaxopt_solver.__class__.__name__}' "
                    "does not support box constraints. Please use a bounded solver like "
                    "'jaxopt.LBFGSB' or 'jaxopt.ScipyBoundedMinimize', or set `use_bounds=False`."
                )
            opt_step = jaxopt_solver.run(init_params=params, bounds=bounds_tuple)
        else:
            opt_step = jaxopt_solver.run(init_params=params)
            
        solver_results = optx.Solution(
            value=opt_step.params,
            result=optx.RESULTS.successful,
            stats={
                "num_steps": int(getattr(opt_step.state, "iter_num", 0)),
                "loss": float(getattr(opt_step.state, "value", 0.0)),
                "error": float(getattr(opt_step.state, "error", 0.0))
            },
            aux=None,
            state=opt_step.state
        )
        
    # Return the results
    final_params = solver_results.value
    optimized_problem = eqx.combine(final_params, static)
    if search_space == "hypercube":
        optimized_problem = hypercube_to_physical(optimized_problem)
        
    solver_results = dataclasses.replace(solver_results, value=None)
    results = OptimizeResult(
        model=optimized_problem.model,
        objective=optimized_problem.evaluator,
        objective_value=optimized_problem(),
        solver_results=solver_results,
    )
    return results