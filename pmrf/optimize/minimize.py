from typing import Callable, Any, TypeVar, Union
import dataclasses
import inspect
import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import parax as prx
import jaxopt

from pmrf.core import Model, Frequency, Problem
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.scipy import ScipyMinimizer

JaxoptSolver = Union[TypeVar('JaxoptSolver'), functools.partial]

def minimize(
    objective: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: ScipyMinimizer | optx.AbstractMinimiser | JaxoptSolver = ScipyMinimizer(),
    use_bounds: bool | None = None,
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
    frequency : Frequency
        The frequency sweep over which the objective should be evaluated.
    solver : ScipyMinimizer | optx.AbstractMinimiser | JaxoptSolver, default=ScipyMinimizer()
        The optimizer to use. Can be either an instance of :class:`pmrf.optimize.ScipyMinimizer`,
        a minimizer from Optimistix, or a jaxopt solver class (or a functools.partial of a jaxopt solver class).
    use_bounds : bool, optional
        Use bounds from the model for optimization algorithms that support it.
        Note that only individual parameter bounds are used (parameter groups are ignored).
    **kwargs : dict
        Additional options passed to the underlying solver backend.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    # Identify if the solver is from jaxopt
    is_jaxopt = False
    if hasattr(solver, 'func') and hasattr(solver.func, '__module__') and 'jaxopt' in solver.func.__module__:
        is_jaxopt = True
    elif inspect.isclass(solver) and hasattr(solver, '__module__') and 'jaxopt' in solver.__module__:
        is_jaxopt = True

    # Error checking for supported solver type
    if not (isinstance(solver, (optx.AbstractMinimiser, ScipyMinimizer)) or is_jaxopt):
        raise TypeError(f"Unsupported solver passed to `minimize`. Got type {type(solver)}")
    
    # Input standardization
    use_bounds_explicitly_passed = False
    if use_bounds is None:
        use_bounds = True if (isinstance(solver, ScipyMinimizer) or is_jaxopt) else False
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
    
    # Setup the parameters and objective
    params, static = prx.partition(problem)
    def obj_fn(transformed_params, _args=None):
        full_eval_problem = eqx.combine(transformed_params, static)
        return full_eval_problem()    

    # Initialize the bounds    
    bounds_tuple = None
    if use_bounds:
        if 'lower' in kwargs and 'upper' in kwargs:
            # Extract explicit bounds if user passed them in kwargs
            lower = kwargs.pop('lower')
            upper = kwargs.pop('upper')
            bounds_tuple = (lower, upper)
        else:
            # Generate bounds from model parameters
            def lower_fn(x: prx.Parameter):
                if not prx.is_free_param(x):
                    return x
                if x.bounds is not None:
                    return x.with_value(x.bounds[..., 0])
                if x.distribution is not None:
                    eps = jnp.finfo(jnp.result_type(x.value)).resolution
                    return x.with_value(jnp.full_like(x.value, eps))
                return x.with_value(jnp.full_like(x.value, -jnp.inf))
                
            def upper_fn(x: prx.Parameter):
                if not prx.is_free_param(x):
                    return x
                if x.bounds is not None:
                    return x.with_value(x.bounds[..., 1])
                if x.distribution is not None:
                    eps = jnp.finfo(jnp.result_type(x.value)).resolution
                    return x.with_value(jnp.full_like(x.value, 1.0-eps))
                return x.with_value(jnp.full_like(x.value, jnp.inf))
            
            lower_tree = jax.tree.map(lower_fn, problem, is_leaf=prx.is_free_param)
            upper_tree = jax.tree.map(upper_fn, problem, is_leaf=prx.is_free_param)
            
            (lower, upper), _ = prx.partition((lower_tree, upper_tree))
            bounds_tuple = (lower, upper)
        
    # Run the solver
    if isinstance(solver, ScipyMinimizer):
        if bounds_tuple is not None:
            options = kwargs.get('options', {})
            options.setdefault('lower', bounds_tuple[0])
            options.setdefault('upper', bounds_tuple[1])
            kwargs['options'] = options
        solver_results = solver(obj_fn, params, **kwargs)
        # solver_results = optx.minimise(obj_fn, solver, params, **kwargs)
        
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
    optimized_problem = eqx.combine(solver_results.value, static)
    solver_results = dataclasses.replace(solver_results, value=None)
    results = OptimizeResult(
        model=optimized_problem.model,
        objective=optimized_problem.evaluator,
        objective_value=optimized_problem(),
        solver_results=solver_results,
    )
    return results