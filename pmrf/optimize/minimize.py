from typing import Callable, TypeVar, Union, Any
import dataclasses
import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import parax as prx

from pmrf.core import Model, Frequency, Problem
from pmrf.utils.module import hypercube_to_physical, physical_to_hypercube, make_bounds

from pmrf.optimize.base import OptimizeResult, AbstractBackendMinimizer, is_minimizer
from pmrf.optimize.scipy import ScipyMinimize

JaxoptSolver = Union[TypeVar('JaxoptSolver'), functools.partial]

def minimize(
    objective: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: optx.AbstractMinimiser | AbstractBackendMinimizer = ScipyMinimize(),
    use_bounds: bool | None = None,
    search_space: str = "physical",
    icdf_bounds: float = 0.001,
    options: dict[str, Any] = None,
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
    options : dict
        Problem-specific runtime options passed to the underlying solver routine.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    if search_space not in ("physical", "hypercube"):
        raise ValueError(f"search_space must be either 'physical' or 'hypercube', got '{search_space}'")

    # Error checking for supported solver type
    if not is_minimizer(solver):
        raise TypeError(f"Unsupported solver passed to `minimize`. Got type {type(solver)}")
    options = options or {}
    
    # Input standardization
    if use_bounds is None:
        use_bounds = True if (isinstance(solver, AbstractBackendMinimizer) and solver.supports_bounds) else False
    else:
        use_bounds = False
        
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
    
    # Initialize the bounds    
    if use_bounds:
        if search_space == "hypercube":
            lower, upper = make_bounds(problem, icdf_bounds, 1.0-icdf_bounds)
        else:
            lower, upper = make_bounds(problem)

        lower, _ = prx.partition(lower)
        upper, _ = prx.partition(upper)

        options.setdefault('lower', lower)
        options.setdefault('upper', upper)

    # Setup the parameters and objective
    params, static = prx.partition(problem)
        
    def obj_fn(params, _args=None):
        problem = eqx.combine(params, static)
        if search_space == "hypercube":
            problem = hypercube_to_physical(problem)
        return problem()            
            
    # Run the solver
    if isinstance(solver, AbstractBackendMinimizer):
        solver_results = solver(obj_fn, params, args=None, options=options)
    elif isinstance(solver, optx.AbstractMinimiser):
        solver_results = optx.minimise(obj_fn, solver, params, args=None, options=None)
    else:
        raise ValueError("Got unexpected solver type")
        
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