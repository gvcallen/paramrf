from typing import Callable, Any, Literal
import dataclasses
import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import parax as prx

from pmrf.core import Model, Frequency, Problem
from pmrf.optimize.result import OptimizeResult
from pmrf.optimize.solvers import ScipyMinimizer
from pmrf.utils import hypercube_to_physical, physical_to_hypercube

def minimize(
    objective: Callable[[Model, Frequency], jnp.ndarray] | list[Callable],
    model: Model,
    frequency: Frequency,
    solver: str | optx.AbstractMinimiser | Callable[[Callable, jnp.ndarray, Any], optx.Solution] = ScipyMinimizer(),
    search_space: Literal["physical", "hypercube"] = "physical",
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
    solver : optx.AbstractMinimiser | Callable[[Callable, jnp.ndarray, Any], optx.Solution], default=ScipyMinimizer()
        The optimizer to use. Can be either an instance of :class:`pmrf.optimize.ScipyMinimizer`
        or a minimizer from `Optimistix <https://docs.kidger.site/optimistix/api/minimise>`_
        (such as :class:`optimistix.LBFGS`). If a string is passed, a ScipyMinimizer is created
        with that method.
    search_space : Literal["physical", "hypercube"], default="physical"
        The domain over which the solver operates. "physical" optimizes within the native parameter bounds,
        while "hypercube" maps all free parameters to a [0, 1] during optimization based on their distributions.
    **kwargs : dict
        Additional options passed to the underlying solver backend.

    Returns
    -------
    OptimizeResult
        A structured result containing the fitted model and solver statistics.
    """
    if search_space not in ("physical", "hypercube"):
        raise ValueError(f"search_space must be 'physical' or 'hypercube', got: {search_space}")    
    
    if isinstance(objective, list):
        for obj in objective:
            if isinstance(obj, prx.Module) and obj.num_flat_params > 0:
                raise Exception("Cannot pass a list of objectives that include parameters.")
        objective = prx.op.Sum([c if isinstance(c, eqx.Module) else prx.op.Lambda(c) for c in objective])
    else:
        objective = objective if isinstance(objective, eqx.Module) else prx.op.Lambda(objective)
    
    problem = Problem(model=model, frequency=frequency, evaluator=objective)   
    if problem.num_flat_params == 0:
        raise Exception("Received no free parameters in `pmrf.optimize.minimize`") 
    
    problem.validate_params()
    
    if search_space == "hypercube":
        problem = physical_to_hypercube(problem)    
    
    params, static = prx.partition(problem)
    
    def obj_fn(transformed_params, _args):
        full_eval_problem = eqx.combine(transformed_params, static)
        
        if search_space == "hypercube":
            full_eval_problem = hypercube_to_physical(full_eval_problem)
            
        return full_eval_problem()
    
    if isinstance(solver, ScipyMinimizer):
        if search_space == "hypercube":
            def lower_fn(x):
                if isinstance(x, prx.Parameter):
                    if x.distribution is not None:
                        eps = jnp.finfo(jnp.result_type(x.value)).resolution
                        return x.with_value(jnp.full_like(x.value, eps))
                    
                    if x.bounds is not None:
                        return x.with_value(x.bounds[..., 0])
                    return x.with_value(jnp.full_like(x.value, -jnp.inf))
                return x
                
            def upper_fn(x):
                if isinstance(x, prx.Parameter):
                    if x.distribution is not None:
                        eps = jnp.finfo(jnp.result_type(x.value)).resolution
                        return x.with_value(jnp.full_like(x.value, 1.0 - eps))
                    
                    if x.bounds is not None:
                        return x.with_value(x.bounds[..., 1])
                    return x.with_value(jnp.full_like(x.value, jnp.inf))
                return x
            
            lower_tree = jax.tree.map(lower_fn, problem, is_leaf=prx.is_free_param)
            upper_tree = jax.tree.map(upper_fn, problem, is_leaf=prx.is_free_param)
            
            (lower, upper), _ = prx.partition((lower_tree, upper_tree))
            kwargs.setdefault('bounds', (lower, upper))
        else:
            if 'bounds' in kwargs:
                lower_tree, upper_tree = kwargs.pop('bounds')
            else:
                groups = problem.param_groups(include_fixed=False)
                flat_params = problem.named_flat_params(include_fixed=False)
                flat_vals = problem.named_flat_param_values(include_fixed=False)
                
                lower_vals = {}
                upper_vals = {}
                
                for group in groups:
                    arrays = [flat_vals[name] for name in group.param_names]
                    x_val = jnp.stack(arrays)
                    if len(arrays) == 1:
                        x_val = jnp.squeeze(x_val, axis=0)
                    
                    # Compute bounds at the group level
                    if group.distribution is not None and hasattr(group.distribution, 'icdf'):
                        low = group.distribution.icdf(0.01 * jnp.ones_like(x_val))
                        high = group.distribution.icdf(0.99 * jnp.ones_like(x_val))
                    else:
                        low_list = []
                        high_list = []
                        for name in group.param_names:
                            p = flat_params[name]
                            if p.bounds is not None:
                                low_list.append(p.bounds[..., 0])
                                high_list.append(p.bounds[..., 1])
                            else:
                                low_list.append(jnp.full_like(flat_vals[name], -jnp.inf))
                                high_list.append(jnp.full_like(flat_vals[name], jnp.inf))
                        
                        low = jnp.stack(low_list) if len(low_list) > 1 else low_list[0]
                        high = jnp.stack(high_list) if len(high_list) > 1 else high_list[0]
                        
                    # Unpack bounds back into the flat dictionary
                    if len(arrays) == 1:
                        lower_vals[group.param_names[0]] = low
                        upper_vals[group.param_names[0]] = high
                    else:
                        for i, name in enumerate(group.param_names):
                            lower_vals[name] = low[i]
                            upper_vals[name] = high[i]
                
                # Apply the bulk updates to generate the bounded modules
                lower_problem = problem.with_params(lower_vals)
                upper_problem = problem.with_params(upper_vals)
                
                # Strip out static attributes using partition
                (lower, upper), _ = prx.partition((lower_problem, upper_problem))
                kwargs.setdefault('bounds', (lower, upper))
            
        if kwargs.get('has_aux', False):
            raise Exception("Auxiliary data not supported for host solvers")
            
        solver_results = solver(obj_fn, params, args=None, **kwargs)
    else:
        solver_results = optx.minimise(obj_fn, solver, params, **kwargs)
        
    optimized_problem = eqx.combine(solver_results.value, static)
    
    if search_space == "hypercube":
        optimized_problem = hypercube_to_physical(optimized_problem)
    
    solver_results = dataclasses.replace(solver_results, value=None)
    results = OptimizeResult(
        model=optimized_problem.model,
        objective=optimized_problem.evaluator,
        objective_value=optimized_problem(),
        solver_results=solver_results,
    )
    
    if isinstance(solver, optx.AbstractMinimiser):
        logging.info(f"Final objective value = {results.objective_value}")
    
    return results