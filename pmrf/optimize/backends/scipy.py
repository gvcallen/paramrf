import logging
from typing import Callable
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.optimize import minimize, Bounds, OptimizeResult

from pmrf.models import Model
from pmrf.frequency import Frequency
from optimize.problem import FrequentistProblem

def optimize_scipy(
    model: Model,
    cost: Callable[[Model, Frequency], jnp.ndarray] | list,
    frequency: Frequency,
    *,
    use_jac: bool = True,
    use_hess: bool = False,
    logger: logging.Logger | None = None,
    **kwargs
) -> tuple[Model, OptimizeResult]:
    """
    Executes SciPy minimize with JAX acceleration on an Equinox Model.

    This function automatically bridges the gap between Equinox's PyTree 
    structures and SciPy's flat float64 array requirements. It handles 
    extracting free parameters, JIT-compiling the objective and its 
    derivatives, and rebuilding the optimized model.

    Parameters
    ----------
    model : Model
        The initial ParamRF model containing free parameters to optimize.
    cost_fn : Callable[[Model, Frequency], float | jnp.ndarray] | list
        A custom function evaluating the model over frequency and returning a scalar loss.
        A list of callables can be passed, in which case they are simply summed.
        See the :meth:`pmrf.features.Goal` class for an easy way to define model costs.
    frequency : Frequency | None, optional
        The frequency grid to evaluate goals over.
    use_jac : bool, default=True
        Whether to calculate and pass exact gradients to SciPy via `jax.value_and_grad`.
    use_hess : bool, default=False
        Whether to calculate and pass the exact Hessian matrix to SciPy via `jax.hessian`.
    logger : logging.Logger | None, optional
        Logger for recording optimization start and completion messages.
    **kwargs
        Additional keyword arguments forwarded directly to `scipy.optimize.minimize` 
        (e.g., `method`, `options`, `tol`).

    Returns
    -------
    tuple[Model, OptimizeResult]
        The newly constructed, optimized Model, along with the raw SciPy result object.
        
    Raises
    ------
    ValueError
        If neither `cost_fn` nor `goals` are provided, or if initial parameters 
        fall outside the model's defined distribution bounds.
    """
    logger = logger or logging.getLogger(__name__)

    opt = FrequentistProblem(model, cost, frequency)

    if use_jac:
        jax_val_and_grad = eqx.filter_jit(jax.value_and_grad(cost))
        
        def scipy_objective(x_np: np.ndarray):
            val, grad = jax_val_and_grad(jnp.array(x_np))
            return float(val), np.array(grad, dtype=np.float64)
        kwargs['jac'] = True
    else:
        def scipy_objective(x_np: np.ndarray):
            val = cost(jnp.array(x_np))
            return float(val)
        kwargs['jac'] = False

    if use_hess:
        jax_hessian = eqx.filter_jit(jax.hessian(cost))
        
        def scipy_hessian_fn(x_np: np.ndarray, *args):
            hess = jax_hessian(jnp.array(x_np))
            return np.array(hess, dtype=np.float64)
        kwargs['hess'] = scipy_hessian_fn

    method_name = kwargs.get('method', 'default')
    logger.info(f"Starting SciPy minimize optimization ({method_name})...")
    
    scipy_result = minimize(scipy_objective, opt.x0, bounds=Bounds(opt.bounds[0], opt.bounds[1]), **kwargs)

    logger.info(
        f"Optimization finished: {scipy_result.message} "
        f"(Cost: {scipy_result.fun:.2e}, Iterations: {scipy_result.nfev})"
    )

    optimized_model = opt.reconstruct_fn(jnp.array(scipy_result.x))
    return optimized_model, scipy_result