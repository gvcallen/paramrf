import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize, Bounds, OptimizeResult
from tqdm.auto import tqdm
from pmrf.models.model import Model

def run_scipy_minimize(
    model: Model,
    cost_fn: callable,
    logger,
    *,
    use_jac=True,
    use_hess=False,
    show_progress=True, 
    debug_save_loss=False,
    loss_file_path="loss_log.csv",
    **kwargs
) -> tuple[Model, OptimizeResult]:
    """
    Shared backend function for executing SciPy minimize with JAX acceleration.
    """
    # 1. Parameter Initialization & Physical Bounds
    dist = model.distribution()
    minimums = np.array(dist.min, dtype=np.float64)
    maximums = np.array(dist.max, dtype=np.float64)
    
    is_bounded = np.isfinite(minimums) & np.isfinite(maximums)
    
    scales = np.ones_like(minimums)
    shifts = np.zeros_like(minimums)
    
    scales[is_bounded] = maximums[is_bounded] - minimums[is_bounded]
    shifts[is_bounded] = minimums[is_bounded]
    
    if np.any(scales <= 0):
        raise ValueError("Maximum bounds must be strictly greater than minimum bounds for normalization.")

    x0_physical = np.array(model.flat_param_values(), dtype=np.float64)

    too_low, too_high = x0_physical < minimums, x0_physical > maximums
    if np.any(too_low | too_high):
        param_names = model.flat_param_names()
        bad_params = [
            f"  {name}: x0={val}, min={minv}, max={maxv} ({'below min' if low else 'above max'})"
            for name, val, minv, maxv, low, high in zip(param_names, x0_physical, minimums, maximums, too_low, too_high)
            if low or high
        ]
        raise ValueError(f"Initial parameters outside bounds:\n" + "\n".join(bad_params))
    
    # 2. Normalize Inputs & Set Normalized Bounds
    z0 = (x0_physical - shifts) / scales
    norm_mins = (minimums - shifts) / scales
    norm_maxs = (maximums - shifts) / scales
    normalized_bounds = Bounds(norm_mins, norm_maxs)

    # 3. Setup Options
    method_name = kwargs.get('method', 'default')
    logger.info(f"Starting SciPy minimize optimization ({method_name})")

    jnp_shifts = jnp.array(shifts)
    jnp_scales = jnp.array(scales)
    
    def unnormalize_fn(z):
        return z * jnp_scales + jnp_shifts

    # Tracking variables
    loss_history = []
    time_history = []
    t0 = [0.0] 

    # 4. Define Objective, Gradient, and Hessian
    if use_jac or use_hess:
        @jax.jit
        def normalized_cost(z):
            x_physical = unnormalize_fn(z)
            return cost_fn(x_physical)

    if use_hess:
        h_fn = jax.jit(jax.hessian(normalized_cost))
        if debug_save_loss:
            _ = h_fn(z0)
            
        def hessian_wrapper(z, *args):
            return np.array(h_fn(z), dtype=np.float64)
        kwargs['hess'] = hessian_wrapper

    if use_jac:
        vg_fn = jax.value_and_grad(normalized_cost)
        if debug_save_loss:
            _ = vg_fn(z0)
        
        def objective(z, pbar):
            val, grad = vg_fn(z)
            val_np = float(val)
            grad_np = np.array(grad, dtype=np.float64)

            if debug_save_loss:
                loss_history.append(val_np)
                time_history.append(time.perf_counter() - t0[0])

            if np.isnan(val_np) or np.isinf(val_np):
                logger.warning(f"Bad value encountered at normalized z = {z}")
            if np.any(np.isnan(grad_np)) or np.any(np.isinf(grad_np)):
                logger.warning(f"Bad gradient encountered at normalized z = {z}")
            
            pbar.update(1)
            pbar.set_postfix({'cost': f"{val_np:.4f}"})
            return val_np, grad_np
        kwargs['jac'] = True 
    else:
        if debug_save_loss:
            x_physical_warmup = np.array(unnormalize_fn(z0), dtype=np.float64)
            _ = cost_fn(x_physical_warmup)
            
        def objective(z, pbar):
            x_physical = np.array(unnormalize_fn(z), dtype=np.float64)
            val = float(cost_fn(x_physical))
            
            if debug_save_loss:
                loss_history.append(val)
                time_history.append(time.perf_counter() - t0[0])

            pbar.update(1)
            pbar.set_postfix({'cost': f"{val:.4f}"})
            return val
        kwargs['jac'] = False
        
    # 5. Optimization Loop
    t0[0] = time.perf_counter() 
    
    with tqdm(desc="Optimizing", unit=" eval", disable=not show_progress) as pbar:
        scipy_result = minimize(
            objective, z0, args=(pbar,), bounds=normalized_bounds, **kwargs
        )
        pbar.set_postfix({'cost': f"{scipy_result.fun:.4f}"})

    logger.info(
        f"Optimization finished: {scipy_result.message} "
        f"(Cost: {scipy_result.fun:.2f}, nfev: {scipy_result.nfev})"
    )
    
    if debug_save_loss and loss_history:
        try:
            data_to_save = np.column_stack((time_history, loss_history))
            np.savetxt(loss_file_path, data_to_save, header="time_seconds,loss", comments="", delimiter=",")
        except Exception as e:
            logger.error(f"Failed to save loss history. Error: {e}")

    # 6. Un-normalize and Return
    x_opt_physical = np.array(unnormalize_fn(scipy_result.x), dtype=np.float64)
    scipy_result.x = x_opt_physical 
    
    optimized_model = model.with_params(x_opt_physical)
    return optimized_model, scipy_result