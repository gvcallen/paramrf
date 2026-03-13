import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize, Bounds, OptimizeResult
from tqdm.auto import tqdm

from pmrf.models.model import Model
from pmrf.optimize.frequentist import FrequentistOptimizer

class SciPyMinimizeOptimizer(FrequentistOptimizer):
    """
    Frequentist optimizer using the SciPy minimize backend with JAX acceleration.
    
    This class wraps ``scipy.optimize.minimize``, executing the optimization
    in a normalized parameter space [0, 1] for fully bounded parameters, 
    while preserving natural scaling for unbounded or semi-bounded parameters.
    """
    def execute(
        self, 
        *, 
        use_jac=True,
        show_progress=True, 
        **kwargs
    ) -> tuple[Model, OptimizeResult]:
        """
        Run the goal-oriented optimization loop using the SciPy backend.
        """
        # 1. Parameter Initialization & Physical Bounds
        dist = self.model.distribution()
        minimums = np.array(dist.min, dtype=np.float64)
        maximums = np.array(dist.max, dtype=np.float64)
        
        is_bounded = np.isfinite(minimums) & np.isfinite(maximums)
        
        scales = np.ones_like(minimums)
        shifts = np.zeros_like(minimums)
        
        scales[is_bounded] = maximums[is_bounded] - minimums[is_bounded]
        shifts[is_bounded] = minimums[is_bounded]
        
        if np.any(scales <= 0):
            raise ValueError("Maximum bounds must be strictly greater than minimum bounds for normalization.")

        x0_physical = np.array(self.model.flat_param_values(), dtype=np.float64)

        too_low, too_high = x0_physical < minimums, x0_physical > maximums
        if np.any(too_low | too_high):
            param_names = self.model.flat_param_names()
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
        self.logger.info(f"Starting SciPy minimize optimization ({method_name})")

        jnp_shifts = jnp.array(shifts)
        jnp_scales = jnp.array(scales)
        
        def unnormalize_fn(z):
            return z * jnp_scales + jnp_shifts

        # 4. Define Objective and Gradient
        if use_jac:
            @jax.jit
            def normalized_cost(z):
                x_physical = unnormalize_fn(z)
                # CRITICAL CHANGE: No 'target' argument passed to cost()
                return self.cost(x_physical)

            vg_fn = jax.value_and_grad(normalized_cost)
            
            def objective(z, pbar):
                val, grad = vg_fn(z)
                val_np = float(val)
                grad_np = np.array(grad, dtype=np.float64)

                if np.isnan(val_np) or np.isinf(val_np):
                    self.logger.warning(f"Bad value encountered at normalized z = {z}")
                if np.any(np.isnan(grad_np)) or np.any(np.isinf(grad_np)):
                    self.logger.warning(f"Bad gradient encountered at normalized z = {z}")
                
                pbar.update(1)
                pbar.set_postfix({'cost': f"{val_np:.4f}"})
                return val_np, grad_np
            
            kwargs['jac'] = True 
        else:
            def objective(z, pbar):
                x_physical = np.array(unnormalize_fn(z), dtype=np.float64)
                # CRITICAL CHANGE: No 'target' argument passed to cost()
                val = float(self.cost(x_physical))
                pbar.update(1)
                pbar.set_postfix({'cost': f"{val:.4f}"})
                return val
            
            kwargs['jac'] = False
            
        # 5. Optimization Loop
        with tqdm(desc="Optimizing", unit=" iter", disable=not show_progress) as pbar:
            scipy_result = minimize(
                objective, 
                z0, 
                args=(pbar,), 
                bounds=normalized_bounds, 
                **kwargs
            )
            pbar.set_postfix({'cost': f"{scipy_result.fun:.4f}"})

        self.logger.info(
            f"Optimization finished: {scipy_result.message} "
            f"(Cost: {scipy_result.fun:.2f}, nfev: {scipy_result.nfev})"
        )
        
        # 6. Un-normalize the result and Return
        x_opt_physical = np.array(unnormalize_fn(scipy_result.x), dtype=np.float64)
        scipy_result.x = x_opt_physical 
        
        optimized_model = self.model.with_params(x_opt_physical)
        return optimized_model, scipy_result