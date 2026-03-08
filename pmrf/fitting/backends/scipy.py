import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize, Bounds, OptimizeResult
from tqdm.auto import tqdm

from pmrf.fitting.frequentist import FrequentistFitter
from pmrf.models.model import Model

class SciPyMinimizeFitter(FrequentistFitter):
    """
    Frequentist fitter using the SciPy minimize backend with JAX acceleration.
    
    This class wraps ``scipy.optimize.minimize``, executing the optimization
    in a normalized parameter space [0, 1] for better numerical stability.
    """
    def execute(
        self, 
        target: jnp.ndarray, 
        *, 
        solver=None, 
        max_iter=None, 
        use_jac=True,
        show_progress=True, 
        **kwargs
    ) -> tuple[Model, OptimizeResult]:
        """
        Run the optimization loop using the SciPy backend.
        """
        solver = solver or 'SLSQP'
        kwargs.setdefault('method', solver)
        
        # 1. Parameter Initialization & Physical Bounds
        minimums, maximums = self.model.distribution().bounds
        minimums = np.array(minimums, dtype=np.float64)
        maximums = np.array(maximums, dtype=np.float64)
        
        # Calculate scales to map between [min, max] and [0, 1]
        scales = maximums - minimums
        if np.any(scales <= 0):
            raise ValueError("Maximum bounds must be strictly greater than minimum bounds for normalization.")

        # Ensure physical x0 is float64
        x0_physical = np.array(self.model.flat_param_values(), dtype=np.float64)

        # Validate initial physical guess
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
        z0 = (x0_physical - minimums) / scales
        normalized_bounds = Bounds(np.zeros_like(z0), np.ones_like(z0))

        # 3. Setup Options
        options = kwargs.get('options', {})
        if max_iter is not None:
            options.setdefault('maxiter', max_iter)
        kwargs['options'] = options

        self.logger.info(f"Starting SciPy minimize ({kwargs['method']})")

        # Prepare JAX arrays for the un-normalization function
        jnp_minimums = jnp.array(minimums)
        jnp_scales = jnp.array(scales)
        
        def unnormalize_fn(z):
            """Converts normalized parameter z [0, 1] back to physical x [min, max]"""
            return z * jnp_scales + jnp_minimums

        # 4. Define Objective and Gradient (in normalized space)
        if use_jac:
            # Wrap the cost function to accept normalized coordinates (z).
            # JAX's value_and_grad automatically applies the chain rule!
            @jax.jit
            def normalized_cost(z):
                x_physical = unnormalize_fn(z)
                return self.cost(x_physical, target)

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
            # Classic mode: Jacobian approximated by SciPy in normalized space
            def objective(z, pbar):
                x_physical = np.array(unnormalize_fn(z), dtype=np.float64)
                val = float(self.cost(x_physical, target))
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
        
        # Overwrite the result object's raw array so the caller sees physical 
        # parameters if they inspect the OptimizeResult directly.
        scipy_result.x = x_opt_physical 
        
        fitted_model = self.model.with_params(x_opt_physical)
        return fitted_model, scipy_result