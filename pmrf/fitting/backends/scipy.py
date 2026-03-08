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
    
    This class wraps ``scipy.optimize.minimize``. It can optionally use JAX's 
    automatic differentiation to provide exact Jacobians to the solver.
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

        Parameters
        ----------
        target : jax.numpy.ndarray
            The extracted target features to fit against.
        solver : str, optional
            The specific solver algorithm (e.g., 'SLSQP', 'L-BFGS-B'). 
            Defaults to 'SLSQP'.
        max_iter : int, optional
            Overrides the ``maxiter`` key in the ``options`` dictionary.
        use_jac : bool, default=False
            If True, use JAX to compute exact gradients. If False, SciPy 
            will approximate the Jacobian using finite differences.
        show_progress : bool, default=True
            Whether to display a ``tqdm`` progress bar.
        """
        solver = solver or 'SLSQP'
        kwargs.setdefault('method', solver)
        
        # 1. Parameter Initialization & Bounds
        minimums, maximums = self.model.distribution().bounds
        minimums, maximums = np.array(minimums), np.array(maximums)
        bounds = Bounds(minimums, maximums)
        
        # Ensure x0 is float64 for SciPy stability
        x0 = np.array(self.model.flat_param_values(), dtype=np.float64)

        # Validate initial guess
        too_low, too_high = x0 < minimums, x0 > maximums
        if np.any(too_low | too_high):
            param_names = self.model.flat_param_names()
            bad_params = [
                f"  {name}: x0={val}, min={minv}, max={maxv} ({'below min' if low else 'above max'})"
                for name, val, minv, maxv, low, high in zip(param_names, x0, minimums, maximums, too_low, too_high)
                if low or high
            ]
            raise ValueError(f"Initial parameters outside bounds:\n" + "\n".join(bad_params))
        
        # 2. Setup Options
        options = kwargs.get('options', {})
        if max_iter is not None:
            options.setdefault('maxiter', max_iter)
        kwargs['options'] = options

        self.logger.info(f"Starting SciPy minimize ({kwargs['method']})")

        # 3. Define Objective and Gradient
        if use_jac:
            # Compiled value and grad pass
            vg_fn = jax.jit(jax.value_and_grad(lambda x: self.cost(x, target)))
            
            def objective(x, pbar):
                val, grad = vg_fn(x)
                # Convert to numpy for SciPy
                val_np = float(val)
                grad_np = np.array(grad, dtype=np.float64)
                
                pbar.update(1)
                pbar.set_postfix({'cost': f"{val_np:.4f}"})
                return val_np, grad_np
            
            kwargs['jac'] = True # Tell SciPy the objective returns (val, grad)
        else:
            # Classic mode: Jacobian approximated by SciPy
            def objective(x, pbar):
                val = float(self.cost(x, target))
                pbar.update(1)
                pbar.set_postfix({'cost': f"{val:.4f}"})
                return val
            
            kwargs['jac'] = False

        # 4. Optimization Loop
        with tqdm(desc="Optimizing", unit=" iter", disable=not show_progress) as pbar:
            scipy_result = minimize(
                objective, 
                x0, 
                args=(pbar,), 
                bounds=bounds, 
                **kwargs
            )
            pbar.set_postfix({'cost': f"{scipy_result.fun:.4f}"})

        self.logger.info(
            f"Optimization finished: {scipy_result.message} "
            f"(Cost: {scipy_result.fun:.2f}, nfev: {scipy_result.nfev})"
        )
        
        # 5. Return Model + Raw Result
        fitted_model = self.model.with_params(scipy_result.x)
        return fitted_model, scipy_result