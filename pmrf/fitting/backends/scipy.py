import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize, Bounds, OptimizeResult
from tqdm.auto import tqdm

from pmrf.fitting.frequentist import FrequentistFitter
from pmrf.models.model import Model

class SciPyMinimizeFitter(FrequentistFitter):
    """
    Frequentist fitter using the SciPy minimize backend.
    
    This class wraps ``scipy.optimize.minimize`` to perform bounded, classical 
    optimization on the model parameters. By default, it uses the Sequential 
    Least Squares Programming (SLSQP) algorithm.
    """
    def optimize(
        self, 
        target: jnp.ndarray, 
        *, 
        solver=None, 
        max_iter=None, 
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
            The specific solver algorithm to use (e.g., 'SLSQP', 'L-BFGS-B', 
            'Nelder-Mead'). Defaults to 'SLSQP'.
        max_iter : int, optional
            The maximum number of iterations or function evaluations allowed. 
            Overrides the ``maxiter`` key in the ``options`` dictionary if provided.
        show_progress : bool, default=True
            Whether to display a ``tqdm`` progress bar tracking the cost function 
            evaluations.
        **kwargs
            Additional keyword arguments passed directly to ``scipy.optimize.minimize``.

        Returns
        -------
        tuple[:class:`~pmrf.models.model.Model`, :class:`~scipy.optimize.OptimizeResult`]
            The fitted model and the raw result object from SciPy.

        Raises
        ------
        ValueError
            If the model's initial parameter values fall outside the bounds 
            defined by their prior distributions.
        """
        solver = solver or 'SLSQP'
        kwargs.setdefault('method', solver)
        
        # 1. Parameter Initialization & Bounds
        minimums, maximums = self.model.distribution().bounds
        minimums, maximums = np.array(minimums), np.array(maximums)
        bounds = Bounds(minimums, maximums)
        
        x0 = np.array(self.model.flat_param_values())

        # Validate initial guess against bounds
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

        self.logger.info(f"Starting SciPy minimize ({kwargs['method']})...")

        # 3. Optimization Loop with tqdm
        with tqdm(desc="Optimizing", unit=" eval", disable=not show_progress) as pbar:
            def cost_wrapper(x):
                c = float(self.cost(x, target))
                pbar.update(1)
                pbar.set_postfix({'cost': f"{c:.4f}"})
                return c
                
            scipy_result = minimize(cost_wrapper, x0, bounds=bounds, **kwargs)
            pbar.set_postfix({'cost': f"{scipy_result.fun:.4f}"})

        self.logger.info(
            f"Optimization finished: {scipy_result.message} "
            f"(Cost: {scipy_result.fun:.2f}, Evals: {scipy_result.nfev})"
        )
        
        # 4. Return Model + Raw Result
        fitted_model = self.model.with_params(scipy_result.x)
        return fitted_model, scipy_result