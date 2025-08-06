import numpy as np
from typing import Any
import jsonpickle
import h5py

from pmrf.fitting._frequentist import FrequentistFitter, FrequentistResults
from pmrf.fitting.results._scipy import ScipyMinimizeResults

class ScipyMinimizeFitter(FrequentistFitter):
    """
    **Overview**

    A concrete fitter that uses the powerful `scipy.optimize.minimize` function
    to find the optimal model parameters.

    This class provides a versatile interface to the wide variety of gradient-based
    and gradient-free optimization algorithms available in SciPy, making it a
    robust choice for many fitting problems. The cost function is internally JIT-compiled
    with JAX for performance.

    **Example**

    ```python
    import pmrf as prf
    import skrf as rf

    # 1. Load measurement data
    measured_ntwk = rf.Network('my_device.s2p')

    # 2. Create the parametric model to fit
    model_to_fit = prf.models.DatasheetCoaxial(length=0.1)

    # 3. Instantiate and configure the fitter
    fitter = prf.fitting.ScipyMinimizeFitter(
        model=model_to_fit,
        measured=measured_ntwk,
        features=[('s_db', (0,0)), ('s_db', (1,0))] # Fit to S11 and S21 in dB
    )

    # 4. Run the fit, passing optimizer-specific options via kwargs
    # Here, we use the 'SLSQP' algorithm.
    fit_result = fitter.run(method='SLSQP')

    # 5. Print the results
    print("Fit complete. Optimized model:")
    print(fit_result.model)
    ```
    """
    def run(self, *args, **kwargs) -> FrequentistResults:
        """Executes the optimization using `scipy.optimize.minimize`.

        This method sets up the JAX-compiled cost function, defines parameter
        bounds, and invokes the SciPy minimizer.

        Args:
            *args: Positional arguments passed directly to `scipy.optimize.minimize`.
            **kwargs: Keyword arguments passed directly to `scipy.optimize.minimize`,
                      allowing full control over the solver (e.g., `method='BFGS'`,
                      `tol=1e-6`).

        Returns:
            FrequentistResults: An object containing the optimized model and results.
        """
        from scipy.optimize import minimize, Bounds
        
        # Extract parameter values and bounds from the model
        params = self.model.flat_params()
        param_names = self.model.flat_param_names()
        
        minimums, maximums = self._bounds()
        minimums, maximums = np.array(minimums), np.array(maximums)
        bounds = Bounds(minimums, maximums)

        x0 = self.model.flat_params()
        cost_fn = self._make_cost_function(as_numpy=True)
        
        # Define a wrapper function compatible with SciPy's interface
        def cost_scipy_fn(x, callback_args):
            cost = cost_fn(x)
            i = callback_args['fevel']
            if i % 500 == 0:
                self.logger.info(f"fevel = {i}, cost = {cost:.2f}")
            callback_args['fevel'] += 1
            return cost

        callback_args = {'fevel': 0}
        self.logger.info(f"Fitting for {len(x0)} parameters with scipy-minimize-{kwargs.get('method', 'default')}")
        self.logger.info(f"Parameter names: {param_names}")
        scipy_result = minimize(cost_scipy_fn, x0, args=(callback_args,), bounds=bounds, *args, **kwargs)
        self.logger.info(f"fevel = {callback_args['fevel']}, cost = {scipy_result.fun:.2f}")
        self.logger.info(f"Optimization finished: {scipy_result.message}")
        
        # Reconstruct the final model with optimized parameters
        fit_model = self.model.with_flat_params(scipy_result.x)
        
        return ScipyMinimizeResults(
            fit_model=fit_model,
            initial_model=self.model,
            frequency=self.frequency,
            measured=self.measured,
            features=self.feature_list,
            logger=self.logger,
            solver_results=scipy_result,
            solver_args=args,
            solver_kwargs=kwargs,
        )