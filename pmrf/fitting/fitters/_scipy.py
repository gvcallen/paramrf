import numpy as np
from typing import Any
import jsonpickle
import h5py

from pmrf.fitting._frequentist import FrequentistFitter, FrequentistResults
from pmrf.fitting.results._scipy import ScipyMinimizeResults

class SciPyMinimizeFitter(FrequentistFitter):
    """
    Scipy fitter using scipy.minimize.
    """
    def run(self, **kwargs) -> FrequentistResults:
        from scipy.optimize import minimize, Bounds
        
        # Extract parameter values and bounds from the model
        params = self.initial_model.flat_params()
        param_names = self.initial_model.flat_param_names()
        
        minimums, maximums = self._bounds()
        minimums, maximums = np.array(minimums), np.array(maximums)
        bounds = Bounds(minimums, maximums)
        
        x0 = np.array(self.initial_model.flat_params())
        cost_fn = self._make_cost_function(as_numpy=True)

        if np.any((maximums - x0) < 0.0) or np.any((x0 - minimums) < 0.0):
            raise Exception('Bad prior bounds')
        
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
        scipy_result = minimize(cost_scipy_fn, x0, args=(callback_args,), bounds=bounds, **kwargs)
        self.logger.info(f"fevel = {callback_args['fevel']}, cost = {scipy_result.fun:.2f}")
        self.logger.info(f"Optimization finished: {scipy_result.message}")
        
        # Reconstruct the final model with optimized parameters
        fitted_model = self.initial_model.with_flat_params(scipy_result.x)
        
        settings = self._settings(kwargs)
        return ScipyMinimizeResults(
            measured=self.measured,
            initial_model=self.initial_model,
            fitted_model=fitted_model,
            solver_results=scipy_result,
            settings=settings,
        )                