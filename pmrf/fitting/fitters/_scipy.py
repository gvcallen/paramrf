import numpy as np

from pmrf.fitting._frequentist import FrequentistFitter, FrequentistResults
from pmrf.fitting.results._scipy import ScipyMinimizeResults

class SciPyMinimizeFitter(FrequentistFitter):
    """
    Scipy fitter using scipy.minimize.
    """
    def run(self, max_iterations=None, log_every=500, **kwargs) -> FrequentistResults:
        from scipy.optimize import minimize, Bounds
        
        # Extract parameter values and bounds from the model
        param_names = self.initial_model.flat_param_names()
        
        minimums, maximums = self._bounds()
        minimums, maximums = np.array(minimums), np.array(maximums)
        bounds = Bounds(minimums, maximums)
        
        x0 = np.array(self.initial_model.flat_params())
        cost_fn = self._make_cost_function(as_numpy=True)

        too_low, too_high = x0 < minimums, x0 > maximums
        if np.any(too_low | too_high):
            bad_params = []
            for i, (name, val, minv, maxv, low, high) in enumerate(zip(param_names, x0, minimums, maximums, too_low, too_high)):
                if low or high:
                    bad_params.append(
                        f"  {name}: x0={val}, min={minv}, max={maxv} "
                        f"({'below min' if low else 'above max'})"
                    )
            bad_param_report = "\n".join(bad_params)
            raise Exception(f"Bad prior bounds:\n{bad_param_report}")
        
        # Define a wrapper function compatible with SciPy's interface
        def cost_scipy_fn(x, callback_args):
            cost = cost_fn(x)
            i = callback_args['fevel']
            if i % log_every == 0:
                self.logger.info(f"fevel = {i}, cost = {cost:.2f}")
            callback_args['fevel'] += 1
            return cost
        
        options = kwargs.get('options', {})
        if max_iterations is not None:
            options.setdefault('maxiter', max_iterations)
        kwargs['options'] = options

        callback_args = {'fevel': 0}
        self.logger.info(f"Fitting for {len(x0)} parameters with scipy-minimize-{kwargs.get('method', 'default')}")
        self.logger.info(f"Parameter names: {param_names}")
        scipy_result = minimize(cost_scipy_fn, x0, args=(callback_args,), bounds=bounds, **kwargs)
        self.logger.info(f"fevel = {callback_args['fevel']}, cost = {scipy_result.fun:.2f}")
        self.logger.info(f"Optimization finished: {scipy_result.message}")
        
        # Reconstruct the final model with optimized parameters
        fitted_model = self.initial_model.with_flat_params(scipy_result.x)
        
        settings = self._settings(kwargs)
        self.results = ScipyMinimizeResults(
            measured=self.measured,
            initial_model=self.initial_model,
            fitted_model=fitted_model,
            solver_results=scipy_result,
            settings=settings,
            fitter=self,
        )                

        return self.results