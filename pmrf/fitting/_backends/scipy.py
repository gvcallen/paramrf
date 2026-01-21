from typing import Any

import numpy as np
import h5py

from pmrf.fitting.frequentist import FrequentistFitter, FrequentistResults, FrequentistContext

class SciPyResults(FrequentistResults):
    """
    SciPy: Results from ``scipy.optimize``.

    This class provides specific serialization logic for ``scipy.optimize.OptimizeResult`` objects.
    """    
    def encode_solver_results(self, grp: h5py.Group):
        """
        Encode the SciPy solver results into an HDF5 group.
        
        This method iterates over the `OptimizeResult` attributes, storing scalars/strings
        as HDF5 attributes and arrays as HDF5 datasets.

        Parameters
        ----------
        grp : h5py.Group
            The HDF5 group to write the results to.
        """
        for key, val in self.solver_results.items():
            if isinstance(val, (int, float, str, np.number)):
                grp.attrs[key] = val
            elif isinstance(val, np.ndarray):
                grp.create_dataset(key, data=val)
            elif val is None:
                grp.attrs[key] = "None"
            else:
                grp.attrs[key] = str(val)  # fallback for e.g. status messages
    
    @classmethod
    def decode_solver_results(cls, grp: h5py.Group) -> Any:
        """
        Decode SciPy solver results from an HDF5 group.

        Parameters
        ----------
        grp : h5py.Group
            The HDF5 group to read the results from.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The reconstructed optimization result object.
        """
        result_dict = dict(grp.attrs)
        for key in grp:
            result_dict[key] = grp[key][()]
            
        from scipy.optimize import OptimizeResult
        return OptimizeResult(result_dict)
    
class SciPyMinimizeFitter(FrequentistFitter):
    """
    SciPy Minimize: Classical optimization using ``scipy.optimize.minimize``.
    """
    def _run_algorithm(self, ctx: FrequentistContext, *, max_iterations=1000, optimizer='SLSQP', log_every=500, **kwargs) -> FrequentistResults:
        """
        Executes the optimization using SciPy.

        Parameters
        ----------
        ctx : FrequentistContext
            The context containing the model, data, and cost function.
        max_iterations : int, optional, default=1000
            The maximum number of iterations allowed for the optimizer.
        optimizer : str, optional, default='SLSQP'
            The name of the SciPy optimizer method to use (e.g., 'SLSQP', 'Nelder-Mead', 'BFGS').
        log_every : int, optional, default=500
            The interval (in function evaluations) at which to log the current cost.
        **kwargs
            Additional keyword arguments passed directly to `scipy.optimize.minimize`.

        Returns
        -------
        FrequentistResults
            The results containing the fitted model and the raw `OptimizeResult`.

        Raises
        ------
        Exception
            If the initial model parameters (`x0`) are outside the defined bounds.
        """
        from scipy.optimize import minimize, Bounds

        kwargs.setdefault('method', optimizer)
        
        # Extract parameter values and bounds from the model
        param_names = ctx.model.flat_param_names()
        
        minimums, maximums = ctx.bounds()
        minimums, maximums = np.array(minimums), np.array(maximums)
        bounds = Bounds(minimums, maximums)
        
        x0 = np.array(ctx.model.flat_param_values())
        cost_fn = ctx.make_cost_function(as_numpy=True)

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
        self.logger.info(f"Using scipy-minimize-{kwargs.get('method', 'default')}")
        scipy_result = minimize(cost_scipy_fn, x0, args=(callback_args,), bounds=bounds, **kwargs)
        self.logger.info(f"fevel = {callback_args['fevel']}, cost = {scipy_result.fun:.2f}")
        self.logger.info(f"Optimization finished: {scipy_result.message}")
        
        # Reconstruct the final model with optimized parameters
        fitted_model = ctx.model.with_params(scipy_result.x)
        
        return SciPyResults(fitted_model=fitted_model, solver_results=scipy_result)