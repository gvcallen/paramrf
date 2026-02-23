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
    def _run_algorithm(self, ctx: FrequentistContext, *, optimizer='SLSQP', max_iterations=None, show_progress=True, **kwargs) -> FrequentistResults:
        """
        Executes the optimization using SciPy.

        Parameters
        ----------
        ctx : FrequentistContext
            The context containing the model, data, and cost function.
        optimizer : str, optional, default='SLSQP'
            The name of the SciPy optimizer method to use (e.g., 'SLSQP', 'Nelder-Mead', 'BFGS').
        max_iterations : int, optional, default=None
            The maximum number of iterations allowed for the optimizer.
        show_progress : bool, optional, default=True
            Whether to display a tqdm progress bar tracking function evaluations.
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
        from tqdm.auto import tqdm  # Using auto so it renders nicely in notebooks or terminals

        kwargs.setdefault('method', optimizer)
        
        # Extract parameter values and bounds from the model
        param_names = ctx.model.flat_param_names()
        
        minimums, maximums = ctx.bounds()
        minimums, maximums = np.array(minimums), np.array(maximums)
        bounds = Bounds(minimums, maximums)
        
        x0 = np.array(ctx.model.flat_param_values())

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
        
        cost_fn = ctx.make_cost_function(as_numpy=True)
        
        options = kwargs.get('options', {})
        if max_iterations is not None:
            options.setdefault('maxiter', max_iterations)
        kwargs['options'] = options

        self.logger.info(f"Using scipy-minimize-{kwargs.get('method', 'default')}")

        # Execute optimization within a tqdm context
        with tqdm(desc="Optimizing", unit=" eval", disable=not show_progress) as pbar:
            def cost_scipy_fn(x):
                cost = cost_fn(x)
                pbar.update(1)
                pbar.set_postfix({'cost': f"{cost:.4f}"})
                return cost
                
            scipy_result = minimize(cost_scipy_fn, x0, bounds=bounds, **kwargs)
            
            # Optional: ensure the progress bar reflects the final cost cleanly
            pbar.set_postfix({'cost': f"{scipy_result.fun:.4f}"})

        self.logger.info(f"Optimization finished: {scipy_result.message} (Cost: {scipy_result.fun:.2f}, Evals: {scipy_result.nfev})")
        
        # Reconstruct the final model with optimized parameters
        fitted_model = ctx.model.with_params(scipy_result.x)
        
        # Note: adjust the return type to match your original implementation if SciPyResults is an alias
        return SciPyResults(fitted_model=fitted_model, solver_results=scipy_result)