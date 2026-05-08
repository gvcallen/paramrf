from typing import Any

import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.evaluators import AbstractEvaluator
from pmrf.optimize import OptimizeResult
from pmrf.infer import InferResult

class FitResult(eqx.Module):
    """
    Standardized return object for a fitting routines.

    Wraps an :class:`pmrf.optimize.OptimizeResult` or :class:`pmrf.infer.InferResult`
    with added information for easy plotting.
    """
    #: The data used for the fit.
    data: Any = None
        
    #: The frequeny used for the fit.
    frequency: Frequency | None = None

    #: The underlying :class:`pmrf.optimize.OptimizeResult` or :class:`pmrf.infer.InferResult` result.
    metrics: OptimizeResult | InferResult | None = None
    
    @property
    def model(self):
        """
        The fitted model.
        """
        return self.metrics.best_model

    def plot(self, features: str | list[str] | AbstractEvaluator = 's', ax=None, **kwargs):
        """
        Plots the best fit.
        
        Parameters
        ----------
        feature : str | Evaluator
            The specific feature to extract and plot (e.g., 's_real', 's_mag_db', 
            or a custom Evaluator instance).
        """
        from pmrf.viz.plots import plot_fit_result
        return plot_fit_result(self, features=features, ax=ax, **kwargs)


    def __getattr__(self, name: str):
        """
        Intercepts calls to undefined methods to dynamically route plot requests.
        Example: `plot_s_db(m=0, n=1)` routes to `plot('s21_db')`.
        """
        if name.startswith('plot_') and name != 'plot':
            # Extract the base feature name (e.g., 's_db' from 'plot_s_db')
            feature_base = name[5:] 
            
            def dynamic_plot(m=None, n=None, ax=None, **kwargs):
                feature_name = feature_base
                
                # If m and n are provided, inject them into the feature string
                if m is not None and n is not None:
                    parts = feature_base.split('_', 1)
                    
                    # Convert 0-based to 1-based indexing. 
                    # The prompt's mapping of (m=0, n=1) to '21' means n comes first.
                    # Adjust {n+1}{m+1} to {m+1}{n+1} here if that was a typo for '12'.
                    port_str = f"{n+1}{m+1}" 
                    
                    if len(parts) == 2:
                        # e.g. parts[0]='s', parts[1]='db' -> 's21_db'
                        feature_name = f"{parts[0]}{port_str}_{parts[1]}"
                    else:
                        # Fallback for things like plot_s(m=0, n=1) -> 's21'
                        feature_name = f"{feature_base}{port_str}" 
                        
                return self.plot(features=feature_name, ax=ax, **kwargs)
            
            return dynamic_plot
            
        # Standard fallback if the missing attribute isn't a plot_ hook
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
