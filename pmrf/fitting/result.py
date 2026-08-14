from typing import Any, Generic, TypeVar

import equinox as eqx

from pmrf.modules.base import Module
from pmrf.frequency import Frequency
from pmrf.evaluators import AbstractEvaluator
from pmrf.optimize import OptimizeResult
from pmrf.infer import InferResult

ModuleT = TypeVar('ModuleT', bound=Module)

class FitResult(eqx.Module, Generic[ModuleT]):
    """
    Standardized return object for a fitting routines.

    Wraps an :class:`pmrf.optimize.OptimizeResult` or :class:`pmrf.infer.InferResult`
    with added information for easy plotting.
    """
    #: The underlying :class:`pmrf.optimize.OptimizeResult` or :class:`pmrf.infer.InferResult` result.
    solution: OptimizeResult[ModuleT] | InferResult[ModuleT]
    
    #: The data used for the fit, if available.
    data: Any = None
        
    #: The frequeny used for the fit, if available.
    frequency: Frequency | None = None

    @property
    def model(self) -> ModuleT:
        """
        The fitted model.
        """
        if isinstance(self.solution, InferResult):
            return self.solution.best_model
        elif isinstance(self.solution, OptimizeResult):
            return self.solution.model
        else:
            return None

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
        Example: `plot_s_db(m=1, n=0)` routes to `plot('s21_db')`.
        """
        if name.startswith('plot_') and name != 'plot':
            feature_base = name[5:] 
            
            def dynamic_plot(m=None, n=None, ax=None, **kwargs):
                feature_name = feature_base
                if m is not None and n is not None:
                    parts = feature_base.split('_', 1)
                    port_str = f"{m+1}{n+1}" 
                    
                    if len(parts) == 2:
                        feature_name = f"{parts[0]}{port_str}_{parts[1]}"
                    else:
                        feature_name = f"{feature_base}{port_str}" 
                return self.plot(features=feature_name, ax=ax, **kwargs)
            
            return dynamic_plot
            
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
