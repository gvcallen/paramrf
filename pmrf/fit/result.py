import parax as prx
from typing import Any

from pmrf.core import Operator, Frequency
from pmrf.optimize import OptimizeResult
from pmrf.infer import InferResult

class FitResult(prx.Module):
    """
    Standardized return object for a fitting routines.

    Attributes
    ----------
    data: Any
        The data used for the fit.
    frequency: Frequency
        The frequeny used for the fit.
    solution: 
        The underlying OptimizeResult or InferResult
    """
    data: Any = None
    frequency: Frequency | None = None
    solution: OptimizeResult | InferResult | None = None
    
    @property
    def model(self):
        return self.solution.model

    def plot(self, features: str | list[str] | Operator = 's', ax=None, **kwargs):
        """
        Plots the best fit.
        
        Parameters
        ----------
        feature : str | Evaluator
            The specific feature to extract and plot (e.g., 's_real', 's_mag_db', 
            or a custom Evaluator instance).
        """
        from pmrf.vis.plots import plot_fit_result
        return plot_fit_result(self, features=features, ax=ax, **kwargs)