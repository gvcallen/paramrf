import parax as prx
from typing import Any

import jax.numpy as jnp

from pmrf.core import Model, Evaluator, Frequency


class OptimizeResult(prx.Module):
    """
    Standardized return object for parameter optimization routines.

    Attributes
    ----------
    model : Model
        The circuit model holding the finalized, optimized parameter state.
    cost : Evaluator
        The evaluator (e.g., metric, sum of goals) used to calculate the objective.
    value : jnp.ndarray
        The final cost value achieved by the optimizer.
    history : Any
        The underlying solution object returned by the solver.
    """
    model: Model
    cost: Evaluator
    value: jnp.ndarray
    data: Any = None               # The raw data (e.g., wrapped in a Measured model)
    frequency: Frequency | None = None
    history: Any = None

    def plot(self, features: str | list[str] | Evaluator = 's', ax=None, **kwargs):
        """
        Plots the fit quality.
        
        Parameters
        ----------
        feature : str | Evaluator
            The specific feature to extract and plot (e.g., 's_real', 's_mag_db', 
            or a custom Evaluator instance).
        """
        from pmrf.vis.plots import plot_optimization_result
        return plot_optimization_result(self, features=features, ax=ax, **kwargs)