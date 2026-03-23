# pmrf.vis.plots.py

import matplotlib.pyplot as plt
import numpy as np
import logging

from pmrf.optimize.result import OptimizeResult
from pmrf.core import Model, Evaluator
from pmrf.evaluators import Alias

logger = logging.getLogger(__name__)

def _setup_figure(frequency, feature, **kwargs):
    kwargs.setdefault('figsize', (8, 6))
    fig, ax = plt.subplots(**kwargs)
    unit = frequency.unit if frequency else "Hz"
    ax.set_xlabel(f"Frequency ({unit})")
    ax.set_ylabel(str(feature))
    return fig, ax

def plot_optimization_result(
    result: OptimizeResult,
    features: str | list[str] | Evaluator = 's',
    ax=None,
    subplots: bool = False,
    use_data_prefix: bool = False,
    **fig_kwargs
):
    if result.frequency is None:
        raise ValueError("Cannot plot: frequency is missing from the result.")
    
    if isinstance(features, str):
        features = [features]
        
    if use_data_prefix:
        if not isinstance(features, list):
            raise Exception("Can only use a data prefix when features are string(s)")
        
        prefix = f"{result.data.data[0].name}."
        features = [f"{prefix}{feature}" for feature in features]
    
    # Standardize evaluator
    evaluator = features if isinstance(features, Evaluator) else Alias(features)

    x = result.frequency.f_scaled

    # --- Evaluate model first to determine shape ---
    y_mod = evaluator(result.model, result.frequency)
    y_mod_real = np.real(y_mod).reshape(len(x), -1)

    n_features = y_mod_real.shape[1]

    # --- Setup axes ---
    if ax is None:
        if subplots and n_features > 1:
            fig, axes = plt.subplots(n_features, 1, sharex=True, **fig_kwargs)
            axes = np.atleast_1d(axes)
        else:
            fig, ax = _setup_figure(result.frequency, features, **fig_kwargs)
            axes = np.array([ax])
    else:
        fig = ax.figure
        axes = np.atleast_1d(ax)

    # --- Plot measured data ---
    if result.data is not None:
        try:
            if isinstance(result.data, Model):
                y_meas = evaluator(result.data, result.frequency)
            else:
                y_meas = result.data

            y_meas_real = np.real(y_meas).reshape(len(x), -1)

            for i, axis in enumerate(axes):
                if i < y_meas_real.shape[1]:
                    lines = axis.plot(
                        x,
                        y_meas_real[:, i],
                        linestyle='--',
                        color='k',
                        linewidth=1.5
                    )
                    if i == 0:
                        lines[0].set_label('Measured Data')

        except Exception as e:
            logger.warning(f"Failed to evaluate feature on data: {e}")

    # --- Plot model ---
    for i, axis in enumerate(axes):
        if i < y_mod_real.shape[1]:
            lines = axis.plot(
                x,
                y_mod_real[:, i],
                linestyle='-',
                linewidth=1.5,
                color='C0'
            )
            if i == 0:
                lines[0].set_label('Fitted Model')

        axis.legend(fontsize='small')

    fig.tight_layout()

    return fig, axes if subplots else axes[0]