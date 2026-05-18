import numpy as np
import logging

import skrf

from pmrf.frequency import Frequency
from pmrf.fitting.result import FitResult
from pmrf.models import Measured
from pmrf.evaluators import AbstractEvaluator, Feature
from pmrf.network_collection import NetworkCollection

logger = logging.getLogger(__name__)

def _setup_figure(frequency, feature, **kwargs):
    import matplotlib.pyplot as plt

    kwargs.setdefault('figsize', (8, 6))
    fig, ax = plt.subplots(**kwargs)
    unit = frequency.unit if frequency else "Hz"
    ax.set_xlabel(f"Frequency ({unit})")
    ax.set_ylabel(str(feature))
    return fig, ax


def plot_fit_result(
    result: FitResult,
    features: str | list[str] | AbstractEvaluator = 's',
    ax=None,
    subplots: bool = False,
    use_data_prefix: bool | None = None,
    model_frequency: Frequency | skrf.Frequency | None = None,
    **fig_kwargs
):
    """
    Plots the fit results comparing measured data and the fitted model.

    Convenience utility called by :class:`pmrf.fitting.FitResult`.

    Parameters
    ----------
    result : FitResult
        The result object containing the fitted model, original data, and base frequency.
    features : str | list[str] | Evaluator, optional
        The features to evaluate and plot. Can be a string (e.g., 's'), a list of 
        strings, or a custom Evaluator object. Defaults to 's'.
    ax : matplotlib.axes.Axes, optional
        An existing Axes object to plot on. If None, a new figure and axes are created.
    subplots : bool, optional
        If True and multiple features are evaluated (e.g., multi-port S-parameters), 
        plots them in a dynamic grid of subplots. Defaults to False.
    use_data_prefix : bool | None, optional
        If True, prefixes feature titles with the name of the data network. If None, 
        this is automatically set to True when using a NetworkCollection.
    model_frequency : pmrf.Frequency | skrf.Frequency | None, optional
        An optional frequency object to evaluate the model against. If provided, the 
        model will be simulated and plotted at these frequencies, while the measured 
        data will still be plotted at `result.frequency`. Useful for extrapolating 
        or smoothing the model plot. Defaults to None (uses `result.frequency`).
    **fig_kwargs : dict
        Additional keyword arguments passed to `plt.subplots()` when creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the plot.
    axes : matplotlib.axes.Axes or numpy.ndarray
        The axes or array of axes used for the plot.
    """
    import matplotlib.pyplot as plt

    if result.frequency is None:
        raise ValueError("Cannot plot: frequency is missing from the result.")
    
    # Resolve feature list
    if isinstance(features, str):
        feature_list = [features]
    elif isinstance(features, list):
        feature_list = features
    else:
        feature_list = [str(features)]

    if use_data_prefix is None:
        if isinstance(result.data, NetworkCollection):
            use_data_prefix = True
        else:
            use_data_prefix = False
        
    # Apply prefix
    if use_data_prefix:
        if isinstance(features, AbstractEvaluator):
            raise ValueError("Can only use a data prefix when features are string(s)")
        
        prefix = ""
        try:
            if isinstance(result.data, NetworkCollection):
                prefix = f"{result.data[0].name}."
            else:
                prefix = f"{result.data.name}."
        except Exception as e:
            logger.debug(f"Could not extract data prefix: {e}")
            
        if prefix:
            model_feature_list = [f"{prefix}{f}" for f in feature_list]
    else:
        model_feature_list = feature_list
    
    model_evaluator = features if isinstance(features, AbstractEvaluator) else Feature(model_feature_list)
    data_evaluator = features if isinstance(features, AbstractEvaluator) else Feature(feature_list)
    
    # Frequency setup
    x_meas = result.frequency.f_scaled
    unit = result.frequency.unit if result.frequency else "Hz"
    mod_freq = model_frequency if model_frequency is not None else result.frequency
    if isinstance(mod_freq, skrf.Frequency):
        mod_freq = Frequency.from_skrf(mod_freq)
    x_mod = mod_freq.f_scaled

    # Evaluate model and extract number of features
    y_mod = model_evaluator(result.model, mod_freq)
    y_mod_real = np.real(y_mod).reshape(len(x_mod), -1)
    n_features = y_mod_real.shape[1]

    # Setup axes
    if ax is None:
        if subplots and n_features > 1:
            cols = int(np.ceil(np.sqrt(n_features)))
            rows = int(np.ceil(n_features / cols))
            
            # Auto-scale figsize if not provided (e.g., 4.5 width per col, 3.5 height per row)
            if 'figsize' not in fig_kwargs:
                fig_width = min(18, cols * 4.5)  # Cap width to prevent off-screen rendering
                fig_kwargs['figsize'] = (fig_width, rows * 3.5)
                
            fig, axes = plt.subplots(rows, cols, sharex=True, **fig_kwargs)
            axes_flat = np.atleast_1d(axes).flatten()
        else:
            fig, ax = _setup_figure(result.frequency, model_feature_list, **fig_kwargs)
            axes_flat = np.array([ax])
            cols = 1
    else:
        fig = ax.figure
        axes_flat = np.atleast_1d(ax).flatten()
        cols = len(axes_flat)

    # Plot measured
    if result.data is not None:
        try:
            if isinstance(result.data, NetworkCollection | skrf.Network):
                if isinstance(result, NetworkCollection):
                    data = Measured(result.data[0])
                else:
                    data = Measured(result.data)
                y_meas = data_evaluator(data, result.frequency)
            else:
                y_meas = result.data

            y_meas_real = np.real(y_meas).reshape(len(x_meas), -1)

            for i, axis in enumerate(axes_flat):
                if i < y_meas_real.shape[1]:
                    lines = axis.plot(
                        x_meas, y_meas_real[:, i], linestyle='--', color='k', linewidth=1.5
                    )
                    if i == 0:
                        lines[0].set_label('Measured Data')

        except Exception as e:
            logger.warning(f"Failed to evaluate feature on data: {e}")

    # Plot model and add labels
    for i, axis in enumerate(axes_flat):
        if i < y_mod_real.shape[1]:
            lines = axis.plot(
                x_mod, y_mod_real[:, i], linestyle='-', linewidth=1.5, color='C0'
            )
            if i == 0:
                lines[0].set_label('Fitted Model')
                axis.legend(fontsize='small')
            
            if len(model_feature_list) == n_features:
                title = model_feature_list[i]
            elif len(model_feature_list) == 1:
                base_feature = model_feature_list[0]
                n_ports = int(np.sqrt(n_features))
                
                if n_ports**2 == n_features:
                    row = (i // n_ports) + 1
                    col = (i % n_ports) + 1
                    title = f"{base_feature}{row}{col}"
                else:
                    title = f"{base_feature}_{i}"
            else:
                title = f"Feature {i}"
                
            axis.set_title(title, fontsize=10)
            if i >= len(axes_flat) - cols:
                axis.set_xlabel(f"Frequency ({unit})")
        else:
            axis.set_visible(False)

    fig.tight_layout()

    # Return the original shape of axes if we created a grid
    if subplots and n_features > 1 and ax is None:
        return fig, axes
    return fig, axes_flat[0]