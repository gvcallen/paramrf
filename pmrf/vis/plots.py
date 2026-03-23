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
    
    # Safely convert to list for iteration/prefixing
    if isinstance(features, str):
        features_list = [features]
    elif isinstance(features, list):
        features_list = features
    else:
        features_list = [str(features)]
        
    # --- Safe Data Prefixing ---
    if use_data_prefix:
        if isinstance(features, Evaluator):
            raise ValueError("Can only use a data prefix when features are string(s)")
        
        # Safely attempt to extract the network name
        prefix = ""
        try:
            # Check if it's a NetworkCollection wrapper
            if hasattr(result.data, 'data') and hasattr(result.data.data[0], 'name'):
                prefix = f"{result.data.data[0].name}."
            # Check if it's a single Network wrapper
            elif hasattr(result.data, 'name') and result.data.name:
                 prefix = f"{result.data.name}."
        except Exception as e:
            logger.debug(f"Could not extract data prefix: {e}")
            
        if prefix:
            features_list = [f"{prefix}{f}" for f in features_list]
    
    # Standardize evaluator
    evaluator = features if isinstance(features, Evaluator) else Alias(features_list)
    x = result.frequency.f_scaled
    unit = result.frequency.unit if result.frequency else "Hz"

    # --- Evaluate model first to determine shape ---
    y_mod = evaluator(result.model, result.frequency)
    y_mod_real = np.real(y_mod).reshape(len(x), -1)

    n_features = y_mod_real.shape[1]

    # --- Setup Dynamic Grid Axes ---
    if ax is None:
        if subplots and n_features > 1:
            # Calculate grid dimensions
            cols = int(np.ceil(np.sqrt(n_features)))
            rows = int(np.ceil(n_features / cols))
            
            # Auto-scale figsize if not provided (e.g., 4.5 width per col, 3.5 height per row)
            if 'figsize' not in fig_kwargs:
                fig_width = min(18, cols * 4.5)  # Cap width to prevent off-screen rendering
                fig_kwargs['figsize'] = (fig_width, rows * 3.5)
                
            fig, axes = plt.subplots(rows, cols, sharex=True, **fig_kwargs)
            axes_flat = np.atleast_1d(axes).flatten()
        else:
            fig, ax = _setup_figure(result.frequency, features_list, **fig_kwargs)
            axes_flat = np.array([ax])
            cols = 1
    else:
        fig = ax.figure
        axes_flat = np.atleast_1d(ax).flatten()
        cols = len(axes_flat)

    # --- Plot measured data ---
    if result.data is not None:
        try:
            if isinstance(result.data, Model):
                y_meas = evaluator(result.data, result.frequency)
            else:
                y_meas = result.data

            y_meas_real = np.real(y_meas).reshape(len(x), -1)

            for i, axis in enumerate(axes_flat):
                if i < y_meas_real.shape[1]:
                    lines = axis.plot(
                        x, y_meas_real[:, i], linestyle='--', color='k', linewidth=1.5
                    )
                    if i == 0:
                        lines[0].set_label('Measured Data')

        except Exception as e:
            logger.warning(f"Failed to evaluate feature on data: {e}")

    # --- Plot model and format axes ---
    for i, axis in enumerate(axes_flat):
        if i < y_mod_real.shape[1]:
            lines = axis.plot(
                x, y_mod_real[:, i], linestyle='-', linewidth=1.5, color='C0'
            )
            if i == 0:
                lines[0].set_label('Fitted Model')
            
            # --- Axes Formatting ---
            axis.legend(fontsize='small')
            
            # --- Smart Titling Upgrade ---
            if len(features_list) == n_features:
                # Exact 1:1 map of strings to features
                title = features_list[i]
            elif len(features_list) == 1:
                # A single string expanded into multiple features (e.g., 's' -> 4 features)
                base_feature = features_list[0]
                n_ports = int(np.sqrt(n_features))
                
                # Check if it forms a perfect square matrix
                if n_ports**2 == n_features:
                    row = (i // n_ports) + 1
                    col = (i % n_ports) + 1
                    title = f"{base_feature}{row}{col}"
                else:
                    # Generic indexed fallback if not a square matrix
                    title = f"{base_feature}_{i}"
            else:
                # Generic fallback for completely mismatched list lengths
                title = f"Feature {i}"
                
            axis.set_title(title, fontsize=10)
            
            # Only label the x-axis on the bottom row to keep it clean
            if i >= len(axes_flat) - cols:
                axis.set_xlabel(f"Frequency ({unit})")
        else:
            # Hide empty subplots if the grid isn't perfectly filled
            axis.set_visible(False)

    fig.tight_layout()

    # Return the original shape of axes if we created a grid
    if subplots and n_features > 1 and ax is None:
        return fig, axes
    return fig, axes_flat[0]