import logging
import numpy as np
import matplotlib.pyplot as plt
import skrf

from pmrf.core import Model
from pmrf.core import Frequency
from goal import FeatureGoal # Adjust import as needed
from pmrf.network_collection import NetworkCollection
from pmrf.features import extract_features

import logging

import matplotlib.pyplot as plt
import numpy as np

class LivePlotter:
    def __init__(self, title="Live Plot", xlabel="X", ylabel="Y"):
        plt.ion()  # interactive mode ON

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Dictionary to store data and line objects: 
        # { "label_name": { "x": [], "y": [], "line": line_object } }
        self.lines = {} 
        
        self.fig.show()

    def _get_or_create_line(self, label, color=None):
        """Helper to create a new line if the label doesn't exist."""
        if label not in self.lines:
            line, = self.ax.plot([], [], label=label, lw=1.0, color=color)
            self.lines[label] = {
                "x": [], 
                "y": [], 
                "line": line
            }
            # self.ax.legend(loc='upper left')
        return self.lines[label]

    def _redraw(self):
        """Handles the canvas refresh and scaling."""
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # MODE 1: Growing Axis (Stream)
    def add_point(self, label, value, x_value=None):
        """
        Appends a single value to the plot. 
        If x_value is None, it increments automatically based on list length.
        """
        data = self._get_or_create_line(label)
        
        # Append Y
        data["y"].append(value)
        
        # Determine X
        if x_value is not None:
            data["x"].append(x_value)
        else:
            # If no X provided, use the current step index
            data["x"].append(len(data["y"]) - 1)

        # Update the specific line object
        data["line"].set_data(data["x"], np.array(data["y"]))
        
        self._redraw()

    # MODE 2: Full Curve (Snapshot)
    def add_curve(self, label, y_values, x_values=None):
        """
        Replaces the entire curve for a specific label.
        Useful for plotting functions or distributions that change over time.
        """
        data = self._get_or_create_line(label)
        
        # Generate X if not provided
        if x_values is None:
            x_values = np.arange(len(y_values))
            
        # Replace data
        data["x"] = x_values
        data["y"] = y_values
        
        # Update line
        data["line"].set_data(data["x"], data["y"])
        
        self._redraw()

logger = logging.getLogger(__name__)

def _setup_figure(frequency: Frequency, feature: str):
    """Internal helper to bootstrap a formatted matplotlib figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(f"Feature: {feature}")
    ax.set_xlabel(f"Frequency ({frequency.unit})")
    
    if feature.endswith('_db'):
        ax.set_ylabel("Magnitude (dB)")
    elif feature.endswith('_deg'):
        ax.set_ylabel("Phase (Degrees)")
    elif feature.endswith('_mag'):
        ax.set_ylabel("Magnitude (Linear)")
        
    ax.grid(True, alpha=0.5)
    return fig, ax

def plot_optimization(
    model: Model, 
    frequency: Frequency, 
    feature: str, 
    *,
    goals: list[FeatureGoal] | None = None,
    initial_model: Model | None = None
):
    """
    Plots a feature of an optimized model, superimposing any relevant goals 
    and optionally the initial model for comparison.
    """
    fig, ax = _setup_figure(frequency, feature)
    x = frequency.f_scaled

    # 1. Plot Initial Model (if provided)
    if initial_model:
        try:
            y_init = extract_features(initial_model, frequency, [feature])
            ax.plot(x, np.real(y_init[:, 0]), label='Initial Model', linestyle=':', color='gray')
        except Exception as e:
            logger.warning(f"Failed to extract '{feature}' from initial model: {e}")

    # 2. Plot Optimized Model
    try:
        y_model = extract_features(model, frequency, [feature])
        ax.plot(x, np.real(y_model[:, 0]), label='Optimized Model', linestyle='-', color='b', linewidth=1.5)
    except Exception as e:
        logger.warning(f"Failed to extract '{feature}' from model: {e}")

    # 3. Overlay Goals
    if goals:
        for goal in goals:
            # Only plot goals that match the requested feature
            if getattr(goal, 'features', None) == feature or getattr(goal, 'extractors', [None])[0].property == feature:
                y_target = np.full(len(frequency), np.nan)
                
                if goal.mask is not None:
                    mask_idx = np.array(goal.mask)
                    y_target[mask_idx] = goal.target[mask_idx] if isinstance(goal.target, np.ndarray) else goal.target
                else:
                    y_target[:] = goal.target
                    
                color = 'r' if goal.operator == '<' else 'g' if goal.operator == '>' else 'k'
                ax.plot(x, y_target, label=f"Target ({goal.operator})", linestyle='--', color=color, linewidth=2)

    ax.legend(fontsize='small')
    fig.tight_layout()
    return fig, ax

def plot_fit(
    model: Model, 
    measured: skrf.Network | NetworkCollection, 
    frequency: Frequency, 
    feature: str,
    *,
    initial_model: Model | None = None
):
    """
    Plots a feature to visualize the fit quality between a model and measured data.
    """
    fig, ax = _setup_figure(frequency, feature)
    x = frequency.f_scaled
    
    networks = measured if isinstance(measured, NetworkCollection) else [measured]

    for nw in networks:
        feature_spec = {nw.name: feature} if isinstance(measured, NetworkCollection) else [feature]
        label_suffix = f" ({nw.name})" if nw.name else ""

        # 1. Plot Measured Data
        try:
            y_meas = extract_features(measured, frequency, feature_spec)
            ax.plot(x, np.real(y_meas[:, 0]), label=f'Measured{label_suffix}', linestyle='--', color='k', linewidth=1.5)
        except Exception as e:
            logger.warning(f"Failed to extract '{feature}' from measured data: {e}")

        # 2. Plot Initial Model (Optional)
        if initial_model:
            try:
                y_init = extract_features(initial_model, frequency, feature_spec)
                ax.plot(x, np.real(y_init[:, 0]), label=f'Initial{label_suffix}', linestyle=':', color='gray')
            except Exception as e:
                pass # Fail silently for initial

        # 3. Plot Fitted Model
        try:
            y_mod = extract_features(model, frequency, feature_spec)
            ax.plot(x, np.real(y_mod[:, 0]), label=f'Model{label_suffix}', linestyle='-', linewidth=1.5)
        except Exception as e:
            logger.warning(f"Failed to extract '{feature}' from fitted model: {e}")

    ax.legend(fontsize='small')
    fig.tight_layout()
    return fig, ax