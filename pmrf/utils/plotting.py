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
        
def plot_with_variance(X, Y_mean, Y_var=None, in_axis=None, out_axis=None, ax=None, is_surface=False, **kwargs):
    ax = ax or plt.gca()
    
    # --- Data Prep ---
    if in_axis is not None:
        X_plot = X[:, in_axis] # 1D case
    else:
        X_plot = X # 2D case (keep all cols)
        
    Y_mean_plot = Y_mean[:, out_axis]
    
    X_plot = np.asarray(X_plot)
    Y_mean_plot = np.asarray(Y_mean_plot)
    
    Y_std = None
    if Y_var is not None:
        Y_var_plot = np.asarray(Y_var[:, out_axis])
        Y_std = np.sqrt(Y_var_plot)

    # --- Plotting Logic ---
    
    if is_surface:
        # === 3D SURFACE PLOT ===
        # We use plot_trisurf so it works even if X isn't a perfect meshgrid
        
        # Plot Mean (Solid)
        try:
            ax.plot_trisurf(X_plot[:,0], X_plot[:,1], Y_mean_plot, cmap='viridis', alpha=0.8, linewidth=0.2, edgecolors='none')
        except:
            logging.warning("Could not plot surface mean")
        
        # Plot Uncertainty (Transparent Shells)
        if Y_std is not None:
            try:
                # Upper bound (Mean + 2 Std)
                ax.plot_trisurf(X_plot[:,0], X_plot[:,1], Y_mean_plot + 2*Y_std, color='gray', alpha=0.15, edgecolor='none')
                # Lower bound (Mean - 2 Std)
                ax.plot_trisurf(X_plot[:,0], X_plot[:,1], Y_mean_plot - 2*Y_std, color='gray', alpha=0.15, edgecolor='none')
            except:
                logging.warning("Could not plot standard deviation")
                
            
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("y")
        
    else:
        # === 1D LINE PLOT ===
        # Sort X for clean line plotting
        sort_idx = np.argsort(X_plot)
        X_plot = X_plot[sort_idx]
        Y_mean_plot = Y_mean_plot[sort_idx]
        if Y_std is not None:
            Y_std = Y_std[sort_idx]

        # Mean
        ax.plot(X_plot, Y_mean_plot, **kwargs)

        # Variance bands
        if Y_std is not None:
            ax.fill_between(X_plot, Y_mean_plot - 3*Y_std, Y_mean_plot + 3*Y_std, alpha=0.15, color=kwargs.get('color'))
            ax.fill_between(X_plot, Y_mean_plot - 2*Y_std, Y_mean_plot + 2*Y_std, alpha=0.25, color=kwargs.get('color'))
            ax.fill_between(X_plot, Y_mean_plot - 1*Y_std, Y_mean_plot + 1*Y_std, alpha=0.35, color=kwargs.get('color'))

        ax.set_xlabel("x")
        ax.set_ylabel("y")