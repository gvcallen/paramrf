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