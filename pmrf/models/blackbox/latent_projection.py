import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

from pmrf.models.blackbox.blackbox import BlackBox, SupervisedBlackBox
from pmrf.parameters import Parameter
from pmrf.models import Model


class LatentProjection(SupervisedBlackBox):
    """
    A model that computes its output via projection into a latent space.
    
    The output is calculated using a encoder-decoder architecture:
    - The model is a function of D parameters.
    - The encoder is an arbitrary equinox Module with D inputs and K "latent" outputs.
    - The decoder is a ParamRF model with K parameters.
    """
    # The latent encoder. Must be callable with D inputs and must have K outputs
    encoder: eqx.Module
    
    # The latent decoder. This must have K flat parameters
    decoder: BlackBox
    
    # The parameters of length P
    params: Parameter
    
    def __post_init__(self):
        self.decoder = self.decoder.with_all_params_fixed()
    
    def forward(self) -> jnp.ndarray:
        # The forward model, which produces a sample for the current parameters
        return self.decoder.with_all_params_free().with_params(self.encoder(self.flat_param_values())).forward()
    
    def plot_latent(self, X, X_train=None, y_train=None, in_axis=None, out_axis=None, uncertainty=False, fig=None, axes=None, clear=True):
        
        # 1. Normalize Input
        X = jnp.atleast_2d(jnp.asarray(X)).reshape(X.shape[0], -1)
        input_dim = X.shape[1]
        
        # 2. Run Model
        if uncertainty:
            Y_mean, Y_var = jax.vmap(lambda x: self.encoder(x, return_var=True))(X)
        else:
            Y_mean = jax.vmap(lambda x: self.encoder(x))(X)
            Y_var = None
            
        # 3. Determine Plotting Mode (1D Line vs 2D Surface)
        is_surface = (in_axis is None) and (input_dim == 2)
        
        # If not a surface and in_axis is still None, default to index 0 for 1D plots
        if in_axis is None and not is_surface:
            in_axis = 0
            
        if out_axis is None:
            out_axis = list(range(Y_mean.shape[1]))

        # --- GRID CALCULATION FIX START ---
        n_plots = len(out_axis)
        # Calculate cols first as ceil(sqrt(N)), then rows as ceil(N / cols)
        cols = int(np.ceil(np.sqrt(n_plots)))
        rows = int(np.ceil(n_plots / cols))
        # --- GRID CALCULATION FIX END ---
        
        subplot_kw = {'projection': '3d'} if is_surface else {}

        # 4. Setup Figure and Axes
        if fig is None:
            fig, axes = plt.subplots(rows, cols, subplot_kw=subplot_kw)
        else:
            if clear:
                fig.clf()
                axes = fig.subplots(rows, cols, subplot_kw=subplot_kw)
            elif axes is None:
                axes = fig.subplots(rows, cols, subplot_kw=subplot_kw)

        # Ensure axes is always a flat array
        if not isinstance(axes, (np.ndarray, list, jnp.ndarray)):
            axes = np.array([axes])
        axes = axes.flatten()

        # 5. Plotting Loop
        for i, out_axis_i in enumerate(out_axis):
            ax = axes[i]
            
            in_label = "ALL (2D)" if is_surface else str(in_axis)
            ax.set_title(f'Latent: in={in_label} -> out={out_axis_i}')
            
            _plot_with_variance(X, Y_mean, Y_var, in_axis=in_axis, out_axis=out_axis_i, ax=ax, label='Mean', is_surface=is_surface)
            
            if X_train is not None and y_train is not None:
                if is_surface:
                    ax.scatter(X_train[:,0], X_train[:,1], y_train[:,out_axis_i], color='black', marker='x', label='Train')
                else:
                    ax.scatter(X_train[:,in_axis], y_train[:,out_axis_i], color='black', marker='x', label='Train')
        
            if i == 0:
                ax.legend()
                
        # Hide empty subplots if grid is larger than N plots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        # 6. Final Polish
        fig.set_size_inches((cols*5, rows*4))
        fig.tight_layout()
        
        if plt.get_backend() != 'agg':
            plt.pause(0.1) 
        
        return fig, axes
        
def _plot_with_variance(X, Y_mean, Y_var=None, in_axis=None, out_axis=None, ax=None, is_surface=False, **kwargs):
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