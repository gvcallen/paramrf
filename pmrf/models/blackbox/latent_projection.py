import jax
import jax.numpy as jnp
import equinox as eqx

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
    encoder: eqx.Module = eqx.field(static=True)
    
    # The latent decoder. This must have K flat parameters
    decoder: BlackBox = eqx.field(static=True)
    
    # The current parameters of length P
    params: Parameter
    
    def forward(self) -> jnp.ndarray:
        # The forward model, which produces a sample for the current parameters
        return self.decoder.with_params(self.encoder(self.flat_param_values())).forward()
    
    def plot_latent(self, X, X_train=None, y_train=None, in_axis=None, out_axis=None, uncertainty=False, fig=None, axes=None, clear=True):
        import matplotlib.pyplot as plt
        import numpy as np # Ensure numpy is imported for checking array types

        X = jnp.atleast_2d(jnp.asarray(X)).reshape(X.shape[0], -1)
        if uncertainty:
            Y_mean, Y_var = jax.vmap(lambda x: self.encoder(x, return_var=True))(X)
        else:
            Y_mean = jax.vmap(lambda x: self.encoder(x))(X)
            Y_var = None
            
        if in_axis is None:
            in_axis = 0
            
        if out_axis is None:
            out_axis = list(range(Y_mean.shape[1]))

        rows = int(jnp.sqrt(len(out_axis)))
        cols = len(out_axis) // rows
            
        # --- LOGIC FIX START ---
        if fig is None:
            # No figure provided: Create a NEW figure and axes
            fig, axes = plt.subplots(rows, cols)
        else:
            # Figure provided: Reuse it
            if clear:
                fig.clf()  # Clear the existing figure
                axes = fig.subplots(rows, cols) # Create new axes on the reused figure
            elif axes is None:
                # Figure exists, not clearing, but no axes provided: create them
                axes = fig.subplots(rows, cols)
        # --- LOGIC FIX END ---

        # Ensure axes is always a flat array (handles single plot case)
        if not isinstance(axes, (np.ndarray, list, jnp.ndarray)):
            axes = np.array([axes])
        axes = axes.flatten()

        for i, out_axis_i in enumerate(out_axis):
            ax = axes[i]
            ax.set_title(f'Latent Projection (in = {in_axis}, out = {out_axis_i})')
            
            _plot_with_variance(X, Y_mean, Y_var, in_axis=in_axis, out_axis=out_axis_i, ax=ax, label='model')
            if X_train is not None and y_train is not None:
                ax.scatter(X_train[:,in_axis], y_train[:,out_axis_i], color='black', marker='x', label='train')
        
            if i == 0:
                ax.legend()
        
        # Update figure size and layout
        fig.set_size_inches((cols*4, rows*3))
        fig.tight_layout()
        
        # Necessary if you want to see the plot update live during the loop
        if plt.get_backend() != 'agg':
            plt.pause(0.1) 
        
        return fig, axes
        
def _plot_with_variance(X, Y_mean, Y_var=None, in_axis=None, out_axis=None, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    
    if in_axis is not None:
        X = X[:,in_axis]
    if in_axis is not None:
        Y_mean = Y_mean[:,out_axis]
    if in_axis is not None and Y_var is not None:
        Y_var = Y_var[:,out_axis]
        
    ax = ax or plt.gca()
    fig = ax.figure    
    
    X = np.asarray(X)
    Y_mean = np.asarray(Y_mean)
    if Y_var is not None:
        Y_var = np.asarray(Y_var)
        Y_std = np.sqrt(Y_var)

    # Mean
    ax.plot(X, Y_mean, **kwargs)

    if Y_var is not None:
        # Variance bands (outer → inner)
        ax.fill_between(
            X,
            Y_mean - 3 * Y_std,
            Y_mean + 3 * Y_std,
            alpha=0.15
        )
        ax.fill_between(
            X,
            Y_mean - 2 * Y_std,
            Y_mean + 2 * Y_std,
            alpha=0.25
        )
        ax.fill_between(
            X,
            Y_mean - 1 * Y_std,
            Y_mean + 1 * Y_std,
            alpha=0.35
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")