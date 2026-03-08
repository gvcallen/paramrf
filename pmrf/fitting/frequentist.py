from abc import ABC
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.fitting.base import BaseFitter
from pmrf.models.model import Model
from pmrf.constants import ArrayFuncT
from pmrf.math_functions import l2_norm_ax0, mag_2_db, conv_inter

L2_ERROR = [l2_norm_ax0, l2_norm_ax0, mag_2_db]
CONVOLUTIONAL_ERROR = [l2_norm_ax0, conv_inter, l2_norm_ax0, mag_2_db]

class FrequentistFitter(BaseFitter, ABC):
    r"""
    A base class for frequentist (classical) optimization methods.
    
    This class sets up the objective function (or cost function) needed by standard 
    numerical optimizers (like those in SciPy). It automatically configures the 
    target features and error metrics based on the type of fit you want to perform.

    .. rubric:: Methods

    .. autosummary::
       :nosignatures:
       
       run
       execute
       cost

    Parameters
    ----------
    model : Model
        The ParamRF model containing the parameters to optimize.
    cost_kind : str, optional
        A preset string to automatically configure the features and error functions. 
        Options include 'convolutional', 'complex', or 'magnitude'. If left 
        as ``None``, it defaults to fitting the real and imaginary parts of all 
        ports in the model.
    error_fn : callable, list of callables, or eqx.Module, optional
        The specific mathematical function(s) used to calculate the final error 
        between the model's output and the target data. If not provided, standard 
        L2 norms are used based on the ``cost_kind``.
    **kwargs
        Additional arguments passed up to :class:`BaseFitter` (such as ``frequency``).
    """
    def __init__(
        self,
        model: Model, 
        *,
        cost_kind: str = None,
        error_fn: ArrayFuncT | list[ArrayFuncT] | eqx.Module | None = None,
        tikhonov_lambda: float = 0.0,
        **kwargs
    ):
        # Let base class consume features, etc.
        super().__init__(model, **kwargs)

        self.tikhonov_lambda = tikhonov_lambda
        self.initial_theta = self.model.flat_param_values()

        # Extract bounds and calculate safe parameter ranges
        lower_bounds, upper_bounds = self.model.distribution().bounds
        param_ranges = jnp.array(upper_bounds) - jnp.array(lower_bounds)
        self.param_ranges = jnp.where(param_ranges == 0.0, 1.0, param_ranges)        
        
        # Apply standard Python defaults/mutations safely to 'self'
        if self.features is None and cost_kind is None and error_fn is None:
            cost_kind = 'complex'
        
        if cost_kind == 'convolutional':
            self.features = ['s', 's_mag']
            error_fn = CONVOLUTIONAL_ERROR
        elif cost_kind == 'complex':
            self.features = ['s']
            error_fn = L2_ERROR
        elif cost_kind == 'magnitude':
            self.features = ['s_mag']
            error_fn = L2_ERROR
                
        if error_fn is None:
            error_fn = [l2_norm_ax0, l2_norm_ax0, mag_2_db]

        if not isinstance(error_fn, eqx.Module):
            if not isinstance(error_fn, list): 
                error_fn = [error_fn]
            error_fn = eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in error_fn])
            
        self._error_fn = eqx.filter_jit(error_fn)
        self._cost_fn = None

    def cost(self, theta: jnp.ndarray, target_features: jnp.ndarray) -> jnp.ndarray:
        r"""
        Calculate the scalar error (cost) for a given set of parameters.
        
        This function computes the model features, calculates the residual against 
        the target data, and passes it through the defined error function. The 
        entire calculation is lazily compiled using ``jax.jit`` on the first call 
        to ensure the optimization loop runs as fast as possible.
        
        Parameters
        ----------
        theta : jax.numpy.ndarray
            A 1D array containing the current parameter values being tested by 
            the optimizer.
        target_features : jax.numpy.ndarray
            The extracted measurement data that the optimizer is trying to match.

        Returns
        -------
        jax.numpy.ndarray
            The scalar cost value representing the total error.
        """
        if self._cost_fn is None:
            self.logger.debug("Lazily compiling Frequentist cost function...")

            # Capture class attributes for the JAX closure at compile time
            reg_weight = self.tikhonov_lambda
            theta_0 = self.initial_theta
            p_ranges = self.param_ranges
            
            # 2. Compile the specific cost loop
            @jax.jit
            def cost_fn(theta, target_feats):
                model_features = self.model_features(theta)
                error = target_feats - model_features

                # Base cost
                base_cost = self._error_fn(error)
                base_cost = base_cost if jnp.isscalar(base_cost) else base_cost[0]

                if reg_weight > 0.0 and theta_0 is not None:
                    # Normalize the difference by the parameter ranges
                    normalized_diff = (theta - theta_0) / p_ranges
                    l2_penalty = reg_weight * jnp.sum(normalized_diff ** 2)
                    return base_cost + l2_penalty                

                cost_val = self._error_fn(error)
                return cost_val if jnp.isscalar(cost_val) else cost_val[0]
                
            self._cost_fn = cost_fn
            
        # 3. Evaluate the cached function
        return self._cost_fn(jnp.array(theta), jnp.array(target_features))