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
    """
    A base class for frequentist (classical) optimization methods.

    Provides a lazily compiled `cost()` function for backends to utilize.
    """
    def __init__(
        self, 
        model: Model, 
        *, 
        cost_kind: str | None = None, 
        error_fn: ArrayFuncT | list[ArrayFuncT] | eqx.Module | None = None, 
        **kwargs
    ):
        # Let base class consume features, output_path, etc.
        super().__init__(model, **kwargs)
        
        # Apply standard Python defaults/mutations safely to 'self'
        self.cost_kind = cost_kind or 'convolutional'
        
        if self.features is None:
            if self.cost_kind == 'convolutional':
                self.features = ['s', 's_mag']
            elif self.cost_kind == 'complex':
                self.features = ['s']
            elif self.cost_kind == 'magnitude':
                self.features = ['s_mag']
            else:
                self.features = [
                    feat for m, n in model.port_tuples 
                    for feat in (f's{m+1}{n+1}_re', f's{m+1}{n+1}_im')
                ]
                
        if error_fn is None:
            if self.cost_kind == 'convolutional':
                error_fn = CONVOLUTIONAL_ERROR
            elif len(self.features) > 1:
                error_fn = L2_ERROR
            else:
                error_fn = [l2_norm_ax0, mag_2_db]

        if not isinstance(error_fn, eqx.Module):
            if not isinstance(error_fn, list): 
                error_fn = [error_fn]
            error_fn = eqx.nn.Sequential([eqx.nn.Lambda(fn) for fn in error_fn])
            
        self._error_fn = eqx.filter_jit(error_fn)
        self._cost_fn = None

    def cost(self, theta: jnp.ndarray, target_features: jnp.ndarray) -> jnp.ndarray:
        """
        Lazily compiles and evaluates the cost function.
        
        This handles expanding 1D parameters (from SciPy) into the 2D format 
        expected by the vmapped feature extractor.
        """
        if self._cost_fn is None:
            self.logger.debug("Lazily compiling Frequentist cost function...")
            
            # 2. Compile the specific cost loop
            @jax.jit
            def cost_fn(theta, target_feats):
                error = target_feats - self.model_features(theta)
                cost_val = self._error_fn(error)
                return cost_val if jnp.isscalar(cost_val) else cost_val[0]
                
            self._cost_fn = cost_fn
            
        # 3. Evaluate the cached function
        return self._cost_fn(jnp.array(theta), jnp.array(target_features))