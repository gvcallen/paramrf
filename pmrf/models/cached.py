"""
pmrf.cached_model
=================

Utilities for wrapping and caching expensive simulations (e.g. CST/HFSS) within ParamRF.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from joblib import Memory
import os
from typing import Any

import pmrf as prf

# -----------------------------------------------------------------------------
# 1. Setup Joblib Memory
# -----------------------------------------------------------------------------
# We use a localized cache directory. 
# verbose=0 suppresses "Loading..." messages.
# mmap_mode='r' allows reading large arrays from disk without loading them fully into RAM.
CACHE_DIR = os.path.join(os.getcwd(), ".paramrf_cache")
memory = Memory(location=CACHE_DIR, verbose=0)


# -----------------------------------------------------------------------------
# 2. Define the Cached Executor
# -----------------------------------------------------------------------------
@memory.cache
def _run_cached_simulation(model: 'Cached', freq: prf.Frequency) -> np.ndarray:
    """
    This is the function that Joblib actually caches. 
    
    It takes the *entire* model instance as an argument. Joblib hashes the 
    model by pickling it. Since Equinox modules are frozen dataclasses, 
    two different instances with the same parameter values will produce 
    the same hash, which is exactly what we want.
    """
    # We explicitly convert the output to a numpy array to ensure it's 
    # pickle-safe and decoupled from JAX tracers during the save process.
    return np.array(model.simulate(freq))


# -----------------------------------------------------------------------------
# 3. The Cached Class
# -----------------------------------------------------------------------------
class Cached(prf.Model):
    """
    A base class for models that require expensive, non-JAX simulations 
    (e.g., CST, HFSS, FEKO) and need persistent disk caching.

    Usage
    -----
    1. Inherit from Cached.
    2. Implement the `simulate(self, freq)` method.
    3. Use the model as normal (s, z, y parameters).

    Attributes
    ----------
    cache_decimals : int
        The number of decimals to round parameters to before hashing.
        This prevents cache misses due to tiny floating point differences 
        (e.g., 10.0 vs 10.00000001). Default is 6.
    """
    
    # Configuration for caching sensitivity
    cache_decimals: int | None = eqx.field(default=None, static=True)
    
    @property
    def primary_property(self):
        return 's'    

    def simulate(self, freq: prf.Frequency) -> np.ndarray | jnp.ndarray:
        """
        The expensive simulation logic goes here.

        This method is only called if a cache miss occurs.
        
        Parameters
        ----------
        freq : prf.Frequency
            The frequency grid for the simulation.

        Returns
        -------
        np.ndarray | jnp.ndarray
            The primary data matrix (e.g., S-parameters) with shape (nf, n, n).
        """
        raise NotImplementedError("Subclasses of Cached must implement 'simulate'.")

    def primary(self, freq: prf.Frequency) -> jnp.ndarray:
        """
        The entry point for all parameter calculations.
        
        It intercepts the call, checks the disk cache via Joblib, 
        and only runs the simulation if necessary.
        """
        # Helper to round a leaf if it's a float-like array
        def round_floats(x):
            # NEW: Explicitly skip rounding if decimals is None
            if self.cache_decimals is None:
                return x
                
            if eqx.is_inexact_array(x):
                return jnp.round(x, decimals=self.cache_decimals)
            return x

        # We only round the *free* parameters to avoid touching static config
        # that might not be numerical.
        rounded_model = jax.tree.map(round_floats, self)

        # 2. Call the cached executor
        # We pass the 'rounded_model' so that the hash is generated from 
        # the stable values.
        result_numpy = _run_cached_simulation(rounded_model, freq)
        
        # 3. Cast back to JAX
        # The result comes back from disk/sim as a numpy array. 
        # We convert it to a JAX array so it works with the rest of ParamRF.
        return jnp.array(result_numpy)

    def clear_cache(self):
        """Clears the entire disk cache for this project."""
        memory.clear()