"""
pmrf.models.cached
=================

Utilities for wrapping and caching expensive simulations (e.g. CST/HFSS) within ParamRF.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from joblib import Memory
import os
from functools import partial

from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf._util import field

# -----------------------------------------------------------------------------
# 1. Setup Joblib Memory
# -----------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.getcwd(), ".paramrf_cache")
memory = Memory(location=CACHE_DIR, verbose=0)


# -----------------------------------------------------------------------------
# 2. Define the Cached Executor
# -----------------------------------------------------------------------------
# We keep the logic simple here: Joblib hashes 'flat_params' and 'freq'.
# It ignores 'model', which is only used to run the simulation if there's a miss.
@memory.cache(ignore=['model'])
def _run_cached_simulation(flat_params: np.ndarray, freq: Frequency, model: 'Simulated') -> np.ndarray:
    # Ensure we return a standard numpy array (not JAX)
    return np.array(model.simulate(freq))


# -----------------------------------------------------------------------------
# 3. The Callback Logic (The Host-Side Bridge)
# -----------------------------------------------------------------------------
def _host_side_cache_lookup(dynamic_vals, freq, static_model):
    """
    This function runs entirely in Python (on the CPU).
    All inputs here are concrete NumPy arrays, NOT JAX tracers.
    """
    # 1. Reconstruct the model instance
    # We combine the dynamic arrays (values) with the static structure (classes/metadata)
    model = eqx.combine(dynamic_vals, static_model)

    # 2. Get the flat parameters for the cache key
    # Since 'model' is now concrete, this returns concrete arrays.
    flat_params = np.array(model.flat_param_values())

    # 3. Rounding (if configured)
    if model.cache_decimals is not None:
        flat_params = np.round(flat_params, decimals=model.cache_decimals)

    # 4. Call the Joblib cached function
    # joblib will hash (flat_params, freq) and check disk.
    return _run_cached_simulation(flat_params, freq, model=model)


# -----------------------------------------------------------------------------
# 4. The Simulated Class
# -----------------------------------------------------------------------------
class Simulated(Model):
    """
    A base class for models that require expensive, non-JAX simulations 
    (e.g., CST, HFSS, FEKO) and need persistent disk caching.
    """
    
    cache_decimals: int | None = field(default=None, static=True)
    
    @property
    def primary_property(self):
        raise NotImplementedError("Override 'primary_property' directly when creating a Simulated model")
    
    @property
    def number_of_ports(self):
        raise NotImplementedError("Override 'number_of_ports' directly when using a Simulated model")

    def simulate(self, freq: Frequency) -> np.ndarray | jnp.ndarray:
        raise NotImplementedError("Subclasses of Simulated must implement 'simulate'.")

    def primary(self, freq: Frequency) -> jnp.ndarray:
        """
        The JIT-compatible entry point.
        """
        # 1. Partition the model
        # We split 'self' into dynamic parts (JAX arrays) and static parts (metadata).
        # - 'dynamic': passed as arguments to the callback (will be tracers in JIT).
        # - 'static': passed via closure (must be hashable/static).
        dynamic, static = self.partition()

        # 2. Define Output Shape
        # JAX needs to know the shape/dtype of the callback result *before* it runs.
        # We assume complex128 for S-parameters.
        # Note: len(freq) must be valid. If freq is a tracer with dynamic shape, this will fail.
        n_p = self.number_of_ports
        n_f = len(freq) 
        result_shape = jax.ShapeDtypeStruct((n_f, n_p, n_p), jnp.complex128)

        # 3. Create the callback
        # We use partial to bake the 'static' structure into the function, 
        # because pure_callback only accepts array-like arguments.
        cb = partial(_host_side_cache_lookup, static_model=static)

        # 4. Execute Callback
        # This tells JAX: "Go to Python, run 'cb(dynamic, freq)', and expect 'result_shape' back."
        return jax.pure_callback(cb, result_shape, dynamic, freq)

    def clear_cache(self):
        memory.clear()