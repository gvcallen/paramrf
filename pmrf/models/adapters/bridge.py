"""
Adapter models to bridge ParamRF with external software.
"""
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from functools import partial
import concurrent.futures

from pmrf.frequency import Frequency
from pmrf.models import Model
            
def _host_side_batched_lookup(dynamic_vals, freq, static_model, leaf_shapes):
    all_leaves = jax.tree.leaves((dynamic_vals, freq))
    
    is_batched = False
    if len(all_leaves) > 0:
        # leaf_shapes already contains the unbatched shapes for both dynamic and freq
        if all_leaves[0].ndim > len(leaf_shapes[0]):
            is_batched = True
    
    # 2. Single execution
    if not is_batched:
        model = eqx.combine(dynamic_vals, static_model)
        return np.array(model.compute(freq))

    # 3. Batched execution (Multithreading)
    batch_size = all_leaves[0].shape[0]
    
    def run_job(i):
        dynamic_i = jax.tree.map(lambda x: x[i], dynamic_vals)
        freq_i = jax.tree.map(lambda x: x[i], freq)
        model_i = eqx.combine(dynamic_i, static_model)
        return np.array(model_i.compute(freq_i))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_job, range(batch_size)))
        
    return np.stack(results)


class Host(Model):
    """
    An abstract base class for models where computation occurs on the Host (CPU/Python)
    rather than the Device (XLA/GPU).
    
    Inherit from this class (defining your parameters as usual)
    to create a ParamRF model that calls external software.
    
    These models break the JAX trace and cannot be JIT-compiled internally.
    Execution is threaded automatically when vmapped.
    
    NB: Any vmapping over model properties will automatically create a thread pool to execute
    the individual batches in parallel. If this feature is to be used, compute() must be written
    in a thread-safe manner, as it may be called from multiple threads that share the same memory.
    """
    
    @property
    def primary_property(self):
        raise NotImplementedError("Override 'primary_property' directly when creating a Host model")
    
    @property
    def number_of_ports(self):
        raise NotImplementedError("Override 'number_of_ports' directly when using a Host model")

    @abstractmethod
    def compute(self, freq: Frequency) -> np.ndarray | jnp.ndarray:
        """
        The user-defined host-side computation, to be overriden.
        
        This method receives concrete values (not tracers) and should return
        a numpy array of shape (n_freq, n_ports, n_ports).
        """
        raise NotImplementedError("Subclasses of `Host` must implement 'compute'.")

    def primary_matrix(self, freq: Frequency) -> jnp.ndarray:
        """
        The JIT-compatible entry point. 
        Automatically handles threading if called inside a vmap.
        """
        dynamic, static = eqx.partition(self, eqx.is_inexact_array)
        
        # Capture original shapes of params + freq to detect vmap batching later
        leaves = jax.tree.leaves((dynamic, freq))
        leaf_shapes = [x.shape for x in leaves]

        n_p, n_f = self.number_of_ports, freq.npoints
        result_shape = jax.ShapeDtypeStruct((n_f, n_p, n_p), jnp.complex128)
        
        # Bake static data and shapes into the callback
        cb = partial(_host_side_batched_lookup, static_model=static, leaf_shapes=leaf_shapes)
        
        # 'broadcast_all' ensures that if vmap is present, ALL arguments passed to 'cb'
        # (even those that weren't mapped) get the batch dimension added. 
        return jax.pure_callback(cb, result_shape, dynamic, freq, vmap_method='broadcast_all')
    
__all__ = ['Host']