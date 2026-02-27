import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from functools import partial
import concurrent.futures

from pmrf.models.adapter.base import Adapter
from pmrf.frequency import Frequency
            
def _host_side_batched_lookup(dynamic_vals, freq, static_model, leaf_shapes):
    """
    Host-side wrapper that handles both single and batched (vmapped) execution.
    """
    # 1. Detect if we are running in a batch (vmap) or single mode
    # We check all input leaves (params + freq). 'broadcast_all' ensures
    # that if we are vmapped, ALL leaves will have an extra leading dimension.
    all_leaves = jax.tree.leaves((dynamic_vals, freq))
    all_shapes = leaf_shapes + [x.shape for x in jax.tree.leaves(freq)]
    
    is_batched = False
    if len(all_leaves) > 0:
        # If the actual array has more dimensions than the original, it's a batch.
        if all_leaves[0].ndim > len(all_shapes[0]):
            is_batched = True
    
    # 2. Single execution (Standard behavior)
    if not is_batched:
        model = eqx.combine(dynamic_vals, static_model)
        # Ensure we return a standard numpy array to JAX
        return np.array(model.compute(freq))

    # 3. Batched execution (Multithreading)
    # vmap_method='broadcast_all' ensures all inputs share the same batch size
    batch_size = all_leaves[0].shape[0]
    
    def run_job(i):
        # a. Slice the dynamic params AND freq for the i-th job
        dynamic_i = jax.tree.map(lambda x: x[i], dynamic_vals)
        freq_i = jax.tree.map(lambda x: x[i], freq)
        
        # b. Reconstruct the i-th model
        model_i = eqx.combine(dynamic_i, static_model)
        
        # c. Run simulation
        return np.array(model_i.compute(freq_i))

    # Use ThreadPoolExecutor to run the external simulations in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(run_job, range(batch_size)))
        
    return np.stack(results)


class Host(Adapter):
    """
    A base class for models where computation occurs on the Host (CPU/Python)
    rather than the Device (XLA/GPU).
    
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

    def compute(self, freq: Frequency) -> np.ndarray | jnp.ndarray:
        """
        The user-defined host-side computation. 
        
        This method receives concrete values (not tracers) and should return
        a numpy array of shape (n_freq, n_ports, n_ports).
        """
        raise NotImplementedError("Subclasses of Host must implement 'compute'.")

    def primary(self, freq: Frequency) -> jnp.ndarray:
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