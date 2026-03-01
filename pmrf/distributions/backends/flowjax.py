import io
import base64
from typing import Callable, Any

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from numpyro.distributions import Distribution

# =============================================================================
# 1. THE WRAPPER CLASS
# =============================================================================

class FlowJAXDistribution(Distribution):
    """
    JAX-native wrapper for FlowJax's `Transformed` class.
    
    This class wraps a trained FlowJax normalizing flow so that it conforms to 
    the standard NumPyro Distribution interface. It natively supports serialization 
    via `jsonpickle` by delegating PyTree array serialization to Equinox.
    """
    def __init__(self, flow: Any, skeleton_fn: Callable[[], Any]):
        """
        Parameters
        ----------
        flow : Any
            The trained FlowJax `Transformed` distribution (an Equinox module).
        skeleton_fn : Callable[[], Any]
            A zero-argument closure that, when called, returns an uninitialized 
            replica of the flow's architecture. Used during deserialization.
        """
        self.flow = flow
        self.skeleton_fn = skeleton_fn
        
        event_shape = (self.flow.base_dist.shape[0],)
        super().__init__(batch_shape=(), event_shape=event_shape)
        
    def sample(self, key: jnp.ndarray, sample_shape: tuple = ()) -> jnp.ndarray:
        return self.flow.sample(key, sample_shape)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        return self.flow.log_prob(value)

    def icdf(self, u: jnp.ndarray) -> jnp.ndarray:
        z = jax.scipy.special.ndtri(u)
        return self.flow.bijection.transform(z)

    def __getstate__(self) -> dict:
        """Package the state natively for jsonpickle."""
        if self.skeleton_fn is None:
            return super().__getstate__()
        
        state = self.__dict__.copy()
        
        # Serialize the Equinox arrays to an in-memory byte buffer
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, self.flow)
        
        # Convert the binary buffer to a base64 string for JSON compatibility
        state['_flow_bytes'] = base64.b64encode(buffer.getvalue()).decode('ascii')
        
        # Remove the un-picklable Equinox module from the state dictionary
        state.pop('flow', None)
        
        return state

    def __setstate__(self, state: dict) -> None:
        """Unpack the state from jsonpickle."""
        if self.skeleton_fn is None:
            return super().__setstate__()
        
        flow_bytes = state.pop('_flow_bytes')
        
        # Restore standard attributes (Crucially, this restores self.skeleton_fn)
        self.__dict__.update(state)
        
        # Decode the base64 string back into a binary buffer
        raw_bytes = base64.b64decode(flow_bytes)
        buffer = io.BytesIO(raw_bytes)
        
        # Reconstruct the Equinox model using the restored closure
        empty_skeleton = self.skeleton_fn()
        self.flow = eqx.tree_deserialise_leaves(buffer, empty_skeleton)