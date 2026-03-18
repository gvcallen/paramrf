import io
import base64
import importlib
from typing import Callable, Any

import jax
import jax.numpy as jnp
import equinox as eqx
from numpyro.distributions import Distribution

class FlowJAXDistribution(Distribution):
    """
    JAX-native wrapper for FlowJax's `Transformed` class.
    """
    def __init__(self, flow: Any, factory_fn: Callable = None, factory_kwargs: dict | None = None):
        self.flow = flow
        # Store the pointer to the external function and its config
        self.factory_fn = factory_fn
        self.factory_kwargs = factory_kwargs or {}
        
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
        """Package the state perfectly for jsonpickle."""
        state = self.__dict__.copy()
        
        # 1. Safely extract ONLY the numerical arrays using Equinox
        buffer = io.BytesIO()
        eqx.tree_serialise_leaves(buffer, self.flow)
        state['_flow_bytes'] = base64.b64encode(buffer.getvalue()).decode('ascii')
        state.pop('flow', None)
        
        # 2. Save the ADDRESS of the external factory, not the function itself
        if self.factory_fn is not None:
            state['_factory_module'] = self.factory_fn.__module__
            state['_factory_name'] = self.factory_fn.__name__
        state.pop('factory_fn', None)
        
        # (self.factory_kwargs is a standard dict, so it stays in state natively)
        return state

    def __setstate__(self, state: dict) -> None:
        """Unpack the state, dynamically import the factory, and rebuild."""
        flow_bytes = state.pop('_flow_bytes')
        mod_name = state.pop('_factory_module', None)
        func_name = state.pop('_factory_name', None)
        
        self.__dict__.update(state)
        
        if mod_name and func_name:
            # 1. Dynamically import the external factory (e.g., from circuitREACH)
            # pmrf doesn't need a hard import statement for this to work!
            module = importlib.import_module(mod_name)
            factory_fn = getattr(module, func_name)
            self.factory_fn = factory_fn
            
            # 2. Rebuild the perfectly clean skeleton natively
            empty_skeleton = factory_fn(**self.factory_kwargs)
            
            # 3. Pour the exact weights back into the skeleton
            raw_bytes = base64.b64decode(flow_bytes)
            buffer = io.BytesIO(raw_bytes)
            self.flow = eqx.tree_deserialise_leaves(buffer, empty_skeleton)
        else:
            raise RuntimeError("Deserialization failed: No factory function address found.")