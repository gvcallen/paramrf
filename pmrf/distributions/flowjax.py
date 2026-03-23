import io
import base64
import importlib
from typing import Callable, Any

import jax
import equinox as eqx

from distreqx.distributions import AbstractSampleLogProbDistribution, AbstractProbDistribution

class FlowJAXDistribution(AbstractSampleLogProbDistribution, AbstractProbDistribution):
    """
    distreqx-native wrapper for FlowJAX's `Transformed` class.
    """
    flow: Any
    factory_fn: Callable | None = eqx.field(static=True, default=None)
    factory_kwargs: dict = eqx.field(static=True, default_factory=dict)

    def __init__(self, flow: Any, factory_fn: Callable = None, factory_kwargs: dict | None = None):
        self.flow = flow
        self.factory_fn = factory_fn
        self.factory_kwargs = factory_kwargs or {}

    @property
    def event_shape(self) -> tuple:
        return (self.flow.base_dist.shape[0],)

    @property
    def batch_shape(self) -> tuple:
        return ()

    def sample(self, key: jax.Array, sample_shape: tuple = ()) -> jax.Array:
        return self.flow.sample(key, sample_shape)

    def log_prob(self, value: jax.Array) -> jax.Array:
        return self.flow.log_prob(value)

    def icdf(self, u: jax.Array) -> jax.Array:
        z = jax.scipy.special.ndtri(u)
        return self.flow.bijection.transform(z)

    # --- Serialization ---

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
        
        return state

    def __setstate__(self, state: dict) -> None:
        """Unpack the state, dynamically import the factory, and rebuild."""
        flow_bytes = state.pop('_flow_bytes')
        mod_name = state.pop('_factory_module', None)
        func_name = state.pop('_factory_name', None)
        
        # Use object.__setattr__ because Equinox modules are frozen dataclasses
        for k, v in state.items():
            object.__setattr__(self, k, v)
        
        if mod_name and func_name:
            # 1. Dynamically import the external factory
            module = importlib.import_module(mod_name)
            factory_fn = getattr(module, func_name)
            object.__setattr__(self, 'factory_fn', factory_fn)
            
            # 2. Rebuild the perfectly clean skeleton natively
            empty_skeleton = factory_fn(**self.factory_kwargs)
            
            # 3. Pour the exact weights back into the skeleton
            raw_bytes = base64.b64decode(flow_bytes)
            buffer = io.BytesIO(raw_bytes)
            flow = eqx.tree_deserialise_leaves(buffer, empty_skeleton)
            object.__setattr__(self, 'flow', flow)
        else:
            raise RuntimeError("Deserialization failed: No factory function address found.")

    # --- Distreqx Abstract Stubs ---

    def mean(self) -> jax.Array:
        raise NotImplementedError("Analytic mean is not defined for FlowJAX distributions.")

    def variance(self) -> jax.Array:
        raise NotImplementedError("Analytic variance is not defined for FlowJAX distributions.")
        
    def stddev(self) -> jax.Array:
        raise NotImplementedError("Analytic stddev is not defined for FlowJAX distributions.")
        
    def median(self) -> jax.Array:
        raise NotImplementedError("Analytic median is not defined for FlowJAX distributions.")

    def mode(self) -> jax.Array:
        raise NotImplementedError("Analytic mode is not defined for FlowJAX distributions.")

    def cdf(self, value: jax.Array) -> jax.Array:
        raise NotImplementedError("Analytic CDF is not defined for FlowJAX distributions.")

    def log_cdf(self, value: jax.Array) -> jax.Array:
        raise NotImplementedError("Analytic log_cdf is not defined for FlowJAX distributions.")

    def survival_function(self, value: jax.Array) -> jax.Array:
        raise NotImplementedError("Analytic survival_function is not defined for FlowJAX distributions.")

    def log_survival_function(self, value: jax.Array) -> jax.Array:
        raise NotImplementedError("Analytic log_survival_function is not defined for FlowJAX distributions.")

    def entropy(self) -> jax.Array:
        raise NotImplementedError("Analytic entropy is not defined for FlowJAX distributions.")

    def kl_divergence(self, other_distribution) -> jax.Array:
        raise NotImplementedError("KL divergence is not defined for FlowJAX distributions.")