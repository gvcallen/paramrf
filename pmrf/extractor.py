"""
Extractor helper class for extracting features such as S-parameters from ParamRF Models and scikit-rf Networks.
"""
from typing import Callable, Any
import operator
import re

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.model import Model
from pmrf.frequency import Frequency
from pmrf.constants import FeatureSpec

class Extractor(eqx.Module):
    def __call__(self, model: Model, frequency: Frequency):
        raise NotImplementedError


class NetworkProperty(Extractor):
    prop: str = eqx.field(static=True)
    ports: tuple[int, int] | None = eqx.field(default=None, static=True)
    subattrs: str | None = eqx.field(default=None, static=True)

    def __call__(self, model: Model, frequency: Frequency) -> jnp.ndarray:
        prop, ports = self.prop, self.ports
        
        if self.subattrs is not None:
            model = operator.attrgetter(self.subattrs)(model)
            
        data = getattr(model, prop)(frequency)
                
        if ports is not None:
            m, n = ports
            if m >= model.nports or n >= model.nports:
                raise IndexError(f"Property {prop}{m+1}{n+1} specified but model is a {model.nports}-port")
            data = data[..., m, n]
        
        return data
    
    @classmethod
    def from_alias(cls, alias: str) -> 'Extractor':
        fields = alias.split('.')
        if len(fields) > 1:
            # FIX: Join back into a dot-separated string for attrgetter
            subattrs = ".".join(fields[:-1]) 
            alias = fields[-1]
        else:
            subattrs = None

        match = re.match(r'^([a-zA-Z]+)(\d)?(\d)?(.*)$', alias)
        if not match:
            raise ValueError(f"Invalid feature alias format: '{alias}'")

        prop_prefix = match.group(1)
        port1 = match.group(2)
        port2 = match.group(3)
        prop_suffix = match.group(4)
        
        prop = prop_prefix + prop_suffix

        # Map 1-indexed string alias (e.g., S11) to 0-indexed port array slices
        if port1 is not None and port2 is not None:
            ports = (int(port1) - 1, int(port2) - 1)
        else:
            ports = None
        
        # FIX: Changed `property` to `prop`
        return cls(prop=prop, ports=ports, subattrs=subattrs)
    

class CallableExtractor(Extractor):
    fn: Callable[[Model, Frequency, Any], jnp.ndarray]
    args: Any = None

    def __call__(self, source: Model, freq: Frequency) -> jnp.ndarray:
        return self.fn(source, freq, self.args)


class StackedExtractor(Extractor):
    extractors: list[Extractor]
    axis: int = eqx.field(static=True, default=-1)

    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        results = [ext(model, freq) for ext in self.extractors]
        return jnp.stack(results, axis=self.axis)
    
    @classmethod
    def from_features(
        cls, 
        features: FeatureSpec, 
        axis: int = -1
    ) -> 'StackedExtractor':
        if isinstance(features, str) or callable(features):
            features = [features]
            
        extractors = []
        for spec in features:
            if callable(spec):
                extractors.append(CallableExtractor(fn=spec))
            else:
                extractors.append(NetworkProperty.from_alias(spec))
                
        return cls(extractors=extractors, axis=axis)
    

class ReflectionExtractor(Extractor):
    prop: str = eqx.field(static=True) # e.g., 's_db'
    subattrs: str | None = eqx.field(default=None, static=True)
    
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        if self.subattrs is not None:
            model = operator.attrgetter(self.subattrs)(model)
        matrix = getattr(model, self.prop)(freq) 
        return jax.vmap(jnp.diag)(matrix)

class TransmissionExtractor(Extractor):
    prop: str = eqx.field(static=True)
    subattrs: str | None = eqx.field(default=None, static=True)
    
    def __call__(self, model: Model, freq: Frequency) -> jnp.ndarray:
        if self.subattrs is not None:
            model = operator.attrgetter(self.subattrs)(model)

        matrix = getattr(model, self.prop)(freq)
        F, N, _ = matrix.shape
        mask = ~jnp.eye(N, dtype=bool)
        return jax.vmap(lambda m: m[mask])(matrix)    