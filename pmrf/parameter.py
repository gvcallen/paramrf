import jax
from dataclasses import dataclass
from scipy.stats import rv_continuous
import scipy.stats
import equinox as eqx

from equinox import field
import jax.numpy as jnp

from pmrf._typing import Array, Scalar


@dataclass
class Parameter:
    value: Scalar | Array = 0.0
    scale: float = 1.0
    fixed: bool = False
    minimum: float | None = None
    maximum: float | None = None
    dist: rv_continuous | None = None

    @property
    def lower(self):
        if not self.minimum is None:
            return self.minimum
        if not self.dist is None:
            return self.dist.ppf(0.01)
        return self.value
    
    @property
    def upper(self):
        if not self.maximum is None:
            return self.maximum
        if not self.dist is None:
            return self.dist.ppf(0.99)
        return self.value
    

def uniform(min, max, **kwargs) -> 'Parameter':
    dist = scipy.stats.distributions.uniform(min, max-min)
    value = kwargs.pop('value', (max + min) / 2.0)
    return Parameter(dist=dist, value=value, **kwargs)

def norm(mean, std, **kwargs) -> 'Parameter':
    dist = scipy.stats.distributions.norm(mean, std)
    value = kwargs.pop('value', mean)
    return Parameter(dist=dist, value=value, **kwargs)

def fixed(value, **kwargs) -> 'Parameter':
    return Parameter(value=value, **kwargs)
