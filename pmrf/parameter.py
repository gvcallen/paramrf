import jax
from dataclasses import dataclass
from scipy.stats import rv_continuous
import scipy.stats
import equinox as eqx

from equinox import field
import jax.numpy as jnp

from _typing import Array, Scalar


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
        return self.minimum if self.minimum is not None else self.dist.ppf(0.99)
    
    @property
    def upper(self):
        return self.maximum if self.maximum is not None else self.dist.ppf(0.99)


def uniform(min, max, **kwargs) -> 'Parameter':
    dist = scipy.stats.distributions.uniform(min, max-min)
    value = kwargs.pop('value', (max - min) / 2.0)
    return Parameter(dist=dist, value=value, **kwargs)

def norm(mean, std, **kwargs) -> 'Parameter':
    dist = scipy.stats.distributions.norm(mean, std)
    value = kwargs.pop('value', mean)
    return Parameter(dist=dist, value=value, **kwargs)

def fixed(value, **kwargs) -> 'Parameter':
    return Parameter(value=value, **kwargs)
