from dataclasses import dataclass
from scipy.stats import rv_continuous
import scipy.stats

import pmrf.numpy as np

@dataclass
class Parameter:
    value: float = 0.0
    fixed: bool = False
    bounds: tuple[float | None, float | None] = (None, None)
    _scale: float = 1.0
    _dist: rv_continuous | None = None

    def __init__(self, value=0.0, fixed=False, bounds=(None, None), dist=None, scale=1.0):
        self.value = value*scale
        self.fixed = fixed
        self.bounds = (bounds[0] * scale if bounds[0] is not None else None, bounds[1] * scale if bounds[1] is not None else None)
        self._scale = scale
        self._dist = dist

    def ppf(self, q):
        return self._dist.ppf(q) * self.scale

    @property
    def lower(self) -> float | None:
        if not self.bounds[0] is None:
            return self.bounds[0]
        if not self.dist is None:
            return self.dist.ppf(0.01)
        return None
    
    @property
    def upper(self) -> float | None:
        if not self.bounds[1] is None:
            return self.bounds[1]
        if not self.dist is None:
            return self.dist.ppf(0.99)
        return None
    
def uniform(min, max, **kwargs) -> 'Parameter':
    dist = scipy.stats.distributions.uniform(min, max-min)
    value = kwargs.pop('value', (max + min) / 2.0)
    return Parameter(dist=dist, value=value, **kwargs)

def norm(mean, std, **kwargs) -> 'Parameter':
    dist = scipy.stats.distributions.norm(mean, std)
    value = kwargs.pop('value', mean)
    return Parameter(dist=dist, value=value, **kwargs)

def fixed(value, **kwargs) -> 'Parameter':
    return Parameter(value=value, fixed=True, **kwargs)

def varying(value, **kwargs) -> 'Parameter':
    return Parameter(value=value, **kwargs)