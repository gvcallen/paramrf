from scipy.stats import rv_continuous
import scipy.stats

from typing import Sequence
import equinox as eqx
import pmrf.numpy as np
from pmrf._misc import field

class Parameter(eqx.Module):
    # Underlying values/dists (unscaled). Multiply by scale above to get to true value (done automatically when converting to array)
    # None of these are marked static so we can update them if we want to
    value: np.ndarray = field(converter=lambda x: np.asarray(x, dtype=np.float64))
    dist: rv_continuous | list[rv_continuous] | None = field(default=None)
    fixed: bool = field(default=False)
    scale: np.ndarray = field(default=1.0, converter=np.asarray)
    # TODO add bounds?
    name: str | None = field(default=None, static=True)
    
    @property
    def min_unscaled(self) -> float | None:
        if not self.dist is None:
            return self.ppf_unscaled(0.01)
        return None
    
    @property
    def max_unscaled(self) -> float | None:
        if not self.dist is None:
            return self.ppf_unscaled(0.99)
        return None
    
    def ppf_unscaled(self, q) -> float:
        return self.dist.ppf(q)
    
    # Arithmetic and array conversions
    def __array__(self, dtype=None):
        return np.asarray(self.value * self.scale, dtype=dtype)
    
    def __jax_array__(self, dtype=None):
        return np.asarray(self.value * self.scale, dtype=dtype)
    
    def __len__(self):
        if len(self.value.shape) == 0:
            return 1 # e.g. for jax scalars
        return len(self.value)
    
    def __add__(self, other):
        return np.add(np.array(self), np.array(other))
    
    def __sub__(self, other):
        return np.subtract(np.array(self), np.array(other))
    
    def __mul__(self, other):
        return np.multiply(np.array(self), np.array(other))

    def __truediv__(self, other):
        return np.divide(np.array(self), np.array(other))

    def __radd__(self, other):
        return np.add(np.array(other), np.array(self))
    
    def __rsub__(self, other):
        return np.subtract(np.array(other), np.array(self))

    def __rmul__(self, other):
        return np.multiply(np.array(other), np.array(self))
    
    def __rtruediv__(self, other):
        return np.divide(np.array(other), np.array(self))

class ParameterSet:
    def __init__(self, parameters: list[Parameter] | None = None):
        self._parameters = parameters if not parameters is None else []
        
    def __len__(self):
        return len(self._parameters)
    
    def __iter__(self):
        return iter(self._parameters)    
        
    def append(self, parameter: Parameter):
        self._parameters.append(parameter)
        
    def values_unscaled(self) -> list:
        # param.value is unscaled
        return [param.value for param in self._parameters]
    
    def minimums_unscaled(self) -> list:
        return [param.min_unscaled for param in self._parameters]
    
    def maximums_unscaled(self) -> list:
        return [param.max_unscaled for param in self._parameters]    
        
    # def to_dict(self, scaled=False) -> dict:
    #     if scaled:
    #         return {param.name: param.value_scaled for param in self.parameters}
    #     else:
    #         return {param.name: param.value for param in self.parameters}
        
    # def update(self, values):
    #     for i, value in enumerate(values):
    #         self.parameters[i].value = value
    
    
def Uniform(min: float | Sequence[float], max: float | Sequence[float], n: int | None = None, **kwargs) -> 'Parameter':
    if isinstance(min, Sequence):
        dists = [scipy.stats.distributions.uniform(mini, maxi-mini) for mini, maxi in zip(min, max)]
        values = [(maxi + mini) / 2.0 for mini, maxi in zip(min, max)]
        return Parameter(value=values, dist=dists, **kwargs)
    else:
        return _make_n((max + min) / 2.0, dist=scipy.stats.distributions.uniform(min, max-min), n=n, **kwargs)

def Normal(mean: float | Sequence[float], std: float | Sequence[float], n: int | None = None, **kwargs) -> 'Parameter':
    if isinstance(min, Sequence):
        dists = [scipy.stats.distributions.norm(meani, stdi) for meani, stdi in zip(mean, std)]
        values = [meani for meani in mean]
        return Parameter(value=values, dist=dists, **kwargs)
    else:
        return _make_n(mean, dist=scipy.stats.distributions.norm(mean, std), n=n, **kwargs)

def Fixed(value, n: int | None = None, **kwargs) -> 'Parameter':
    return _make_n(value=value, fixed=True, n=n, **kwargs)

def Free(value, n: int | None = None, **kwargs) -> 'Parameter':
    return _make_n(value=value, fixed=False, n=n, **kwargs)

def is_param(x):
    return isinstance(x, Parameter)

def is_free_param(x):
    return isinstance(x, Parameter) and not x.fixed

def is_fixed_param(x):
    return isinstance(x, Parameter) and x.fixed

def asparam(x, name=None) -> Parameter:
    if isinstance(x, Parameter):
        return x
    return Parameter(value=x, name=name)

def _make_n(value, dist=None, n: int | None = None, **kwargs) -> 'Parameter':
    if n == 1 or n is None:
        return Parameter(value=value, dist=dist, **kwargs)
    else:
        return Parameter(value=[value]*n, dist=[dist]*n, **kwargs)