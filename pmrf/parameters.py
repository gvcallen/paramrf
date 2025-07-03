# from scipy.stats import rv_continuous
# import scipy.stats
import json
import dataclasses

import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution

from typing import Sequence
import equinox as eqx
import jax.numpy as jnp
from pmrf._util import field

class Parameter(eqx.Module):
    """
    **Overview**

    A container for a numerical parameter of a `Model`.

    This class serves as the fundamental building block for defining the
    tunable or fixed parameters within a **paramrf** `Model`. It is designed
    to be a flexible container that behaves like a standard numerical type
    (e.g., a `numpy.ndarray`) while holding additional metadata for model
    fitting and analysis.

    Key features include:
    - **Array-like Behavior**: A `Parameter` can be used in mathematical
      operations just like a numpy array.
    - **JAX and Equinox Compatibility**: As an `equinox.Module`, `Parameter`
      objects are JAX PyTrees, making them seamlessly compatible with JAX's
      transformations (`jit`, `grad`, etc.).
    - **Fit Control**: A parameter can be marked as `fixed`, which prevents
      it from being updated during a fitting process.
    - **Statistical Priors**: A `scipy.stats` distribution can be associated
      with a parameter to define a prior for Bayesian analyses or to set
      optimization bounds.

    **Example:**

    The following demonstrates how to create and use `Parameter` objects.

    ```python
    import pmrf as prf
    import numpy as np

    # A simple, single-valued parameter, initialized with a float
    p1 = prf.Parameter(value=1.0e-12, name='C1')

    # This parameter can be used in calculations directly
    impedance = 1 / (2j * np.pi * 1e9 * p1)
    print(f"Impedance: {impedance}")

    # A parameter that is fixed and will not be optimized during a fit
    p2 = prf.Parameter(value=50.0, fixed=True, name='Z0')

    # A parameter with a uniform distribution prior
    # Factory functions are a convenient way to create these
    p3 = prf.Uniform(min=0.9e-9, max=1.1e-9, name='L1')

    # The parameter's value is initialized to the mean of the distribution
    print(f"Initial value of L1: {p3.value}")
    ```
    """
    # Underlying values/dists (unscaled). Multiply by scale above to get to true value (done automatically when converting to array)
    # None of these are marked static so we can update them if we want to
    value: jnp.ndarray = field(converter=lambda x: jnp.asarray(x, dtype=jnp.float64))
    prior: Distribution | None = field(default=None)
    fixed: bool = field(default=False)
    scale: float = field(default=1.0)
    # TODO add bounds?
    name: str | None = field(default=None, static=True)
    
    @property
    def min(self) -> jnp.array:
        """The unscaled minimum value of the parameter's distribution (0.01 quantile).

        Returns:
            jnp.array: The minimum value, or -np.inf if no distribution is set.
        """
        if self.prior is not None:
            if isinstance(self.prior, dist.Uniform):
                return self.prior.low
            else:
                return self.prior.icdf(0.01)
            
        if jnp.isscalar(self.value):
            return -jnp.inf
        return jnp.array([-jnp.inf] * self.value.shape[0])
    
    @property
    def max(self) -> jnp.array:
        """The unscaled maximum value of the parameter's distribution (0.99 quantile).

        Returns:
            jnp.array: The maximum value, or np.inf if no distribution is set.
        """
        if self.prior is not None:
            if isinstance(self.prior, dist.Uniform):
                return self.prior.high
            else:
                return self.prior.icdf(0.99)
            
        if jnp.isscalar(self.value):
            return -jnp.inf
        return jnp.array([-jnp.inf] * self.value.shape[0])    
    
    def flattened(self, separator='_') -> 'Parameter | list[Parameter]':
        """Flattens self, either returning a single Parameter
        if the internal parameter is scalar, or a list.
        
        Returns:
            'Parameter' | list['Parameter']: The raveled parameters.
        """
        if jnp.isscalar(self.value):
            return self
        else:
            if self.prior is not None:
                priors_split = _split_vectorized_distribution(self.prior)
            else:
                priors_split = [None] * len(self.value)
            return [Parameter(value=val, prior=p, fixed=self.fixed, scale=self.scale, name=f"{self.name}{separator}{i}") for i, (val, p) in enumerate(zip(self.value, priors_split))]
        
    def as_fixed(self) -> 'Parameter':
        """Returns a fixed copy of self.

        Returns:
            Parameter: The new, fixed parameter.
        """
        return dataclasses.replace(self, fixed=True)
    
    def as_free(self) -> 'Parameter':
        """Returns a copy of self with fixed set to False.

        Returns:
            Parameter: The new, fixed parameter.
        """
        return dataclasses.replace(self, fixed=False)
    
    # Arithmetic and array conversions
    def __array__(self, dtype=None):
        return jnp.asarray(self.value * self.scale, dtype=dtype)
    
    def __jax_array__(self, dtype=None):
        return jnp.asarray(self.value * self.scale, dtype=dtype)
    
    def __len__(self):
        if len(self.value.shape) == 0:
            return 1 # e.g. for jax scalars
        return len(self.value)
    
    def __add__(self, other):
        return jnp.add(jnp.array(self), jnp.array(other))
    
    def __sub__(self, other):
        return jnp.subtract(jnp.array(self), jnp.array(other))
    
    def __mul__(self, other):
        return jnp.multiply(jnp.array(self), jnp.array(other))

    def __truediv__(self, other):
        return jnp.divide(jnp.array(self), jnp.array(other))

    def __radd__(self, other):
        return jnp.add(jnp.array(other), jnp.array(self))
    
    def __rsub__(self, other):
        return jnp.subtract(jnp.array(other), jnp.array(self))

    def __rmul__(self, other):
        return jnp.multiply(jnp.array(other), jnp.array(self))
    
    def __rtruediv__(self, other):
        return jnp.divide(jnp.array(other), jnp.array(self))
    
    def copy(self):
        return dataclasses.replace(self)
    
    # Serialization
    def to_json(self) -> str:
        d = {
            "value": self.value.tolist(),
            "prior": _serialize_distribution(self.prior),
            "fixed": self.fixed,
            "scale": self.scale,
            "name": self.name
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Parameter":
        d = json.loads(s)
        return cls(
            value=jnp.asarray(d["value"]),
            prior=_deserialize_distribution(d["prior"]),
            fixed=d["fixed"],
            scale=d["scale"],
            name=d["name"]
        )    
    
def Uniform(low: float | Sequence[float], high: float | Sequence[float], n: int | None = None, value=None, **kwargs) -> 'Parameter':
    """Creates a `Parameter` with a uniform prior distribution.

    Args:
        low (float | Sequence[float]): The lower value of the distribution. Can be a sequence for a multi-valued Parameter.
        upper (float | Sequence[float]): The upper value of the distribution. Can be a sequence for a multi-valued Parameter.
        n (int, optional): The number of identical parameters to create in an array. Defaults to None.
        value (optional): The initial value. If None, the midpoint of the distribution is used. Defaults to None.
        **kwargs: Additional keyword arguments passed to the `Parameter` constructor.

    Returns:
        Parameter: The created Parameter object.
    """
    
    if n is not None and n != 1:
        low, high = jnp.array([low] * n), jnp.array([high] * n)
        if value is not None:
            value = jnp.array([value] * n)
    else:
        low, high = jnp.array(low), jnp.array(high)
    
    priors = dist.Uniform(low, high)
    values = (low + high) / 2.0 if value is None else value
    return Parameter(value=values, prior=priors, **kwargs)

def Normal(mean: float | Sequence[float], std: float | Sequence[float], n: int | None = None, value=None, **kwargs) -> 'Parameter':
    """Creates a `Parameter` with a normal (Gaussian) prior distribution.

    Args:
        mean (float | Sequence[float]): The mean of the distribution. Can be a sequence for a multi-valued Parameter.
        std (float | Sequence[float]): The standard deviation of the distribution. Can be a sequence for a multi-valued Parameter.
        n (int, optional): The number of identical parameters to create in an array. Defaults to None.
        value (optional): The initial value. If None, the mean of the distribution is used. Defaults to None.
        **kwargs: Additional keyword arguments passed to the `Parameter` constructor.

    Returns:
        Parameter: The created Parameter object.
    """
    if n is not None and n != 1:
        mean, std = jnp.array([mean] * n), jnp.array([std] * n)
        if value is not None:
            value = jnp.array([value] * n)
    else:
        mean, std = jnp.array(mean), jnp.array(std)
    
    priors = dist.Normal(mean, std)
    values = jnp.array(mean) if value is None else value
    return Parameter(value=values, prior=priors, **kwargs)
    
def PercentNormal(mean: float | Sequence[float], perc: float | Sequence[float], **kwargs) -> 'Parameter':
    """Creates a `Parameter` with a normal (Gaussian) distribution and a percentage standard deviation.

    Args:
        mean (float | Sequence[float]): The mean of the distribution. Can be a sequence for a multi-valued Parameter.
        perc (float | Sequence[float]): The percentage width to use to initialize the standard deviation,
                                        assuming the percentage represents +-2*sigma (95% coverage).
                                        As an example, passing `5.0` results in `std = 0.025*mean`.
                                        Can be a sequence for a multi-valued Parameter.
        **kwargs:                       Additional keyword arguments passed to the `Normal` factory function.

    Returns:
        Parameter: The created Parameter object.
    """
    if isinstance(perc, Sequence):
        std = [p * mean[i] / 200.0 for i, p in enumerate(perc)]
    else:
        std = perc * mean / 200.0
    return Normal(mean=mean, std=std, **kwargs)

def Fixed(value, n: int | None = None, **kwargs) -> 'Parameter':
    """Creates a `Parameter` that is marked as fixed.

    Args:
        value: The value of the parameter.
        n (int, optional): The number of identical parameters to create in an array. Defaults to None.
        **kwargs: Additional keyword arguments passed to the `Parameter` constructor.

    Returns:
        Parameter: The created fixed Parameter object.
    """
    if n is not None and n != 1:
        value = jnp.array([value] * n)
    else:
        value = jnp.array(value)
    return Parameter(value=value, fixed=True, **kwargs)

def Free(value, n: int | None = None, **kwargs) -> 'Parameter':
    """Creates a `Parameter` that is marked as not fixed (i.e., free to vary).

    Args:
        value: The value of the parameter.
        n (int, optional): The number of identical parameters to create in an array. Defaults to None.
        **kwargs: Additional keyword arguments passed to the `Parameter` constructor.

    Returns:
        Parameter: The created free Parameter object.
    """
    if n is not None and n != 1:
        value = jnp.array([value] * n)
    else:
        value = jnp.array(value)
    return Parameter(value=value, **kwargs)

def is_param(x) -> bool:
    """Checks if an object is an instance of a `Parameter`.

    Args:
        x: The object to check.

    Returns:
        bool: `True` if the object is a Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter)

def is_valid_param(x) -> bool:
    """Checks if an object is an instance of a `Parameter`,
    and if its value is not None.

    Args:
        x: The object to check.

    Returns:
        bool: `True` if the object is a valid Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and x.value is not None

def is_free_param(x) -> bool:
    """Checks if an object is a non-fixed `Parameter`.

    Args:
        x: The object to check.

    Returns:
        bool: `True` if the object is a non-fixed Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and not x.fixed

def is_fixed_param(x) -> bool:
    """Checks if an object is a fixed `Parameter`.

    Args:
        x: The object to check.

    Returns:
        bool: `True` if the object is a fixed Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and x.fixed

def asparam(x, name=None) -> Parameter:
    """Ensures an object is a `Parameter`.

    If the object is already a `Parameter`, it is returned unchanged.
    Otherwise, the object is converted into a new `Parameter`.

    Args:
        x: The object to convert.
        name (str, optional): The name to assign to a newly created `Parameter`. Defaults to None.

    Returns:
        Parameter: The object as a `Parameter`.
    """
    if isinstance(x, Parameter):
        return x
    return Parameter(value=x, name=name)

def _split_vectorized_distribution(dist):
    if dist.event_shape != ():
        raise ValueError(f"Cannot split distribution with event_shape={dist.event_shape} (likely an Independent)")

    batch_size = dist.batch_shape[0] if len(dist.batch_shape) == 1 else None
    if batch_size is None:
        raise ValueError("Distribution is not a 1D batch of univariate distributions")

    # Get all init params used to construct the distribution
    dist_class = dist.__class__
    param_names = dist.arg_constraints.keys()
    param_values = {name: getattr(dist, name) for name in param_names}

    # Split each parameter
    split_params = [
        {name: param[i] if isinstance(param, jnp.ndarray) else param
         for name, param in param_values.items()}
        for i in range(batch_size)
    ]

    return [dist_class(**params) for params in split_params]

def _serialize_distribution(d: Distribution | None) -> dict | None:
    if d is None:
        return None
    return {
        "class": d.__class__.__name__,
        "params": {k: v.tolist() if isinstance(v, jnp.ndarray) else v for k, v in d.__dict__.items() if not k.startswith("_")}
    }

# Helper to deserialize a numpyro Distribution
def _deserialize_distribution(dct: dict | None) -> Distribution | None:
    if dct is None:
        return None
    cls = getattr(dist, dct["class"], None)
    if cls is None:
        raise ValueError(f"Unknown distribution class: {dct['class']}")
    return cls(**dct["params"])