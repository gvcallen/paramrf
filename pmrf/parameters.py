from scipy.stats import rv_continuous
import scipy.stats

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
    dist: rv_continuous | None | list[rv_continuous | None] = field(default=None)
    fixed: bool = field(default=False)
    scale: float = field(default=1.0)
    # TODO add bounds?
    name: str | None = field(default=None, static=True)
    
    def __post_init__(self):
        if self.dist is None:
            if not jnp.isscalar(self.value):
                self.dist = [None] * len(self.value)
    
    @property
    def min(self) -> float:
        """The unscaled minimum value of the parameter's distribution (0.01 quantile).

        Returns:
            float | None: The minimum value, or -np.inf if no distribution is set.
        """
        if self.dist is not None:
            if self.dist.dist.name == "uniform":
                return self.dist.args[0]
            else:
                return self.ppf(0.01)
        return -jnp.inf
    
    @property
    def max(self) -> float:
        """The unscaled maximum value of the parameter's distribution (0.99 quantile).

        Returns:
            float: The maximum value, or np.inf if no distribution is set.
        """
        if self.dist is not None:
            if self.dist.dist.name == "uniform":
                return self.dist.args[0] + self.dist.args[1]
            else:
                return self.ppf(0.99)
        return jnp.inf
    
    def ppf(self, q) -> float:
        """The unscaled percent point function (inverse CDF) of the distribution.

        Args:
            q (float): The quantile to compute the value for.

        Returns:
            float: The value at the specified quantile.
        """
        return self.dist.ppf(q)
    
    def ravel(self):
        """Flattens self, either returning a single Parameter
        if the internal parameter is scalar, or a list.
        
        Returns:
            'Parameter' | list['Parameter']: The raveled parameters.
        """
        if jnp.isscalar(self.value):
            return self
        else:
            return [Parameter(value=val, dist=dst, fixed=self.fixed, scale=self.scale, name=f"{self.name}_{i}") for i, (val, dst) in enumerate(zip(self.value, self.dist))]
    
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
    
def Uniform(min: float | Sequence[float], max: float | Sequence[float], n: int | None = None, value=None, **kwargs) -> 'Parameter':
    """Creates a `Parameter` with a uniform distribution.

    Args:
        min (float | Sequence[float]): The minimum value of the distribution. Can be a sequence for a multi-valued Parameter.
        max (float | Sequence[float]): The maximum value of the distribution. Can be a sequence for a multi-valued Parameter.
        n (int, optional): The number of identical parameters to create in an array. Defaults to None.
        value (optional): The initial value. If None, the midpoint of the distribution is used. Defaults to None.
        **kwargs: Additional keyword arguments passed to the `Parameter` constructor.

    Returns:
        Parameter: The created Parameter object.
    """
    if isinstance(min, Sequence):
        dists = [scipy.stats.distributions.uniform(mini, maxi-mini) for mini, maxi in zip(min, max)]
        values = [(maxi + mini) / 2.0 for mini, maxi in zip(min, max)]
        return Parameter(value=values, dist=dists, **kwargs)
    else:
        value = value if value is not None else (max + min) / 2.0
        return _make_n(value, dist=scipy.stats.distributions.uniform(min, max-min), n=n, **kwargs)

def Normal(mean: float | Sequence[float], std: float | Sequence[float], n: int | None = None, value=None, **kwargs) -> 'Parameter':
    """Creates a `Parameter` with a normal (Gaussian) distribution.

    Args:
        mean (float | Sequence[float]): The mean of the distribution. Can be a sequence for a multi-valued Parameter.
        std (float | Sequence[float]): The standard deviation of the distribution. Can be a sequence for a multi-valued Parameter.
        n (int, optional): The number of identical parameters to create in an array. Defaults to None.
        value (optional): The initial value. If None, the mean of the distribution is used. Defaults to None.
        **kwargs: Additional keyword arguments passed to the `Parameter` constructor.

    Returns:
        Parameter: The created Parameter object.
    """
    if isinstance(mean, Sequence):
        dists = [scipy.stats.distributions.norm(meani, stdi) for meani, stdi in zip(mean, std)]
        values = [meani for meani in mean]
        return Parameter(value=values, dist=dists, **kwargs)
    else:
        value = value or mean
        return _make_n(value, dist=scipy.stats.distributions.norm(mean, std), n=n, **kwargs)
    
def PercentNormal(mean: float | Sequence[float], perc: float | Sequence[float], **kwargs) -> 'Parameter':
    """Creates a `Parameter` with a normal (Gaussian) distribution and a percentage standard deviation.

    Args:
        mean (float | Sequence[float]): The mean of the distribution. Can be a sequence for a multi-valued Parameter.
        perc (float | Sequence[float]): The percentage width to initialize the standard deviation with (e.g. `5.0` for `std = 0.025*mean`). Can be a sequence for a multi-valued Parameter.
        **kwargs: Additional keyword arguments passed to the `Normal` factory function.

    Returns:
        Parameter: The created Parameter object.
    """
    if isinstance(perc, Sequence):
        std = []
        for i, p in enumerate(perc):
            std.append(p * mean[i] / 200.0)
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
    return _make_n(value=value, fixed=True, n=n, **kwargs)

def Free(value, n: int | None = None, **kwargs) -> 'Parameter':
    """Creates a `Parameter` that is marked as not fixed (i.e., free to vary).

    Args:
        value: The value of the parameter.
        n (int, optional): The number of identical parameters to create in an array. Defaults to None.
        **kwargs: Additional keyword arguments passed to the `Parameter` constructor.

    Returns:
        Parameter: The created free Parameter object.
    """
    return _make_n(value=value, fixed=False, n=n, **kwargs)

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

def _make_n(value, dist=None, n: int | None = None, **kwargs) -> 'Parameter':
    if n == 1 or n is None:
        return Parameter(value=value, dist=dist, **kwargs)
    else:
        return Parameter(value=[value]*n, dist=[dist]*n, **kwargs)