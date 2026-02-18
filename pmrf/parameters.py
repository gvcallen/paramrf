import json
import dataclasses
from dataclasses import dataclass
from typing import Sequence
import warnings

import jax.numpy as jnp
import equinox as eqx
import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution

from pmrf.distributions.stacked import StackedDistribution
from pmrf._util import field, interp_distribution

MIN_PERCENTILE = 0.01
MAX_PERCENTILE = 0.99

class Parameter(eqx.Module):
    """
    A container for a parameter, usually used within a `Model`.

    This class serves as the fundamental building block for defining the
    tunable or fixed parameters within a **paramrf** `Model` and for fitting.
    It is designed to be a flexible container that behaves like a standard numerical type
    (e.g., a `numpy.ndarray`) while holding additional metadata for model
    fitting and analysis.

    Usage
    -----
    
    * Use in mathematical operations just like a JAX/numpy array.
    * ``Parameter`` objects are JAX PyTrees, compatible with JAX transformations (jit, grad).
    * Mark as ``fixed`` (honoured by fitting and sampling routines).
    * Associate distributions, specified as numpyro distributions (uniform, normal, etc.).

    Attributes
    ----------
    value : jnp.ndarray
        The underlying unscaled value. Automatically converted to a float64 array.
    distribution : numpyro.distributions.Distribution or None
        The prior distribution associated with this parameter.
    fixed : bool
        If True, the parameter is treated as a constant during optimization/sampling.
    scale : float
        A scaling factor. The effective value used in calculations is ``value * scale``.
    name : str or None
        An optional name for the parameter (marked as static).

    Examples
    --------
    .. code-block:: python

        import pmrf as prf
        import jax.numpy as jnp

        # A simple, single-valued parameter, initialized with a float
        p1 = prf.Parameter(value=1.0e-12, name='C1')

        # This parameter can be used in calculations directly (scaling is done during casting)
        impedance = 1 / (2j * jnp.pi * 1e9 * p1)
        print(f"Impedance: {impedance}")

        # A parameter that is fixed and will not be optimized during a fit
        p2 = prf.Parameter(value=50.0, fixed=True, name='Z0')

        # A parameter with a uniform distribution
        # Factory functions are a convenient way to create these
        p3 = prf.Uniform(min=0.9e-9, max=1.1e-9, name='L1')

        # The parameter's value is initialized to the mean of the distribution
        print(f"Initial value of L1: {p3.value}")
    """
    # Underlying values/dists (unscaled). Multiply by scale above to get to true value (done automatically when converting to array)
    value: jnp.ndarray = field(converter=lambda x: jnp.asarray(x, dtype=jnp.float64))
    distribution: Distribution | None = field(default=None)
    
    # Static (metadata) fields
    fixed: bool = field(default=False, static=True)
    scale: float = field(default=1.0, static=True)
    name: str | None = field(default=None, static=True)
    flat_names: list[str] | None = field(default=None, converter=lambda x: list(x) if x is not None else x, static=True)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The shape of this parameter.
        """
        return self.value.shape
    
    @property
    def size(self) -> int:
        """
        The number of dimensions for this parameter.
        """
        return self.value.size
    
    @property
    def min(self) -> jnp.array:
        r"""
        The unscaled minimum value of the parameter's distribution.

        This is determined by the `MIN_PERCENTILE` quantile of the distribution.

        Returns
        -------
        jnp.array
            The minimum value, or -inf if no distribution is set.
        """
        if self.distribution is not None:
            if isinstance(self.distribution, dist.Uniform):
                return self.distribution.low
            else:
                return self.distribution.icdf(MIN_PERCENTILE)
            
        if self.value.ndim == 0:
            return -jnp.inf
        return jnp.full(self.shape, -jnp.inf)
    
    @property
    def max(self) -> jnp.array:
        r"""
        The unscaled maximum value of the parameter's distribution.
        
        This is determined by the `MAX_PERCENTILE` quantile of the distribution.

        Returns
        -------
        jnp.array
            The maximum value, or inf if no distribution is set.
        """
        if self.distribution is not None:
            if isinstance(self.distribution, dist.Uniform):
                return self.distribution.high
            else:
                return self.distribution.icdf(MAX_PERCENTILE)
            
        if self.value.ndim == 0:
            return jnp.inf
        return jnp.full(self.shape, jnp.inf)
    
    def with_value(self, value: jnp.array) -> 'Parameter':
        r"""
        Return a copy of the parameter with a new unscaled value.

        Parameters
        ----------
        value : jnp.array
            The new unscaled value to set.

        Returns
        -------
        Parameter
            A copy of this object with ``value`` replaced.
        """
        return dataclasses.replace(self, value=value)
    
    def with_distribution(self, distribution: Distribution) -> 'Parameter':
        r"""
        Return a copy of the parameter with a new distribution.

        Parameters
        ----------
        distribution : numpyro.distributions.Distribution
            The distribution to associate with this parameter.

        Returns
        -------
        Parameter
            A copy of this object with ``distribution`` replaced.

        Raises
        ------
        Exception
            If ``dist`` is not a numpyro Distribution.
        """
        if not isinstance(distribution, Distribution):
            raise Exception('Only numpyro distributions are supported as parameter distributions')
        
        return dataclasses.replace(self, distribution=distribution)
    
    def flattened(self, separator='_') -> 'list[Parameter]':
        r"""
        Flatten self into a list of scalar Parameters.
        
        If the internal parameter is scalar, the list will contain self.
        Otherwise, the parameter is split (de-vectorized) if possible.
        
        Parameters
        ----------
        separator : str, optional, default='_'
            Separator used for naming split parameters (e.g., name_0, name_1).

        Returns
        -------
        list[Parameter]
            The list of individual parameters.

        Raises
        ------
        ValueError
            If any internal distributions cannot be de-vectorized.
        """
        # if jnp.isscalar(self.value):
        #     return [self]
        # else:
        #     if self.distribution is not None:
        #         dists_split = _split_vectorized_distribution(self.distribution)
        #     else:
        #         dists_split = [None] * len(self.value)

        #     flat_names = self.flat_names if self.flat_names is not None else [f"{self.name}{separator}{i}" for i in range(len(self.value))] if self.name is not None else [None] * len(self.value)
        #     return [Parameter(value=val, distribution=p, fixed=self.fixed, scale=self.scale, name=flat_names[i]) for i, (val, p) in enumerate(zip(self.value, dists_split))]
        
        # Handle scalar / 0-d array
        if self.value.ndim == 0 and self.flat_names is None:
            return [self]
            
        # Flatten the value
        flat_val = jnp.ravel(self.value)
        
        # Split distribution if present
        if self.distribution is not None:
            dists_split = _split_vectorized_distribution(self.distribution)
        else:
            dists_split = [None] * flat_val.size

        # Generate names
        flat_names = self.flat_names
        if flat_names is None:
            if self.name is not None:
                flat_names = [f"{self.name}{separator}{i}" for i in range(flat_val.size)]
            else:
                flat_names = [None] * flat_val.size
                
        return [
            Parameter(value=val, distribution=p, fixed=self.fixed, scale=self.scale, name=n) 
            for val, p, n in zip(flat_val, dists_split, flat_names)
        ]        
        
    def interpolated(self, x_old, x_new) -> 'Parameter':
        """
        Return a new parameter interpolated to a new domain.

        Interpolates both the value and the distribution parameters.

        Parameters
        ----------
        x_old : array_like
            The original domain coordinates.
        x_new : array_like
            The new domain coordinates.

        Returns
        -------
        Parameter
            The interpolated parameter.
        """
        value = jnp.interp(x_old, x_new, self.value)
        dist = interp_distribution(x_old, x_new, self.distribution)

        return Parameter(
            value=value,
            distribution=dist,
            fixed=self.fixed,
            scale=self.scale,
            name=self.name
        )        
        
    def as_fixed(self) -> 'Parameter':
        r"""
        Return a copy of self with ``fixed=True``.

        Returns
        -------
        Parameter
            The new, fixed parameter.
        """
        return dataclasses.replace(self, fixed=True)
    
    def as_free(self) -> 'Parameter':
        r"""
        Return a copy of self with ``fixed=False``.

        Returns
        -------
        Parameter
            The new, free parameter.
        """
        return dataclasses.replace(self, fixed=False)
    
    # Arithmetic and array conversions
    def __array__(self, dtype=None):
        r"""
        NumPy array interface.

        Parameters
        ----------
        dtype : Any, optional
            Desired dtype.

        Returns
        -------
        jnp.ndarray
            The scaled value as an array (``value * scale``).
        """
        return jnp.asarray(self.value * self.scale, dtype=dtype)
    
    def __jax_array__(self, dtype=None):
        r"""
        JAX array interface.

        Parameters
        ----------
        dtype : Any, optional
            Desired dtype.

        Returns
        -------
        jnp.ndarray
            The scaled value as an array (``value * scale``).
        """
        return jnp.asarray(self.value * self.scale, dtype=dtype)
    
    def __len__(self):
        r"""
        Length of the parameter value.

        Returns
        -------
        int
            ``1`` for scalars, otherwise ``len(value)``.
        """
        if len(self.value.shape) == 0:
            return 1 # e.g. for jax scalars
        return len(self.value)
    
    def __add__(self, other):
        r"""Elementwise addition."""
        return jnp.add(jnp.array(self), jnp.array(other))
    
    def __sub__(self, other):
        r"""Elementwise subtraction."""
        return jnp.subtract(jnp.array(self), jnp.array(other))
    
    def __mul__(self, other):
        r"""Elementwise multiplication."""
        return jnp.multiply(jnp.array(self), jnp.array(other))

    def __truediv__(self, other):
        r"""Elementwise true division."""
        return jnp.divide(jnp.array(self), jnp.array(other))

    def __radd__(self, other):
        r"""Reflected elementwise addition."""
        return jnp.add(jnp.array(other), jnp.array(self))
    
    def __rsub__(self, other):
        r"""Reflected elementwise subtraction."""
        return jnp.subtract(jnp.array(other), jnp.array(self))

    def __rmul__(self, other):
        r"""Reflected elementwise multiplication."""
        return jnp.multiply(jnp.array(other), jnp.array(self))
    
    def __rtruediv__(self, other):
        r"""Reflected elementwise true division."""
        return jnp.divide(jnp.array(other), jnp.array(self))
    
    def copy(self):
        r"""
        Return a shallow copy.

        Returns
        -------
        Parameter
            A copy created via ``dataclasses.replace``.
        """
        return dataclasses.replace(self)
    
     # Serialization
    def to_json(self) -> str:
        r"""
        Serialize the parameter to a JSON string.

        Returns
        -------
        str
            A JSON-formatted string containing value, distribution, fixed, scale, and name.
        """
        d = {
            "value": self.value.tolist(),
            "distribution": _serialize_distribution(self.distribution),
            "fixed": self.fixed,
            "scale": self.scale,
            "name": self.name
        }
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Parameter":
        r"""
        Deserialize a parameter from a JSON string.

        Parameters
        ----------
        s : str
            The JSON string produced by :meth:`to_json`.

        Returns
        -------
        Parameter
            A reconstructed :class:`Parameter` instance.
        """
        d = json.loads(s)
        return cls(
            value=jnp.asarray(d["value"]),
            distribution=_deserialize_distribution(d["distribution"]),
            fixed=d["fixed"],
            scale=d["scale"],
            name=d["name"]
        )

            
@dataclass
class ParameterGroup:
    r"""
    A metadata class that groups a set of named flat parameters and defines any relationships between them.

    Attributes
    ----------
    parameter_names : list[str]
        The names of the parameters included in this group.
    distribution : dist.Distribution or None
        An optional joint distribution over the flattened parameters.
    """
    param_names: list[str]
    distribution: dist.Distribution | None = field(default=None)
    
    def __init__(self, param_names: list[str] | dict[str, Parameter], distribution: dist.Distribution | None = None):
        r"""
        Construct a :class:`ParameterGroup`.

        Parameters
        ----------
        param_names : list[str] | dict[str, Parameter]
            The names of the flattened parameters (or a mapping to parameters).
        dist : numpyro.distributions.Distribution, optional
            An optional joint distribution over the flattened parameters.
        """
        self.param_names = param_names
        self.distribution = distribution
        
    @property
    def num_params(self):
        r"""
        Number of flattened parameters in the group.

        Returns
        -------
        int
            The count of names in ``parameter_names``.
        """
        return len(self.param_names)
            
    @property
    def min(self) -> jnp.array:
        r"""
        The unscaled minimum value of the parameter group's distribution.
        
        Determined by the `MIN_PERCENTILE` quantile.

        Returns
        -------
        jnp.array
            The minimum value, or -inf if no distribution is set.
        """
        if self.distribution is not None:
            if hasattr(self.distribution, 'min'):
                return self.distribution.min.reshape((self.num_params))
            elif hasattr(self.distribution, 'low'):
                return self.distribution.low.reshape((self.num_params))
            else:
                # TODO implement optimization to determine minima
                return self.distribution.icdf(jnp.array([MIN_PERCENTILE] * self.num_params))
            
        return jnp.array([-jnp.inf] * self.num_params)
    
    @property
    def max(self) -> jnp.array:
        r"""
        The unscaled maximum value of the parameter group's distribution.
        
        Determined by the `MAX_PERCENTILE` quantile.

        Returns
        -------
        jnp.array
            The maximum value, or inf if no distribution is set.
        """
        if self.distribution is not None:
            if hasattr(self.distribution, 'max'):
                return self.distribution.max.reshape((self.num_params))
            elif hasattr(self.distribution, 'high'):
                return self.distribution.high.reshape((self.num_params))
            else:
                # TODO implement optimization to determine maximum
                return self.distribution.icdf(jnp.array([MAX_PERCENTILE] * self.num_params))
            
        return jnp.array([jnp.inf] * self.num_params)
    
    def with_distribution(self, distribution: Distribution) -> 'Parameter':
        r"""
        Return a copy of the parameter group with a new distribution.

        Parameters
        ----------
        distribution : numpyro.distributions.Distribution
            The distribution to associate with this parameter.

        Returns
        -------
        Parameter
            A copy of this object with ``distribution`` replaced.

        Raises
        ------
        Exception
            If ``dist`` is not a numpyro Distribution.
        """
        if not isinstance(distribution, Distribution):
            raise Exception('Only numpyro distributions are supported as parameter distributions')
        
        return dataclasses.replace(self, distribution=distribution)
    
    
def Uniform(low: float | Sequence[float], high: float | Sequence[float], n: int | None = None, value=None, **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` with a uniform distribution.

    Parameters
    ----------
    low : float | Sequence[float]
        The lower value of the distribution. Can be a sequence for a multi-valued Parameter.
    high : float | Sequence[float]
        The upper value of the distribution. Can be a sequence for a multi-valued Parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    value : optional
        The initial value. If None, the midpoint of the distribution is used. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        low = jnp.broadcast_to(jnp.array(low), shape)
        high = jnp.broadcast_to(jnp.array(high), shape)
        if value is not None:
            value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        low, high = jnp.array(low), jnp.array(high)
    
    dists = dist.Uniform(low, high)
    values = (low + high) / 2.0 if value is None else value
    return Parameter(value=values, distribution=dists, **kwargs)

def PercentUniform(mean: float | Sequence[float], perc: float | Sequence[float], **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` with a uniform distribution defined by a percentage width.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean of the distribution. Can be a sequence for a multi-valued Parameter.
    perc : float | Sequence[float]
        The percentage deviation from the mean to either of the bounds.
        Bounds are calculated as `mean +/- (perc * mean / 200)`.
    **kwargs
        Additional keyword arguments passed to the `Uniform` factory function.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    warnings.warn(
        "PercentUniform is deprecated and will be removed in a future version. "
        "Please use RelativeUniform instead",
        category=DeprecationWarning,
        stacklevel=2
    )    
    
    if isinstance(perc, Sequence) or isinstance(perc, jnp.ndarray):
        delta = jnp.array(perc) * jnp.array(mean) / 200.0
    else:
        delta = perc * jnp.array(mean) / 200.0
    return Uniform(low=mean-delta, high=mean+delta, **kwargs)

def RelativeUniform(mean: float | Sequence[float], deviation_fraction: float | Sequence[float], **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` with a uniform distribution defined by a fractional deviation.

    The bounds are calculated as: `mean * (1 +/- deviation_fraction)`

    Parameters
    ----------
    mean : float | Sequence[float]
        The center (mean) of the distribution.
    deviation_fraction : float | Sequence[float]
        The relative radius of the distribution bounds as a fraction of the mean.
        e.g., 0.1 results in bounds of [0.9 * mean, 1.1 * mean].
    **kwargs
        Additional keyword arguments passed to the `Uniform` constructor.

    Returns
    -------
    Parameter
    """
    mean_arr = jnp.array(mean)
    frac_arr = jnp.array(deviation_fraction)
    
    # Calculate the absolute deviation (radius)
    # delta = 10% of mean
    delta = jnp.abs(mean_arr * frac_arr)
    
    return Uniform(low=mean_arr - delta, high=mean_arr + delta, **kwargs)

def CenteredUniform(mean: float | Sequence[float], half_width: float | Sequence[float], **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` with a uniform distribution.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean value of the distribution. Can be a sequence for a multi-valued Parameter.
    half_width : float | Sequence[float]
        The half-width value of the distribution. Can be a sequence for a multi-valued Parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    value : optional
        The initial value. If None, the midpoint of the distribution is used. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    low = mean - half_width
    high = mean + half_width
    
    return Uniform(low, high, **kwargs)

def Normal(mean: float | Sequence[float], std: float | Sequence[float], n: int | None = None, value=None, **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` with a normal (Gaussian) distribution.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean of the distribution. Can be a sequence for a multi-valued Parameter.
    std : float | Sequence[float]
        The standard deviation of the distribution. Can be a sequence for a multi-valued Parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    value : optional
        The initial value. If None, the mean of the distribution is used. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        mean = jnp.broadcast_to(jnp.array(mean), shape)
        std = jnp.broadcast_to(jnp.array(std), shape)
        if value is not None:
            value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        mean, std = jnp.array(mean), jnp.array(std)
    
    dists = dist.Normal(mean, std)
    values = mean if value is None else value
    return Parameter(value=values, distribution=dists, **kwargs)
    
def PercentNormal(mean: float | Sequence[float], perc: float | Sequence[float], **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` with a normal (Gaussian) distribution and a percentage standard deviation.

    Parameters
    ----------
    mean : float | Sequence[float]
        The mean of the distribution. Can be a sequence for a multi-valued Parameter.
    perc : float | Sequence[float]
        The percentage width to use to initialize the standard deviation,
        assuming the percentage represents +/- 2*sigma (95% coverage).
        As an example, passing `5.0` results in `std = 0.025 * mean`.
        Can be a sequence for a multi-valued Parameter.
    **kwargs
        Additional keyword arguments passed to the `Normal` factory function.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    warnings.warn(
        "PercentNormal is deprecated and will be removed in a future version. "
        "Please use RelativeNormal instead",
        category=DeprecationWarning,
        stacklevel=2
    )        
    
    if isinstance(perc, Sequence) or isinstance(perc, jnp.ndarray):
        std = jnp.array(perc) * jnp.array(mean) / 200.0
    else:
        std = perc * jnp.array(mean) / 200.0
    return Normal(mean=mean, std=std, **kwargs)

def RelativeNormal(mean: float | Sequence[float], std_fraction: float | Sequence[float], **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` with a normal distribution defined by a relative standard deviation.

    The scale (sigma) is calculated as: `mean * std_fraction`

    Parameters
    ----------
    mean : float | Sequence[float]
        The center (mean) of the distribution.
    std_fraction : float | Sequence[float]
        The standard deviation expressed as a fraction of the mean 
        (also known as the coefficient of variation).
        e.g., 0.1 results in a distribution with sigma = 0.1 * mean.
    **kwargs
        Additional keyword arguments passed to the `Normal` constructor.

    Returns
    -------
    Parameter
    """
    mean_arr = jnp.array(mean)
    frac_arr = jnp.array(std_fraction)
    
    # Calculate absolute standard deviation
    # sigma = 10% of mean
    sigma = jnp.abs(mean_arr * frac_arr)
    
    return Normal(loc=mean_arr, scale=sigma, **kwargs)

def Fixed(value, n: int | None = None, **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` that is marked as fixed.

    Parameters
    ----------
    value
        The value of the parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created fixed Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        value = jnp.array(value)
    return Parameter(value=value, fixed=True, **kwargs)

def Free(value, n: int | None = None, **kwargs) -> 'Parameter':
    r"""
    Create a `Parameter` that is marked as not fixed (i.e., free to vary).

    Parameters
    ----------
    value
        The value of the parameter.
    n : int, optional
        The number of identical parameters to create in an array. Defaults to None.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        The created free Parameter object.
    """
    if n is not None:
        shape = (n,) if isinstance(n, int) else n
        value = jnp.broadcast_to(jnp.array(value), shape)
    else:
        value = jnp.array(value)
    return Parameter(value=value, **kwargs)

def Stacked(parameters: Sequence[Parameter], name: str | None = None, **kwargs) -> Parameter:
    """
    Combine multiple scalar or identically-shaped Parameters into a single vectorized Parameter.
    
    This acts as the inverse of `Parameter.flattened()`.

    Parameters
    ----------
    parameters : Sequence[Parameter]
        The list/tuple of Parameter objects to stack.
    name : str, optional
        The overarching name for the new stacked parameter.
    **kwargs
        Additional arguments passed to the `Parameter` constructor.

    Returns
    -------
    Parameter
        A new parameter containing the stacked values and distributions.
    """
    if not parameters:
        raise ValueError("Cannot stack an empty sequence of parameters.")
        
    # 1. Stack the unscaled values
    values = jnp.stack([p.value for p in parameters])
    
    # 2. Combine distributions
    dists = [p.distribution for p in parameters]
    stacked_dist = _stack_vectorized_distributions(dists)
    
    # 3. Preserve or generate flat names
    flat_names = []
    for p in parameters:
        if p.flat_names is not None:
            flat_names.extend(p.flat_names)
        elif p.name is not None:
            flat_names.append(p.name)
        else:
            flat_names.append(None)
            
    # 4. Handle the 'fixed' flag
    # Note: Parameter.size evaluates `if self.fixed:`, meaning `fixed` must remain a scalar bool.
    fixed_flags = [p.fixed for p in parameters]
    if not all(f == fixed_flags[0] for f in fixed_flags):
        raise ValueError(
            "All parameters must have the exact same 'fixed' status to be stacked. "
            "Element-wise fixed arrays are not supported by the base Parameter class."
        )
        
    # 5. Handle scales (we DON'T allow heterogeneous scales)
    scales = [p.scale for p in parameters]
    if not all(s == scales[0] for s in scales):
        raise Exception("Cannot create a stacked Parameter with differing scales")
    
    scale = scales[0]
    return Parameter(
        value=values,
        distribution=stacked_dist,
        fixed=fixed_flags[0],
        scale=scale,
        name=name,
        flat_names=flat_names,
        **kwargs
    )

def is_param(x) -> bool:
    r"""
    Check if an object is an instance of a `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter)

def is_valid_param(x) -> bool:
    r"""
    Check if an object is an instance of a `Parameter` and if its value is not None.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a valid Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and x.value is not None

def is_free_param(x) -> bool:
    r"""
    Check if an object is a non-fixed `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a non-fixed Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and not x.fixed

def is_fixed_param(x) -> bool:
    r"""
    Check if an object is a fixed `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a fixed Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and x.fixed

def as_param(x, **kwargs) -> Parameter:
    r"""
    Ensure an object is a `Parameter`.

    If the object is already a `Parameter`, it is returned unchanged.
    Otherwise, the object is converted into a new `Parameter`.

    Parameters
    ----------
    x
        The object to convert.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor (e.g. `name`).

    Returns
    -------
    Parameter
        The object wrapped as a `Parameter`.
    """
    if isinstance(x, Parameter):
        return x
    return Parameter(value=x, **kwargs)

def _split_vectorized_distribution(d: Distribution) -> list[Distribution]:
    """
    Split an arbitrarily shaped batch of univariate numpyro distributions into a list of scalar distributions.
    
    Handles broadcasting of distribution parameters to the batch shape.
    
    Parameters
    ----------
    d : numpyro.distributions.Distribution
        A distribution with ``event_shape == ()`` and arbitrary ``batch_shape``.

    Returns
    -------
    list[numpyro.distributions.Distribution]
        A flat list of scalar distributions corresponding to the flattened batch.
    """
    if d.event_shape != ():
        raise ValueError(f"Cannot split distribution with event_shape={d.event_shape} (likely an Independent or Multivariate dist)")

    batch_shape = d.batch_shape
    if not batch_shape: # Scalar distribution
        return [d]

    # Calculate total size to verify flatten length
    total_size = 1
    for dim in batch_shape:
        total_size *= dim

    # Get all init params used to construct the distribution (e.g., 'loc', 'scale', 'low', 'high')
    # d.arg_constraints keys usually match the __init__ arguments
    param_names = d.arg_constraints.keys()
    
    # Extract current values of parameters
    param_values = {name: getattr(d, name) for name in param_names}

    # Broadcast all parameters to the distribution's batch_shape and flatten them
    flat_params = {}
    for name, val in param_values.items():
        val = jnp.asarray(val)
        # Broadcast the parameter value to the distribution's batch shape.
        # This handles cases where e.g. Normal(0, [1, 2]) has scalar loc and vector scale.
        try:
            val_broadcast = jnp.broadcast_to(val, batch_shape)
        except ValueError:
             # Fallback or error if shapes are strictly incompatible, though numpyro usually prevents this earlier
             raise ValueError(f"Parameter '{name}' with shape {val.shape} cannot be broadcast to batch_shape {batch_shape}")
             
        flat_params[name] = jnp.ravel(val_broadcast)

    # Reconstruct individual scalar distributions
    split_dists = []
    dist_class = d.__class__
    
    for i in range(total_size):
        # Extract the i-th scalar value for each parameter
        args = {name: vals[i] for name, vals in flat_params.items()}
        split_dists.append(dist_class(**args))

    return split_dists

def _stack_vectorized_distributions(dists: list[Distribution | None]) -> Distribution | None:
    """
    Combine a list of scalar numpyro distributions into a single batched distribution.
    
    Parameters
    ----------
    dists : list[numpyro.distributions.Distribution | None]
        A list of distributions to stack.

    Returns
    -------
    numpyro.distributions.Distribution | None
        The vectorized distribution, or None if no distributions were provided.
    """
    if all(d is None for d in dists):
        return None
    if any(d is None for d in dists):
        raise ValueError("Cannot stack a mix of parameters where some have distributions and others do not.")
        
    dist_classes = set(type(d) for d in dists)
    
    # Fast path: If they are all the exact same family, use native NumPyro batching
    if len(dist_classes) == 1:
        dist_cls = dists[0].__class__
        param_names = dists[0].arg_constraints.keys()
        stacked_kwargs = {name: jnp.stack([getattr(d, name) for d in dists]) for name in param_names}
        return dist_cls(**stacked_kwargs)
        
    # Flexible path: Use our custom meta-distribution for mixed types!
    return StackedDistribution(dists)

def _serialize_distribution(d: Distribution | None) -> dict | None:
    r"""
    Serialize a numpyro distribution to a lightweight dictionary.

    Parameters
    ----------
    d : numpyro.distributions.Distribution or None
        The distribution to serialize.

    Returns
    -------
    dict or None
        A dictionary with ``class`` and ``params`` keys, or ``None``.
    """
    if d is None:
        return None
    return {
        "class": d.__class__.__name__,
        "params": {k: v.tolist() if isinstance(v, jnp.ndarray) else v for k, v in d.__dict__.items() if not k.startswith("_")}
    }

# Helper to deserialize a numpyro Distribution
def _deserialize_distribution(dct: dict | None) -> Distribution | None:
    r"""
    Deserialize a numpyro distribution from a dictionary.

    Parameters
    ----------
    dct : dict or None
        A dictionary produced by :func:`_serialize_distribution`.

    Returns
    -------
    numpyro.distributions.Distribution or None
        The reconstructed distribution, or ``None``.
    
    Raises
    ------
    ValueError
        If the distribution class is unknown.
    """
    if dct is None:
        return None
    cls = getattr(dist, dct["class"], None)
    if cls is None:
        raise ValueError(f"Unknown distribution class: {dct['class']}")
    return cls(**dct["params"])