import json
import dataclasses
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import equinox as eqx
import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution

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
    # None of these are marked static so we can update them if we want to
    value: jnp.ndarray = field(converter=lambda x: jnp.asarray(x, dtype=jnp.float64))
    distribution: Distribution | None = field(default=None)
    fixed: bool = field(default=False)
    scale: float = field(default=1.0)
    name: str | None = field(default=None, static=True)
    flat_names: list[str] | None = field(default=None, static=True)
    
    @property
    def ndim(self) -> int:
        """
        The number of free dimensions for this parameter.
        
        Returns 0 if the parameter is fixed.
        """
        if self.fixed:
            return 0
        
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
            
        if jnp.isscalar(self.value):
            return -jnp.inf
        return jnp.array([-jnp.inf] * self.value.shape[0])
    
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
            
        if jnp.isscalar(self.value):
            return -jnp.inf
        return jnp.array([-jnp.inf] * self.value.shape[0])
    
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
        if jnp.isscalar(self.value):
            return [self]
        else:
            if self.distribution is not None:
                dists_split = _split_vectorized_distribution(self.distribution)
            else:
                dists_split = [None] * len(self.value)

            flat_names = self.flat_names if self.flat_names is not None else [f"{self.name}{separator}{i}" for i in range(len(self.value))] if self.name is not None else [None] * len(self.value)
            return [Parameter(value=val, distribution=p, fixed=self.fixed, scale=self.scale, name=flat_names[i]) for i, (val, p) in enumerate(zip(self.value, dists_split))]
        
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
    
    if n is not None and n != 1:
        low, high = jnp.array([low] * n), jnp.array([high] * n)
        if value is not None:
            value = jnp.array([value] * n)
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
        The percentage width to use to initialize the lower and upper bounds.
        Bounds are calculated as `mean +/- (perc * mean / 200)`.
    **kwargs
        Additional keyword arguments passed to the `Uniform` factory function.

    Returns
    -------
    Parameter
        The created Parameter object.
    """
    if isinstance(perc, Sequence):
        delta = [p * mean[i] / 200.0 for i, p in enumerate(perc)]
    else:
        delta = perc * mean / 200.0
    return Uniform(low=mean-delta, high=mean+delta, **kwargs)

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
    if n is not None and n != 1:
        mean, std = jnp.array([mean] * n), jnp.array([std] * n)
        if value is not None:
            value = jnp.array([value] * n)
    else:
        mean, std = jnp.array(mean), jnp.array(std)
    
    dists = dist.Normal(mean, std)
    values = jnp.array(mean) if value is None else value
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
    if isinstance(perc, Sequence):
        std = [p * mean[i] / 200.0 for i, p in enumerate(perc)]
    else:
        std = perc * mean / 200.0
    return Normal(mean=mean, std=std, **kwargs)

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
    if n is not None and n != 1:
        value = jnp.array([value] * n)
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
    if n is not None and n != 1:
        value = jnp.array([value] * n)
    else:
        value = jnp.array(value)
    return Parameter(value=value, **kwargs)

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

def asparam(x, **kwargs) -> Parameter:
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

def _split_vectorized_distribution(dist):
    r"""
    Split a 1D batch of univariate numpyro distributions into a list.

    Parameters
    ----------
    dist : numpyro.distributions.Distribution
        A distribution with ``event_shape == ()`` and 1D ``batch_shape``.

    Returns
    -------
    list[numpyro.distributions.Distribution]
        A list of per-element distributions.

    Raises
    ------
    ValueError
        If the distribution cannot be split (e.g., wrong shapes).
    """
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