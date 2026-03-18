import json
import dataclasses
from copy import deepcopy

import jax.numpy as jnp
import equinox as eqx
import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution

from pmrf.core.field import field
from pmrf.utils import format_val, split_vectorized_distribution, serialize_distribution, deserialize_distribution, format_distribution
from pmrf.constants import MIN_PERCENTILE, MAX_PERCENTILE

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

        # A simple, single-valued parameter with a scale, initialized with a float
        p1 = prf.Parameter(value=1.0, scale=1e-12, name='C1')

        # This parameter can be used in calculations directly (scaling is done during casting)
        impedance = 1 / (2j * jnp.pi * 1e9 * p1)
        print(f"Impedance: {impedance}")

        # A parameter that is fixed and will not be optimized during a fit
        p2 = prf.Parameter(value=50.0, fixed=True, name='R2')

        # A parameter with a uniform distribution
        # The provided factory functions in pmrf.parameters are a convenient way to create these
        from pmrf.parameters import Uniform
        p3 = Uniform(0.9, 1.1, scale=1e-9, name='L1')
        print(f"Initial value of L1: {p3.value}") # initialized to the mean

        # More complicated parameters can be initialized with any "numpyro" distribution
        from numpyro.distributions import LogNormal
        p4 = prf.Parameter(value=10.0, distribution=LogNormal(10.0, 1.0))
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
    def min(self) -> jnp.ndarray:
        r"""
        The unscaled minimum value of the parameter's distribution.
        
        This uses a percentile for all distributions other than uniform.
        
        Returns
        -------
        jnp.ndarray
            The minimum value, or the current value if no distribution is set.
        """        
        if isinstance(self.distribution, dist.ImproperUniform):
            return jnp.full(self.shape, -jnp.inf)
        elif isinstance(self.distribution, dist.Delta):
            return self.distribution.v
        elif isinstance(self.distribution, dist.Uniform):
            return self.distribution.low
        elif self.distribution is not None:
            return self.distribution.icdf(MIN_PERCENTILE)
        
        return jnp.full(self.shape, -jnp.inf)

    @property
    def max(self) -> jnp.ndarray:
        r"""
        The unscaled maximum value of the parameter's distribution.
        
        This uses a percentile for all distributions other than uniform.
        
        Returns
        -------
        jnp.ndarray
            The maximum value, or the current value if no distribution is set.
        """              
        if isinstance(self.distribution, dist.ImproperUniform):
            return jnp.full(self.shape, jnp.inf)
        elif isinstance(self.distribution, dist.Delta):
            return self.distribution.v
        elif isinstance(self.distribution, dist.Uniform):
            return self.distribution.high
        elif self.distribution is not None:
            return self.distribution.icdf(MAX_PERCENTILE)
            
        return jnp.full(self.shape, jnp.inf)

    @property
    def mean(self) -> jnp.ndarray:
        r"""
        The unscaled mean value of the parameter's distribution.
        
        Returns
        -------
        jnp.ndarray
            The mean value, or the current value if no distribution is set.
        """        
        # ImproperUniform doesn't have a defined mean, so we fall back to the initialized value
        if isinstance(self.distribution, dist.ImproperUniform) or self.distribution is None:
            return self.value
        elif isinstance(self.distribution, dist.Delta):
            return self.distribution.v
            
        return self.distribution.mean
    
    def with_value(self, value: jnp.ndarray) -> 'Parameter':
        r"""
        Return a copy of the parameter with a new unscaled value.

        Parameters
        ----------
        value : jnp.ndarray
            The new unscaled value to set.

        Returns
        -------
        Parameter
            A copy of this object with ``value`` replaced.
        """
        return dataclasses.replace(self, value=value)

    def shifted(self, amount: jnp.ndarray) -> 'Parameter':
        r"""
        Returns a copy of this parameter with its mean shifted by an amonut.

        Parameters
        ----------
        amount : jnp.ndarray
            The amount to shift by. Can be positive or negative.

        Returns
        -------
        Parameter
            A copy of this object with ``value`` replaced.
        """
        new_value = self.value + amount
        new_distribution = deepcopy(self.distribution)
        if isinstance(new_distribution, dist.Uniform):
            new_distribution.low += amount
            new_distribution.high += amount
        elif isinstance(new_distribution, dist.Normal):
            new_distribution.loc += amount
        else:
            raise Exception("Can only call 'shifted' on a normal or uniform parameter")
        return dataclasses.replace(self, value=new_value, distribution=new_distribution)
    
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
        # Handle scalar / 0-d array
        if self.value.ndim == 0 and self.flat_names is None:
            return [self]
            
        # Flatten the value
        flat_val = jnp.ravel(self.value)
        
        # Split distribution if present
        if self.distribution is not None:
            dists_split = split_vectorized_distribution(self.distribution)
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
    
    def __repr__(self):
        # Build the representation dynamically
        # 'value' is always printed as it is the core of the Parameter
        args = [f"value={format_val(self.value)}"]
        
        # Only add attributes if they deviate from the default
        if self.scale != 1.0:
            args.append(f"scale={self.scale}")
            
        if self.fixed is not False:
            args.append(f"fixed={self.fixed}")
            
        if self.distribution is not None:
            args.append(f"distribution={format_distribution(self.distribution)}")
            
        if self.name is not None:
            args.append(f"name={repr(self.name)}")
            
        return f"Parameter({', '.join(args)})"
            
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
            "distribution": serialize_distribution(self.distribution),
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
            distribution=deserialize_distribution(d["distribution"]),
            fixed=d["fixed"],
            scale=d["scale"],
            name=d["name"]
        )