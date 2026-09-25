"""
Base class for RF models.
"""

from typing import Any, Callable, ClassVar, TypeVar, Union, TypeGuard
import functools
import inspect
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx
import skrf
import parax as prx

from pmrf.frequency import Frequency
from pmrf.rf import (
    a2s, s2a, s2y, y2s, s2z, z2s, y2z, z2y, a2y, y2a, a2z, z2a, s2mna, y2mna, z2mna, a2mna, mna2s,
    MNAStamp,
)
from pmrf.math import CONVERSION_LOOKUP
from pmrf.utils.type import is_overridden
from pmrf.utils import field, unwrap, unwrap_self
from pmrf.utils.tree import pytree_cached_property
from pmrf.distributions import AbstractDistribution
from pmrf.modules.base import Module, validate

T = TypeVar('T')

PRIMARY_DOMAINS = ('s', 'a', 'y', 'z', 'mna')
PRIMARY_METHODS = PRIMARY_DOMAINS + ('primary_matrix',)
PLOT_DOMAINS = ('s', 'a', 'y', 'z')
HUB_Z0 = 50.0 + 0.0j

# Classes that have already emitted the direct ``Model.build`` deprecation warning.
_BUILD_DEPRECATION_WARNED: set[type] = set()


def _z0_as_array(method: Callable) -> Callable:
    """Wrap a primary method so an explicit ``z0`` reaches it as an array.

    A list or scalar ``z0`` is converted once, here, for every model. ``None`` is
    passed through unchanged to a model with a native reference impedance, and
    raises for any other. A scalar becomes a 0-d array, which ``jnp.isscalar``
    still treats as a scalar.
    """
    # Position of ``z0`` among the arguments after ``self``.
    params = list(inspect.signature(method).parameters.values())[1:]
    z0_index = next(
        (i for i, p in enumerate(params) if p.name == 'z0' and p.kind == p.POSITIONAL_OR_KEYWORD),
        None,
    )

    def resolve_z0(model, z0):
        if z0 is not None:
            return jnp.asarray(z0)
        if not type(model).supports_native_z0:
            raise ValueError(
                f"{type(model).__name__} has no native reference impedance, so it cannot "
                "take z0=None. Pass a z0, such as z0=50.0."
            )
        return None

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if 'z0' in kwargs:
            kwargs['z0'] = resolve_z0(self, kwargs['z0'])
        elif z0_index is not None and len(args) > z0_index:
            args = list(args)
            args[z0_index] = resolve_z0(self, args[z0_index])
        return method(self, *args, **kwargs)
    return wrapper


def _wrap_primary(method: Callable) -> Callable:
    """Wrap a primary method: unwrap ``self``, JIT-compile, and convert ``z0``."""
    return _z0_as_array(eqx.filter_jit(unwrap_self(method)))


class Model(Module):
    """
    Base class for RF models.

    Derive from this class to define your own, custom model.

    This class should not be instantiated directly. It is created internally in ParamRF when models are
    built compositionally, or can be inherited from. When inheriting, at least one primary matrix method,
    such as :meth:`pmrf.Model.s`, :meth:`pmrf.Model.a`, :meth:`pmrf.Model.y`, :meth:`pmrf.Model.z`, 
    or :meth:`pmrf.Model.primary_matrix`, must be overridden. To implement a
    model by returning another model, inherit from
    :class:`pmrf.models.AbstractBuilder`. Legacy classes may still override
    :meth:`pmrf.Model.build` directly, but that interface is deprecated.

    The model is an Equinox `Module <https://docs.kidger.site/equinox/api/module/module/>`_
    (an immutable dataclass) and a JAX PyTree. Parameters are declared using standard dataclass
    field syntax and should be annotated with type :type:`pmrf.Param` and field specifier :func:`pmrf.param`.
    For more details on parameter definitions, see :mod:`pmrf.parameters`.

    This class is not marked abstract: it acts more like a mix-in than an ABC.

    Usage
    -----
    - Define new models by sub-classing the model and adding custom parameters and/or sub-models
    - Construct models by passing parameters and/or submodels to the initializer (like a dataclass).
    - Use :func:`pmrf.update` and methods such as :meth:`.terminated` and :meth:`.flipped` to create modified versions of your model.
    
    Methods & Properties Summary
    ----------------------------

    **Core Methods**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`s`                         Scattering (S) parameter matrix at impedance z0.
    :meth:`a`                         ABCD parameter matrix.
    :meth:`z`                         Impedance (Z) parameter matrix.
    :meth:`y`                         Admittance (Y) parameter matrix.
    :meth:`mna`                       Modified Nodal Analysis (MNA) stamp matrices.
    :meth:`primary_matrix`            Return the primary matrix. Can be overridden for dynamic dispatch.
    :attr:`primary_domain`            The domain of the primary matrix as a string (e.g. ``"s"``, ``"a"``).
    :meth:`build`                     Deprecated here; use :class:`pmrf.models.AbstractBuilder`.
    :meth:`expand`                    Expands the model's topology. Used for circuit model flattening.
    ================================= ====================================================================

    **Helper Methods**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :attr:`number_of_ports`           Number of ports.
    :attr:`nports`                    Alias of :attr:`number_of_ports`.
    :attr:`port_tuples`               All (m, n) port index pairs.
    ================================= ====================================================================

    **Model Transformation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`flipped`                   Return a version of the model with ports flipped.
    :meth:`renumbered`                Return a version of the model with ports renumbered.
    :meth:`terminated`                Return a new model terminated by another (e.g. load).
    ================================= ====================================================================

    **File & Conversion Utilities**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`to_skrf`                   Convert the model at frequencies to an :class:`skrf.Network`.
    :meth:`export_touchstone`         Export the model response to a Touchstone file.
    ================================= ====================================================================    

    Examples
    --------
    A general ``PiCLC`` network model:

    .. code-block:: python

        import jax.numpy as jnp
        import pmrf as prf        

        class PiCLC(prf.Model):
            C1: prf.Param
            L:  prf.Param
            C2: prf.Param

            def a(self, freq: prf.Frequency) -> jnp.ndarray:
                w = freq.w
                Y1, Y2, Y3 = (1j * w * self.C1), (1j * w * self.C2), 1 / (1j * w * self.L)
                return jnp.array([
                    [1 + Y2 / Y3,        1 / Y3],
                    [Y1 + Y2 + Y1*Y2/Y3, 1 + Y1 / Y3],
                ]).transpose(2, 0, 1)

    """
    #: Whether the model has a native reference impedance, so that ``s()`` accepts
    #: ``z0=None`` (ADR-0006). A model that sets it defaults to ``z0=None``.
    supports_native_z0: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for name in PRIMARY_METHODS:
            if name in cls.__dict__:
                original_method = cls.__dict__[name]
                wrapped_method = _wrap_primary(original_method)
                setattr(cls, name, wrapped_method)        
            
        # Dynamic methods such as s_mag and s_mn_mag
        def make_dynamic_method(prop_name, func):
            def dynamic_method(self, *args, **kwargs):
                matrix = getattr(self, prop_name)(*args, **kwargs)
                return func(matrix)
            return dynamic_method
            
        for prop in PLOT_DOMAINS:
            for suffix, lookup in CONVERSION_LOOKUP.items():
                func = lookup[1]
                
                # Base function (e.g. s_mag)
                func_name = f"{prop}_{suffix}"
                if not hasattr(cls, func_name):
                    m = make_dynamic_method(prop, func)
                    m._pmrf_auto = True
                    setattr(cls, func_name, m)
                
                # Indexed function (e.g. s_mn_mag)
                func_name_mn = f"{prop}_mn_{suffix}"
                if not hasattr(cls, func_name_mn):
                    m_mn = make_dynamic_method(f"{prop}_mn", func)
                    m_mn._pmrf_auto = True
                    setattr(cls, func_name_mn, m_mn)

    def __getattribute__(self, name: str):
        attribute = super().__getattribute__(name)
        if (
            name == 'build'
            and is_overridden(type(self), Model, 'build')
            and not getattr(type(self), '_pmrf_explicit_builder', False)
            and type(self) not in _BUILD_DEPRECATION_WARNED
        ):
            cls = type(self)

            def deprecated_build(*args, **kwargs):
                if cls not in _BUILD_DEPRECATION_WARNED:
                    _BUILD_DEPRECATION_WARNED.add(cls)
                    warnings.warn(
                        f"{cls.__name__}: Model.build() is deprecated when overridden "
                        "directly. Inherit from pmrf.models.AbstractBuilder instead, or, "
                        "for a composite with no parameters of its own, use a plain "
                        "function that returns the model.",
                        FutureWarning,
                        stacklevel=2,
                    )
                return attribute(*args, **kwargs)

            return deprecated_build
        return attribute

    # ---- Introspection properties --------------------------------------------------------
    
    @pytree_cached_property
    def number_of_ports(self) -> int:
        """Number of ports.

        Returns
        -------
        int
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().number_of_ports

        freq = Frequency(1, 2, 2)
        eval = jax.eval_shape(lambda: self.s(freq))
        return eval.shape[1]

    @pytree_cached_property
    def nports(self) -> int:
        """Alias of :attr:`number_of_ports`."""
        return self.number_of_ports
    
    @property
    def port_tuples(self) -> list[tuple[int, int]]:
        """All (m, n) port index pairs.

        Returns
        -------
        list[tuple[int, int]]
        """
        return [(y, x) for x in range(self.nports) for y in range(self.nports)]
    
    # ---- Core API -------------------------------------------------------------
    
    def build(self) -> 'Model':
        """Build the model (deprecated on direct ``Model`` subclasses).

        Inherit from :class:`pmrf.models.AbstractBuilder` to define a supported
        model-returning ``build()`` hook. This method remains temporarily
        available here for compatibility with existing classes that override it
        directly.

        Returns
        -------
        Model

        Raises
        ------
        NotImplementedError
            In the base class; override in derived classes to build
            a compositional representation.
        """     
        raise NotImplementedError
    
    def expand(self) -> tuple[list[tuple['Model', int]], list[list[tuple['Model', int]]]] | None:
        """
        Expands this model into its internal graph representation for circuit flattening.

        Used by `Circuit.flattened` to unpack composite models into a single flat
        netlist for a global matrix solve.

        :class:`pmrf.models.AbstractBuilder` delegates this to its built model, as
        does a direct ``Model.build()`` override, so user classes rarely need it.
        It is mainly for built-in composites such as :class:`pmrf.models.Cascade`
        and :class:`pmrf.models.Renumbered`.

        Returns
        -------
        tuple or None
            If the model is a composite or routing container, it returns a tuple of:
            - `port_mapping`: A list of length `nports` mapping each external port index 
              of this model to an internal `(Model, port_index)` tuple.
            - `internal_connections`: A list of sub-nodes (connections) to add to the 
              netlist. Each node is a list of `(Model, port_index)` tuples.
            
            If the model is a fundamental leaf component, it returns `None`.

        Examples
        --------
        A 2-port model holding an inductor and capacitor in series:

        >>> def expand(self):
        ...     L, C = self.inductor, self.capacitor
        ...     port_mapping = [(L, 0), (C, 1)]
        ...     internal_connections = [[(L, 1), (C, 0)]]
        ...     return port_mapping, internal_connections
        """
        if is_overridden(self.__class__, Model, 'build'):
            return self.build().expand()
                
        return None
    
    def primary_matrix(self, freq: Frequency, **kwargs) -> jnp.ndarray:
        """The primary matrix (e.g. ``s``, ``a`` etc.) as a function of frequency.

        The primary matrix represents the matrix returned by :attr:`pmrf.Model.primary_domain`,
        which is either overridden by sub-classes, or is the first property directly overridden
        out of :meth:`pmrf.Model.s`, :meth:`pmrf.Model.a`, :meth:`pmrf.Model.y`, :meth:`pmrf.Model.z`
        (in that order), unless :meth:``pmrf.Model.build`` is overridden, in which case the primary matrix
        of the built model is returned.
        
        This method can also be overridden itself in order to dynamically
        implement one of the matrices as opposed to overriding it explicitly. 
        
        If this method is called and `self.primary_domain` is 's',
        then 'z0' should be passed in `kwargs`.
        
        Parameters
        ----------
        freq : Frequency
            Frequency grid.
        kwargs
            Keyword arguments forwarded to the primary matrix function, such as z0.

        Returns
        -------
        jnp.ndarray

        Raises
        ------
        NotImplementedError
            If no primary property is overridden.
        """      
        primary_domain = self.primary_domain
        return getattr(self, primary_domain)(freq, **kwargs)

    @property
    def primary_domain(self) -> str:
        """The primary domain (e.g. ``"s"``, ``"a"``) as a string.

        The primary property is the first overridden among
        :data:`PRIMARY_DOMAINS`, unless ``build`` is overridden,
        in which case the primary property of the built model is returned.

        Returns
        -------
        str

        Raises
        ------
        NotImplementedError
            If no primary property is overridden.
        """
        prioritized = () # for future expansion
        unprioritized = tuple(p for p in PRIMARY_DOMAINS if p not in prioritized)

        if is_overridden(type(self), Model, 'build'):
            return self.build().primary_domain
        
        for property in prioritized:
            if is_overridden(type(self), Model, property):
                return property
        for property in unprioritized:
            if is_overridden(type(self), Model, property):
                return property
        raise NotImplementedError(f"No primary properties in {PRIMARY_DOMAINS} are overridden, which are the only ones supported")     
    
    @_wrap_primary
    def s(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        """Scattering parameter matrix at port impedance z0.

        If a different parameter type (a, z, y, mna) is primary, this converts it to S.
        
        To convert between port impedances, use :meth:`pmrf.rf.renormalize_s`.
        
        Derived classes that implement S-parameters should use the **power wave**
        definition. Convert other definitions (such as traveling waves) with
        :meth:`pmrf.rf.s2s`.

        Parameters
        ----------
        frequency : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            S-parameter matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().s(frequency, z0=z0)

        primary_domain = self.primary_domain
        kwargs = {'z0': z0} if primary_domain == 's' else {}
        val = self.primary_matrix(frequency, **kwargs)

        if primary_domain == 's':
            return val
        elif primary_domain == 'a':
            return a2s(val, z0)
        elif primary_domain == 'z':
            return z2s(val, z0)
        elif primary_domain == 'y':
            return y2s(val, z0)
        elif primary_domain == 'mna':
            return mna2s(val, z0)

        raise NotImplementedError(f"Conversion from '{primary_domain}' to 's' is not implemented.")
    
    @eqx.filter_jit
    @unwrap_self
    def a(self, frequency: Frequency) -> jnp.ndarray:
        """ABCD parameter matrix.

        If a different parameter type is primary, this converts it to A.

        Parameters
        ----------
        frequency : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            ABCD matrix with shape ``(nf, 2, 2)``.
        """        
        if is_overridden(type(self), Model, 'build'):
            return self.build().a(frequency)

        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(frequency, **kwargs)

        if primary_domain == 'a':
            return val
        elif primary_domain == 's':
            return s2a(val, z0=HUB_Z0)
        elif primary_domain == 'z':
            return z2a(val)
        elif primary_domain == 'y':
            return y2a(val)
        else:
            raise NotImplementedError(f"Conversion from '{primary_domain}' to 'a' is not implemented.")

    @eqx.filter_jit
    @unwrap_self
    def z(self, frequency: Frequency) -> jnp.ndarray:
        """Impedance (Z) parameter matrix.

        If a different parameter type is primary, this converts it to Z.

        Parameters
        ----------
        frequency : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            Z matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().z(frequency)

        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(frequency, **kwargs)

        if primary_domain == 'z':
            return val
        elif primary_domain == 's':
            return s2z(val, z0=HUB_Z0)
        elif primary_domain == 'a':
            return a2z(val)
        elif primary_domain == 'y':
            return y2z(val)
        else:
            raise NotImplementedError(f"Conversion from '{primary_domain}' to 'z' is not implemented.")

    @eqx.filter_jit
    @unwrap_self
    def y(self, frequency: Frequency) -> jnp.ndarray:
        """Admittance (Y) parameter matrix.

        If a different parameter type is primary, this converts it to Y.

        Parameters
        ----------
        frequency : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            Y matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().y(frequency)

        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(frequency, **kwargs)

        if primary_domain == 'y':
            return val
        elif primary_domain == 's':
            return s2y(val, HUB_Z0)
        elif primary_domain == 'a':
            return a2y(val)
        elif primary_domain == 'z':
            return z2y(val)
        else:
            raise NotImplementedError(f"Conversion from '{primary_domain}' to 'y' is not implemented.")
        
    @eqx.filter_jit
    @unwrap_self
    def mna(self, frequency: Frequency) -> MNAStamp:
        """
        (experimental) Modified Nodal Analysis (MNA) stamp.

        Can be overridden in sub-classes.
        
        Otherwise converts from another domain (`y2mna`, `z2mna`, etc.). Y is
        preferred because it gives the sparsest stamp; other domains add auxiliary
        variables.
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().mna(frequency)

        primary_domain = self.primary_domain
        
        if primary_domain == 'mna':
            return self.primary_matrix(frequency)
            
        # We prioritize y, z and a for sparsity, assuming the caller
        # has created a numerically stable implementation.
        if primary_domain == 'y' or is_overridden(type(self), Model, 'y'):
            return y2mna(self.y(frequency))
        
        if primary_domain == 'z' or is_overridden(type(self), Model, 'z'):
            return z2mna(self.z(frequency))
        
        if primary_domain == 'a' or is_overridden(type(self), Model, 'a'):
            return a2mna(self.a(frequency))
        
        if primary_domain == 's' or is_overridden(type(self), Model, 's'):
            return s2mna(self.s(frequency, z0=HUB_Z0), z0=HUB_Z0)
            
        return y2mna(self.y(frequency))
    
    # ---- Magic methods and copying --------------------------------------------------

    def __getattr__(self, name: str):
        """
        Dynamic dispatch for scikit-rf plotting methods.
        
        Captures calls like `model.plot_s_db(freq)` and redirects them 
        to `model.to_skrf(freq).plot_s_db()`.
        """
        if name.startswith('plot_'):
            def plotter(freq: Frequency, *args, **kwargs):
                ntwk = self.to_skrf(freq)
                
                if not hasattr(ntwk, name):
                    raise AttributeError(f"scikit-rf Network object has no attribute '{name}'")
                
                return getattr(ntwk, name)(*args, **kwargs)
            return plotter
            
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")    
    
    def __pow__(self, other: 'Model') -> 'Model':
        """Cascade operator `**`."""    
        if other.nports == self.nports:    
            return self.cascaded(other)
        else:
            return self.terminated(other)
    
    def __matmul__(self, other: 'Model') -> 'Model':
        """Termination operator `@`."""        
        return self.terminated(other)
    
    def cascaded(self, other, **kwargs) -> 'Model':
        """Cascade this model with another, returning a new model.
        
        See :class:`pmrf.models.composite.interconnected.Cascade`.

        Returns
        -------
        Model
        """
        from pmrf.models import Cascade
        return Cascade([self, other], **kwargs)
        
    def flipped(self, **kwargs) -> 'Model':
        """Return a version of the model with ports flipped.
        
        See :class:`pmrf.models.composite.transformed.Flipped`.

        Returns
        -------
        Model
        """
        from pmrf.models import Flipped
        if isinstance(self, Flipped):
            return self.model
        return Flipped(self, **kwargs)

    def renumbered(self, from_ports: tuple[int], to_ports: tuple[int]= None, **kwargs) -> 'Model':
        """Return a version of the model with ports renumbered.
        
        See :class:`pmrf.models.composite.transformed.Renumbered`.

        Parameters
        ----------
        from_ports : tuple[int]
            The original port indices that map to `to_ports`.
        to_ports : tuple[int]
            The new port indices.
            
        Returns
        -------
        Model
        """
        from pmrf.models import Renumbered
        return Renumbered(self, from_ports, to_ports, **kwargs)
    
    def terminated(self, other: 'Model' = 'short', **kwargs) -> 'Model':
        """Terminate this model in another, returning a new model.
        
        See :class:`pmrf.models.composite.transformed.Terminated`.

        Parameters
        ----------
        other : Model | str, optional
            The model to terminate this one in. Can be literals 'short', 'open'
            or any model with half the ports of this one.
            Defaults to a 'short'.

        Returns
        -------
        Model
        """
        from pmrf.models import Short, Open, Terminated

        if isinstance(other, str):
            if other == 'short':
                other = Short()
            elif other == 'open':
                other = Open()
            else:
                raise ValueError(f"Unknown load alias {other} received in 'Model.terminated()'")

        other = other or Short()
        return Terminated(self, other, **kwargs)
    
    # ---- File and conversion utilities  --------------------------------------------------            
    
    def to_skrf(self, frequency: Frequency | Any, z0: ArrayLike | None = None, sigma=0.0, **kwargs) -> skrf.Network:
        """Convert the model at frequencies to an :class:`skrf.Network`.

        The active primary property (``self.primary_domain``) is used.

        Parameters
        ----------
        frequency : pmrf.frequency.Frequency | skrf.Frequency
            Frequency grid.
        z0 : ArrayLike, optional
            The reference impedance, scalar or per-port. Defaults to the model's
            native reference impedance, such as a :class:`pmrf.models.Circuit`'s
            Port impedances, or to 50 Ω for a model without one. The same value
            is used for the S-parameters and for ``Network.z0``.
        sigma : float, default=0.0
            If nonzero, add complex Gaussian noise with stdev ``sigma`` to ``s``.
        **kwargs
            Forwarded to :class:`skrf.Network` constructor. ``port_names`` defaults
            to a :class:`pmrf.models.Circuit`'s Port names, if any Port is named.

        Returns
        -------
        skrf.Network
        """
        import skrf
        import numpy as np

        if isinstance(frequency, Frequency):
            model_freq = frequency
            measured_freq = frequency.to_skrf()
        else:
            model_freq = Frequency.from_skrf(frequency)
            measured_freq = frequency
        
        defaults = _skrf_defaults(self)
        if z0 is None:
            z0 = defaults['z0']
        s_matrix = self.s(model_freq, z0=z0)
        
        if 'port_names' in defaults:
            kwargs.setdefault('port_names', defaults['port_names'])
        kwargs.update({
            's': np.array(s_matrix),
            'frequency': measured_freq,
            'z0': z0,
        })
        ntwk = skrf.Network(**kwargs)
        if sigma != 0.0:
            ntwk.s += (np.random.normal(0, sigma, ntwk.s.shape) + 1j * np.random.normal(0, sigma, ntwk.s.shape))
        return ntwk        
    
    def export_touchstone(self, filename: str, frequency: Frequency | Any, sigma: float = 0.0, **skrf_kwargs):
        """Export the model response to a Touchstone file via scikit-rf.

        Parameters
        ----------
        filename : str
        frequency : Frequency | skrf.Frequency
        sigma : float, default=0.0
            Additive complex noise std for S-parameters.
        **skrf_kwargs
            Forwarded to :meth:`skrf.Network.write_touchstone`.

        Returns
        -------
        Any
            Return value of ``Network.write_touchstone``.
        """
        if not isinstance(filename, str):
            raise Exception('Filename must be a string')
        
        ntwk = self.to_skrf(frequency, sigma=sigma)
        return ntwk.write_touchstone(filename, **skrf_kwargs)
    

def _skrf_defaults(model: Model) -> dict[str, Any]:
    """The :class:`skrf.Network` arguments :meth:`Model.to_skrf` derives from a model.

    ``z0`` is the model's native reference impedance (ADR-0006): each external
    Port's ``z0`` for a :class:`~pmrf.models.Circuit`, and its own ``z0`` for a
    :class:`~pmrf.models.Port`. A model without one gets 50 Ω.

    ``port_names`` is set for a Circuit with at least one named Port: each Port's
    ``name`` in port order, or its 1-based index for an unnamed Port.
    """
    from pmrf.models.composite.interconnected.circuit.circuit import Circuit
    from pmrf.models.components.ideal import Port

    model = unwrap(model)
    if isinstance(model, Circuit):
        ports = model.ports
        defaults = {'z0': np.array([np.asarray(port.z0) for port in ports])}
        if any(port.name is not None for port in ports):
            defaults['port_names'] = [
                str(i + 1) if port.name is None else port.name
                for i, port in enumerate(ports)
            ]
        return defaults
    elif isinstance(model, Port):
        return {'z0': np.asarray(model.z0)}
    else:
        return {'z0': 50.0}


def is_model(x: Any) -> TypeGuard[Model]:
    """
    Returns if `x` is an instance of :class:`pmrf.Model`.
    """
    return isinstance(x, Model)
