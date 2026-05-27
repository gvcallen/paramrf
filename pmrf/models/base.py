"""
Base class for RF models.
"""

from typing import Any, Callable, Self, TypeVar, Union
from functools import cached_property
import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx
import skrf
import parax as prx

from pmrf.rf.mna import MNAStamp
from pmrf.utils.optix import focus, Lens
from pmrf.parameters import Param, is_param
from pmrf.frequency import Frequency
from pmrf.rf import (
    a2s, s2a, s2y, y2s, s2z, z2s, y2z, z2y, a2y, y2a, a2z, z2a, s2mna, y2mna, z2mna, a2mna,
)
from pmrf.math import CONVERSION_LOOKUP
from pmrf.utils.type import is_overridden
from pmrf.utils import field, unwrap, unwrap_self

T = TypeVar('T')

PRIMARY_DOMAINS = ('s', 'a', 'y', 'z', 'mna')
PRIMARY_METHODS = PRIMARY_DOMAINS + ('build', 'primary_matrix')
HUB_Z0 = 50.0 + 0.0j


class Model(eqx.Module):
    """
    Base class for RF models.

    Derived from this class to define your own, custom model.

    This class should not be instantiated directly. It is created internally in ParamRF when models are
    built compositionally, or can be inherited from, in which case at least one of
    :meth:`pmrf.Model.s`, :meth:`pmrf.Model.a`, :meth:`pmrf.Model.y`, :meth:`pmrf.Model.z`,
    :meth:`pmrf.Model.build`, or :meth:`pmrf.Model.primary_matrix` should be overridden.

    The model is a Equinox `Module <https://docs.kidger.site/equinox/api/module/module/>`_
    (an immutable dataclass) and a JAX PyTree. Parameters are declared using standard dataclass
    field syntax and should be annotated with type :type:`pmrf.Param` and field specifier :func:`pmrf.param`.
    For more details in parameter definitions, see :mod:`pmrf.parameters`.

    Usage
    -----
    - Define new models by sub-classing the model and adding custom parameters and/or sub-models
    - Construct models by passing parameters and/or submodels to the initializer (like a dataclass).
    - Use :attr:`pmrf.Model.at` and methods such as :meth:`.terminated` and :meth:`.flipped` to create modified versions of your model.
    
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
    :meth:`build`                     Build the model. Can be overridden for advanced models.
    :meth:`primary_matrix`            Return the primary matrix. Can be overridden for dynamic dispatch.
    :attr:`primary_domain`            The domain of the primary matrix as a string (e.g. ``"s"``, ``"a"``).
    ================================= ====================================================================

    **Helper Methods**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :attr:`number_of_ports`           Number of ports.
    :attr:`nports`                    Alias of :attr:`number_of_ports`.
    :attr:`port_tuples`               All (m, n) port index pairs.
    :meth:`named_params`              Extracts all named parameters in the model.
    ================================= ====================================================================

    **Model Transformation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`at`                        Modify, filter or inspect a value at some path in the model.
    :meth:`flipped`                   Return a version of the model with ports flipped.
    :meth:`renumbered`                Return a version of the model with ports renumbered.
    :meth:`terminated`                Return a new model terminated by another (e.g. load).
    :meth:`tied`                      Tie certain parameters/sub-models together.
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

    An ``RLC`` model built using cascading and with parameters specified:

    .. code-block:: python

        import pmrf as prf
        from pmrf.models import Resistor, Capacitor, Inductor

        class RLC(prf.Model):
            res: Resistor = Resistor(prf.Bounded(9.0, 11.0))
            ind: Inductor = Inductor(prf.Bounded(0.0, 10.0, scale=1e-12))
            cap: Capacitor = Capacitor(prf.Bounded(0.0, 10.0, scale=1e-12))

            def build(self) -> prf.Model:
                return self.res ** self.ind ** self.cap.terminated()
            
    """
    #: A name for the model.
    name: str | None = field(default=None, kw_only=True, static=True)

    #: Arbitrary metadata to store alongside the model.
    metadata: Any = field(default=None, kw_only=True, static=True)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for name in PRIMARY_METHODS:
            if name in cls.__dict__:
                original_method = cls.__dict__[name]
                wrapped_method = eqx.filter_jit(unwrap_self(original_method))
                setattr(cls, name, wrapped_method)        
            
        # --- Implement dynamic functions (s_mag, s_mn_mag, etc.) ---
        def make_dynamic_method(prop_name, func):
            def dynamic_method(self, *args, **kwargs):
                matrix = getattr(self, prop_name)(*args, **kwargs)
                return func(matrix)
            return dynamic_method
            
        for prop in PRIMARY_DOMAINS:
            for suffix, lookup in CONVERSION_LOOKUP.items():
                func = lookup[1]
                
                # Base function (e.g. s_mag)
                func_name = f"{prop}_{suffix}"
                if not hasattr(cls, func_name):  # Protect user overrides!
                    m = make_dynamic_method(prop, func)
                    m._pmrf_auto = True
                    setattr(cls, func_name, m)
                
                # Indexed function (e.g. s_mn_mag)
                func_name_mn = f"{prop}_mn_{suffix}"
                if not hasattr(cls, func_name_mn):
                    m_mn = make_dynamic_method(f"{prop}_mn", func)
                    m_mn._pmrf_auto = True
                    setattr(cls, func_name_mn, m_mn)

    # ---- Introspection properties --------------------------------------------------------
    
    @cached_property
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

    @cached_property
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
    
    def pathed_params(
        self,
        values: bool = True,
        include_fixed: bool = True,
        keystr: bool = False,
        separator: str | None = None,
    ) -> list[tuple[Any, Param]]:
        """
        Returns the parameters as a list of tuples alongside their paths.
        
        The paths represent JAX tree paths.
        
        Parameters
        ----------
        include_fixed : bool, default=False
            Whether to include fixed parameters in the returned dictionary.
            Defaults to False.
        values : bool, default=True
            Unwraps the parameters into raw floats/arrays. Defaults to True.
            To inspect or modify internal parameter states (e.g. distributions, fixed etc.)
            pass `unwrap=False`.        
        keystr : bool, default=False
            Whether equivalent strings should be returned as opposed to full JAX paths.
            Defaults to False.
        separator : str, optional
            The separator to use if `keystr` is True.
        """
        # Setup callables for filtering/flattening
        if not include_fixed:
            filter_spec = lambda x: prx.is_param(x) and not prx.is_constant(x)
            is_leaf = lambda x: prx.is_param(x) or prx.is_constant(x)
        else:
            filter_spec = lambda x: prx.is_param(x)
            is_leaf = lambda x: prx.is_param(x)

        # Get rid of any non-param leaves
        filtered_self = eqx.filter(self, filter_spec, is_leaf=is_leaf)
        pathed, _ = jax.tree.flatten_with_path(filtered_self, is_leaf=is_leaf)
        
        if values or keystr:
            for i in range(len(pathed)):
                key, value = pathed[i]
                
                if keystr:
                    kwargs = {'separator': separator} if separator is not None else {}
                    key = jax.tree_util.keystr(key, **kwargs)
                
                if values:
                    value = prx.unwrap(value)
                    if jnp.isscalar(value):
                        value = float(value)
                
                pathed[i] = (key, value)
                
        return pathed
    
    def named_params(
        self,
        values: bool = True,
        include_fixed: bool = True,
        namespace_separator: str = '_',
    ) -> dict[str, Param]:
        """
        Returns a named dictionary of parameters in the model.

        Parameters and models can be given names upon construction.
        The naming convention is as follows:
        
        1. If no names are present in the path of a parameter, its Python 
           path is used.
        2. If there are any named models in the path of a parameter, the path 
           up until the left of that model is collapsed, forming a namespace prefix.
           If multiple models in the path have names, they are joined
           using the supplied namespace separator.
        3. If the parameter itself is named, its path to the left is collapsed, 
           either to the root or to the first named model.

        This enables you to choose your own naming convention:

        1. For flat parameter names across your entire root model,
           name all your parameters but none of your models.
        2. For flat model names across your entire root model,
           name all your leaf models but none of your parameters
           or composite models.
        3. For fully nested naming, name all of your models and
           optionally your parameters.

        Parameters
        ----------
        values : bool
            Unwraps the parameters into raw floats/arrays. Defaults to True.
            To inspect or modify internal parameter states (e.g. distributions, fixed etc.)
            pass `unwrap=False`.
        include_fixed : bool
            Whether to include fixed parameters in the returned dictionary.
            Defaults to False.
        namespace_separator : str
            The separator to use to create a parameter namespace using model names.
        
        Returns
        -------
        dict[str, Any]
            A dictionary mapping string paths (e.g., '.ind.value') to their 
            corresponding JAX arrays or parameter objects.
            
        """
        pathed = self.pathed_params(include_fixed=include_fixed, values=values)
        
        # Detect collisions
        named = {}
        for path, leaf in pathed:
            name = tree_path_to_name(self, path, namespace_separator=namespace_separator)
            if name in named:
                raise ValueError(
                    f"Parameter name collision: '{name}'.\n\n"
                    f"Multiple paths resolved to the same name during flattening. "
                    f"To fix this, either assign unique names directly to the parameters, "
                    f"or give their parent models distinct names to create unique prefixes."
                )
            named[name] = leaf
        
        return named
    
    # ---- Core API -------------------------------------------------------------
    
    @eqx.filter_jit
    @unwrap_self
    def build(self) -> 'Model':
        """Build the model.

        This function can be over-ridden by sub-classes.

        It is useful to define advanced models that are built
        using several sub-models or parameters, as opposed to
        simpler models built using standard equations.

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
    
    def primary_matrix(self, freq: Frequency, **kwargs) -> jnp.ndarray:
        """The primary matrix (e.g. ``s``, ``a`` etc.) as a function of frequency.

        The primary matrix represents the matrix returned by :attr:`pmrf.Model.primary_domain`,
        which is either overridden by sub-classes, or is the first proprerty directly overriden
        out of :meth:`pmrf.Model.s`, :meth:`pmrf.Model.a`, :meth:`pmrf.Model.y`, :meth:`pmrf.Model.z`
        (in that order), unless :meth:``pmrf.Model.build`` is overridden, in which case the primary matrix
        of the built model is returned.
        
        This method can also be overriden itself in order to to dynamically
        implement one of the matrices as opposed to overriding it explicitly. 
        
        If this method is called and `self.primary_domain` is 's',
        then 'z0' should be passed in `kwargs`.
        
        Parameters
        ----------
        freq : Frequency
            Frequency grid.
        kwargs
            Key-word arguments forwarded to the primary matrix function, such as z0.

        Returns
        -------
        jnp.ndarray

        Raises
        ------
        NotImplementedError
            If no primary property is overridden.
        """      
        primary_function = getattr(self, self.primary_domain)
        return primary_function(freq, **kwargs)
    
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
        raise NotImplementedError(f"No primary properties in {PRIMARY_DOMAINS} are overridden, which are the only ones supported currently")     
    
    @eqx.filter_jit
    @unwrap_self
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        """Scattering parameter matrix at port impedance z0.

        If a different parameter type (a, z, y) is primary, this converts it to S.
        
        To convert between port impedances, use :meth:`pmrf.rf.renormalize_s`.
        
        Note that, derived classes should use the **power wave** definition of S-parameters
        when implementing components using S-parameters.
        If you have a formulation in terms of another definition
        (such as traveling waves), simply use :meth:`pmrf.rf.s2s`.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            S-parameter matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().s(freq, z0=z0)

        # Fetch primary
        primary_domain = self.primary_domain
        kwargs = {'z0': z0} if primary_domain == 's' else {}
        val = self.primary_matrix(freq, **kwargs)

        # Return or Convert
        if primary_domain == 's':
            return val
        elif primary_domain == 'a':
            return a2s(val, z0)
        elif primary_domain == 'z':
            return z2s(val, z0)
        elif primary_domain == 'y':
            return y2s(val, z0)
        
        raise NotImplementedError(f"Conversion from '{primary_domain}' to 's' is not implemented.")
    
    @eqx.filter_jit
    @unwrap_self
    def a(self, freq: Frequency) -> jnp.ndarray:
        """ABCD parameter matrix.

        If a different parameter type is primary, this converts it to A.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            ABCD matrix with shape ``(nf, 2, 2)``.
        """        
        if is_overridden(type(self), Model, 'build'):
            return self.build().a(freq)
        
        # Fetch primary
        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(freq, **kwargs)

        # Return direct
        if primary_domain == 'a':
            return val
        
        # Convert with priority s, z, y
        if primary_domain == 's':
            return s2a(val, z0=HUB_Z0)
        elif primary_domain == 'z':
            return z2a(val)
        elif primary_domain == 'y':
            return y2a(val)
        else:
            raise NotImplementedError(f"Conversion from '{primary_domain}' to 'a' is not implemented.")

    @eqx.filter_jit
    @unwrap_self
    def z(self, freq: Frequency) -> jnp.ndarray:
        """Impedance (Z) parameter matrix.

        If a different parameter type is primary, this converts it to Z.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            Z matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().z(freq)

        # Fetch primary
        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(freq, **kwargs)

        # Return direct
        if primary_domain == 'z':
            return val

        # Convert with priority s, a, y
        if primary_domain == 's':
            return s2z(val, z0=HUB_Z0)
        elif primary_domain == 'a':
            return a2z(val)
        elif primary_domain == 'y':
            return y2z(val)
        else:
            raise NotImplementedError(f"Conversion from '{primary_domain}' to 'z' is not implemented.")

    @eqx.filter_jit
    @unwrap_self
    def y(self, freq: Frequency) -> jnp.ndarray:
        """Admittance (Y) parameter matrix.

        If a different parameter type is primary, this converts it to Y.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            Y matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().y(freq)

        # Fetch primary
        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(freq, **kwargs)

        # Return direct
        if primary_domain == 'y':
            return val

        # Convert with priority s, a, z
        if primary_domain == 's':
            return s2y(val, HUB_Z0)
        elif primary_domain == 'a':
            return a2y(val)
        elif primary_domain == 'z':
            return z2y(val)
        else:
            raise NotImplementedError(f"Conversion from '{primary_domain}' to 'y' is not implemented.")
        
    @eqx.filter_jit
    @unwrap_self
    def mna(self, freq: Frequency) -> MNAStamp:
        """
        (experimental) Modified Nodal Analysis (MNA) stamp.

        Can be overridden in sub-classes.
        
        If the model does not explicitly define an MNA stamp, this automatically 
        delegates to the appropriate conversion utility (`s2mna`, `z2mna`, etc.). 
        Explicitly defined Y-matrices are prioritized to maximize matrix sparsity, 
        while other domains fall back to auxiliary variables to guarantee stability.
        """
        if is_overridden(type(self), Model, 'build'):
            return self.build().mna(freq)

        primary_domain = self.primary_domain
        
        if primary_domain == 'mna':
            return self.primary_matrix(freq)
            
        # We prioritize y, z and a for sparsity, assuming the caller
        # has created a numerically stable implementation.
        if primary_domain == 'y' or is_overridden(type(self), Model, 'y'):
            return y2mna(self.y(freq))
        
        if primary_domain == 'z' or is_overridden(type(self), Model, 'z'):
            return z2mna(self.z(freq))
        
        if primary_domain == 'a' or is_overridden(type(self), Model, 'a'):
            return a2mna(self.a(freq))
        
        if primary_domain == 's' or is_overridden(type(self), Model, 's'):
            return s2mna(self.s(freq, z0=HUB_Z0), z0=HUB_Z0)
            
        return y2mna(self.y(freq))

    # ---- Magic methods and copying --------------------------------------------------

    def __repr__(self) -> str:
        """String representation of the Model."""
        import numpy as np
        import jax

        class _RawFormatter:
            """Wrapper to print arrays cleanly with rounded float values."""
            def __init__(self, val):
                self.val = np.asarray(val)
                
            def __repr__(self):
                # precision=4 limits decimal places
                # suppress_small=True formats numbers very close to zero as 0
                return np.array2string(
                    self.val, 
                    separator=', ', 
                    precision=4, 
                )

        # Unwrap the model to resolve variables
        unwrapped = unwrap(self)
        
        # Identify JAX arrays
        is_array = lambda x: isinstance(x, (jax.Array, jnp.ndarray))
        
        # Replace JAX arrays with our custom formatter
        unwrapped_clean = jax.tree_util.tree_map(
            lambda x: _RawFormatter(x) if is_array(x) else x,
            unwrapped,
            is_leaf=is_array
        )

        # Use Equinox's formatter, displaying full internal arrays instead of generic shape labels
        return eqx.tree_pformat(unwrapped_clean, short_arrays=False)

    def __str__(self) -> str:
        return repr(self)    

    def __getattr__(self, name: str):
        """
        Dynamic dispatch for scikit-rf plotting methods.
        
        Captures calls like `model.plot_s_db(freq)` and redirects them 
        to `model.to_skrf(freq).plot_s_db()`.
        """
        if name.startswith('plot_'):
            def plotter(freq: Frequency, *args, **kwargs):
                # 1. Convert to scikit-rf Network at the specified frequency
                ntwk = self.to_skrf(freq)
                
                # 2. Check if the generated Network actually supports this plot type
                if not hasattr(ntwk, name):
                    raise AttributeError(f"scikit-rf Network object has no attribute '{name}'")
                
                # 3. Call the scikit-rf plot method with remaining args (e.g. labels, colors)
                return getattr(ntwk, name)(*args, **kwargs)
            return plotter
            
        # Standard fallback if the attribute isn't a plot command
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
    
    def at(
        self: Self, 
        where: Union[Callable[[Self], T], str, tuple[str, ...], list[str]]
    ) -> Lens[Self, T]:
        """(experimental) A functional interface for model manipulation.
        
        This is a wrapper around `equinox.tree_at` via the `jax-optix` library.
        
        Pass in a callable, a string parameter name, or a tuple of names that 
        returns the attributes you would like to retrieve/modify. Then, use
        methods like `.get()` and `.set()` to retrieve values
        or an updated model.
        
        WARNING: All updates made by this method are "surgical".
        In order words, values are replaced *as-is* without any converters
        or verification applied (a new instance is still returned).
        Any invariants must therefore be enforced or checked manually.
        For example, when replacing parameters, ensure to pass in a fully
        constructed parameter and not a float.
        
        Examples
        --------
        >>> import pmrf as prf
        >>> from pmrf.models import Resistor
        >>> model = Resistor(R=50.0, name="res")
        >>> # Retrieve a value using the lens with a parameter name
        >>> model.at("res.R").get()
        50.0
        >>> # Return a new model instance with the updated value
        >>> updated_model = model.at("res.R").set(100.0)

        Returns
        -------
        Lens
            A lens object focused on the root of the current instance.

        """
        resolved_where = _resolve_target(self, where)
        return focus(self).at(resolved_where)
    
    def map(self: Self, fn: Callable[[Any], Any], predicate: Callable | None = None) -> Self:
        """(experimental) A functional interface for model mapping.
        
        This is a wrapper around `jax.tree.map`.
        
        To map parameters, pass `is_target=prf.is_param`.
        
        Examples
        --------
        >>> import pmrf as prf
        >>> from pmrf.models import Resistor, Capacitor
        >>> model = Resistor(R=50.0) ** Capacitor(C=1e-12)
        >>> # Scale all parameters in the model by a factor of 2
        >>> scaled_model = model.map(lambda p: p * 2.0, is_leaf=prf.is_param)

        Returns the mapped model.
        """
        def _wrapped_fn(node):
            if not predicate(node):
                return node
            return fn(node)

        return jax.tree.map(_wrapped_fn, self, is_leaf=predicate)
        
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
    
    def tied(
        self, 
        target: Union[Callable[[Any], Any], str, tuple[str, ...], list[str]], 
        source: Union[Callable[[Any], Any], str, tuple[str, ...], list[str]], 
        tie_fn: Callable[[Any], Any] = lambda x: x, 
        **kwargs
    ) -> 'Model':
        """Tie parameters or sub-models within this model together.
        
        See :class:`pmrf.models.composite.wrapped.Tied` for more details.

        Examples
        --------
        >>> import pmrf as prf
        >>> from pmrf.models import Resistor, Capacitor
        >>> 
        >>> rc = Resistor(R=50.0, name="res") ** Capacitor(C=1.0e-12, name="cap")
        >>> 
        >>> # Tie the resistor's R to always be 50e12 times the capacitor's C
        >>> tied_rc = rc.tied(
        ...     target="res.R",
        ...     source="cap.C",
        ...     tie_fn=lambda c: c * 50e12
        ... )
        >>> 
        >>> # The optimizer will now only see the Capacitor's C parameter.
        >>> # When evaluated, R will automatically track C.

        Parameters
        ----------
        target : callable | str | tuple[str, ...] | list[str]
            A callable extracting the parameter to be overwritten 
            (e.g., `lambda m: m.resistor.R`), or the parameter's name.
        source : callable | str | tuple[str, ...] | list[str]
            A callable extracting the parameter to draw the value from 
            (e.g., `lambda m: m.capacitor.C`), or the parameter's name.
        tie_fn : callable, optional
            An optional transformation function applied to the source 
            before injecting it into the target. Defaults to the identity 
            function (`lambda x: x`).

        Returns
        -------
        Model
        """
        from pmrf.models import Tied
        
        resolved_target = _resolve_target(self, target)
        resolved_source = _resolve_target(self, source)
        
        return Tied(self, target=resolved_target, source=resolved_source, tie_fn=tie_fn, **kwargs)
    
    # ---- File and conversion utilities  --------------------------------------------------            
    
    def to_skrf(self, frequency: Frequency | Any, z0: ArrayLike = 50.0, sigma=0.0, **kwargs) -> skrf.Network:
        """Convert the model at frequencies to an :class:`skrf.Network`.

        The active primary property (``self.primary_domain``) is used.

        Parameters
        ----------
        frequency : pmrf.frequency.Frequency | skrf.Frequency
            Frequency grid.
        z0 : ArrayLike, default=50.0
            The charactestic impedance.
        sigma : float, default=0.0
            If nonzero, add complex Gaussian noise with stdev ``sigma`` to ``s``.
        **kwargs
            Forwarded to :class:`skrf.Network` constructor.

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
        
        s_matrix = self.s(model_freq, z0=z0)
        
        kwargs = kwargs or {}
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
    

def is_model(x: Any):
    """
    Returns if `x` is an instance of :class:`pmrf.Model`.
    """
    return isinstance(x, Model)
    
    
def validate(tree):
    """
    Recursively walks a PyTree and ensures no pmrf.Model instances contain 
    unprotected raw inexact arrays that optimizers might corrupt.
    """
    import numpy as np
    
    # Treat our models as leaves so JAX doesn't instantly unpack them into raw arrays
    nodes, _ = jax.tree_util.tree_flatten(
        tree, is_leaf=lambda x: isinstance(x, Model)
    )
    
    for node in nodes:
        if isinstance(node, Model):
            for f in dataclasses.fields(node):
                val = getattr(node, f.name)
                
                is_array = isinstance(val, jnp.ndarray)
                is_static = f.metadata.get("static", False)
                
                if is_array and not is_static and jnp.issubdtype(val.dtype, jnp.inexact):
                    raise TypeError(
                        f"Field '{f.name}' in '{node.__class__.__name__}' is a raw JAX array, "
                        f"meaning it is unclear whether this is a free or fixed parameter.\n\n"
                        f"To make your intention clear, you must either:\n"
                        f"  1. Use a factory in `pmrf.parameters` (e.g. `prf.Unconstrained`) for free variables, or a numpy array for fixed variables\n"
                        f"  2. Use a field specifier in the model class definition, e.g. `{f.name}: prf.Param = prf.param()` for automatic parameter conversion, "
                        f"or `{f.name}: jnp.ndarray = prf.field(converter=prf.freeze)` combined with `prf.unwrap` to ensure the variable is not optimized.\n"
                        f"This restriction is enforced to allow compatibility with machine learning models from othe libraries."
                    )
                
                # Recurse into submodels
                validate(val)


def tree_path_to_name(tree: Any, path: list[Any], namespace_separator: str, ignore_names: bool = False) -> str:
    """
    Converts a JAX-style path to an equivalent parameter or model name.

    Parameters
    ----------
    tree : PyTree
        The base PyTree to extract the names of.
    path : list[Any]
        The JAX path to a node in the PyTree.
    namespace_separator : str
        A string separator to use when combing multiple nodes in the PyTree
        together to create a new namespace.
    ignore_names : bool, default=False
        If True, ignores custom Model/Parameter names and generates the string 
        based strictly on the structural path.
    
    Returns
    -------
    str
        The name of the parameter or path.
    """
    namespace = []
    current_obj = tree
    unnamed_path_parts = []
    
    for key in path:
        if hasattr(key, "name"):
            attr = key.name
            current_obj = getattr(current_obj, attr)
            part_type = "attr"
        elif hasattr(key, "idx"):
            attr = key.idx
            current_obj = current_obj[key.idx]
            part_type = "idx"
        elif hasattr(key, "key"):
            attr = key.key
            current_obj = current_obj[key.key]
            part_type = "key"
        else:
            attr = key
            part_type = "fallback"
            
        if not ignore_names and isinstance(current_obj, Model) and current_obj.name is not None:
            namespace.append(current_obj.name)
            unnamed_path_parts = []
        else:
            unnamed_path_parts.append((part_type, attr))
            
    param_name = None
    if not ignore_names:
        if is_param(current_obj):
            param_name = current_obj.name
        
        if param_name is None and isinstance(current_obj, Model) and current_obj.name is not None:
            if namespace:
                param_name = namespace.pop()

    if param_name is not None:
        final_name_part = param_name
        separator = namespace_separator
    else:
        formatted_parts = []
        for i, (p_type, p_val) in enumerate(unnamed_path_parts):
            if p_type == "attr":
                if i == 0:
                    formatted_parts.append(str(p_val))
                else:
                    formatted_parts.append(f".{p_val}")
            elif p_type == "idx":
                formatted_parts.append(f"[{p_val}]")
            elif p_type == "key":
                if isinstance(p_val, str):
                    formatted_parts.append(f"['{p_val}']")
                else:
                    formatted_parts.append(f"[{p_val}]")
            else:
                formatted_parts.append(f"[{p_val}]")
                
        final_name_part = "".join(formatted_parts)
        
        if not final_name_part:
            separator = ""
        elif final_name_part.startswith("["):
            separator = ""
        else:
            separator = "."
            
    if namespace:
        prefix = namespace_separator.join(namespace)
        name = f"{prefix}{separator}{final_name_part}"
    else:
        name = final_name_part
            
    return name


def _make_getter(path: list[Any]) -> Callable[[Any], Any]:
    """Creates a callable that retrieves a value from a PyTree given its JAX path."""
    def getter(tree):
        curr = tree
        for p in path:
            if hasattr(p, "name"):
                curr = getattr(curr, p.name)
            elif hasattr(p, "idx"):
                curr = curr[p.idx]
            elif hasattr(p, "key"):
                curr = curr[p.key]
            else:
                # Fallback for unexpected path elements
                try:
                    curr = getattr(curr, p)
                except AttributeError:
                    curr = curr[p]
        return curr
    return getter

def _resolve_target(model: Any, target: Any, namespace_separator: str = '_') -> Callable[[Any], Any]:
    """Resolves callables, string names, or iterables of string names into a callable getter."""
    if callable(target):
        return target
        
    # Determine if target is a single string or an iterable of strings
    is_single = isinstance(target, str)
    if is_single:
        names_to_find = [target]
    elif isinstance(target, (list, tuple)) and all(isinstance(x, str) for x in target):
        names_to_find = list(target)
    else:
        raise TypeError(
            "Targets must be a callable, a string parameter name, "
            "or a tuple/list of string parameter names."
        )

    # Build name -> path mapping using the model's pathed_params
    pathed = model.pathed_params(include_fixed=True, values=False)
    name_to_path = {}
    for path, _ in pathed:
        name = tree_path_to_name(model, path, namespace_separator=namespace_separator)
        name_to_path[name] = path
        
    getters = []
    for name in names_to_find:
        if name not in name_to_path:
            raise ValueError(f"Parameter name '{name}' not found in the model.")
        getters.append(_make_getter(name_to_path[name]))
        
    # Return a single element getter or a tuple getter depending on the input
    if is_single:
        return getters[0]
    else:
        return lambda m: tuple(g(m) for g in getters)