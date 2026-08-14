"""
Base class for RF models.
"""

from typing import Any, Callable, TypeVar, Union, TypeGuard
from functools import cached_property
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx
import skrf
import parax as prx

from pmrf.utils.tree import resolve_target
from pmrf.parameters import tree_param_names_to_path
from pmrf.frequency import Frequency
from pmrf.rf import (
    a2s, s2a, s2y, y2s, s2z, z2s, y2z, z2y, a2y, y2a, a2z, z2a, s2mna, y2mna, z2mna, a2mna,
    MNAStamp,
)
from pmrf.math import CONVERSION_LOOKUP
from pmrf.utils.type import is_overridden
from pmrf.utils import field, unwrap, unwrap_self
from pmrf.distributions import AbstractDistribution
from pmrf.module import Module, validate

T = TypeVar('T')

PRIMARY_DOMAINS = ('s', 'a', 'y', 'z', 'mna')
PRIMARY_METHODS = PRIMARY_DOMAINS + ('primary_matrix',)
PLOT_DOMAINS = ('s', 'a', 'y', 'z')
HUB_Z0 = 50.0 + 0.0j
    

class Model(Module):
    """
    Base class for RF models.

    Derived from this class to define your own, custom model.

    This class should not be instantiated directly. It is created internally in ParamRF when models are
    built compositionally, or can be inherited from. When inheriting, at least one primary matrix method,
    such as or :meth:`pmrf.Model.s`, :meth:`pmrf.Model.a`, :meth:`pmrf.Model.y`, :meth:`pmrf.Model.z`, 
    or :meth:`pmrf.Model.primary_matrix`, must be overridden. Legacy classes may still
    override :meth:`pmrf.Model.build`, but that interface is deprecated.

    The model is a Equinox `Module <https://docs.kidger.site/equinox/api/module/module/>`_
    (an immutable dataclass) and a JAX PyTree. Parameters are declared using standard dataclass
    field syntax and should be annotated with type :type:`pmrf.Param` and field specifier :func:`pmrf.param`.
    For more details in parameter definitions, see :mod:`pmrf.parameters`.

    Note that this class is not marked as "abstract" since it should be treated more like a mix-in
    than an ABC class with specific methods to implement.

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
    :meth:`primary_matrix`            Return the primary matrix. Can be overridden for dynamic dispatch.
    :attr:`primary_domain`            The domain of the primary matrix as a string (e.g. ``"s"``, ``"a"``).
    :meth:`build`                     Build the model. Can be overridden for advanced models.
    :meth:`expand`                    Expands the model's topology. Used for circuit model flattening.
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
            
        for prop in PLOT_DOMAINS:
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

    def __getattribute__(self, name: str):
        attribute = super().__getattribute__(name)
        if name == 'build' and is_overridden(type(self), Model, 'build'):
            def deprecated_build(*args, **kwargs):
                warnings.warn(
                    "Model.build() is deprecated. Use a pmrf.Module to hold "
                    "parameters and models, with explicit methods returning RF models.",
                    FutureWarning,
                    stacklevel=2,
                )
                return attribute(*args, **kwargs)

            return deprecated_build
        return attribute

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
    
    # ---- Core API -------------------------------------------------------------
    
    def build(self) -> 'Model':
        """Build the model (deprecated).

        Use a :class:`pmrf.Module` with an explicit, domain-specific method that
        returns an RF model or circuit instead. This method remains temporarily
        available for compatibility with existing composite model classes.

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
    
    def expand(self) -> tuple[list[tuple['Model', int]], list[list[tuple['Model', int]]]] | None:
        """
        Expands this model into its internal graph representation for circuit flattening.

        This method is used by graph algorithms (like the solver in `Circuit.flattened`) 
        to unpack composite models, wrappers, and nested hierarchies into a 
        single flat netlist. This allows global matrix solves to be used, where desired.

        Note that `expand` is automatically implemented if :meth:`pmrf.Model.build` is overridden
        and the built model also implements expand. This means that most user-classes
        do NOT need to manually implement this method, and it is mainly intended for
        built-in composite models in ParamRF to override e.g. :class:`pmrf.models.Cascade`
        or :class:`pmrf.models.Renumbered`.

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
        Imagine a custom 2-port model that internally connects an Inductor and Capacitor 
        in series. When asked to expand, it exposes the inner components and their wiring:

        >>> def expand(self):
        ...     # 1. Grab internal components
        ...     L, C = self.inductor, self.capacitor
        ...     
        ...     # 2. Map our external ports to the internal components
        ...     port_mapping = [
        ...         (L, 0),  # External port 0 maps to Inductor port 0
        ...         (C, 1)   # External port 1 maps to Capacitor port 1
        ...     ]
        ...     
        ...     # 3. Define the internal connections (the netlist)
        ...     # Connect Inductor port 1 to Capacitor port 0
        ...     internal_connections = [
        ...         [(L, 1), (C, 0)]
        ...     ]
        ...     
        ...     return port_mapping, internal_connections
        """
        if is_overridden(self.__class__, Model, 'build'):
            return self.build().expand()
                
        return None
    
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
    
    @eqx.filter_jit
    @unwrap_self
    def s(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        """Scattering parameter matrix at port impedance z0.

        If a different parameter type (a, z, y) is primary, this converts it to S.
        
        To convert between port impedances, use :meth:`pmrf.rf.renormalize_s`.
        
        Note that, derived classes should use the **power wave** definition of S-parameters
        when implementing components using S-parameters.
        If you have a formulation in terms of another definition
        (such as traveling waves), simply use :meth:`pmrf.rf.s2s`.

        Parameters
        ----------
        frequency : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            S-parameter matrix with shape ``(nf, n, n)``.
        """
        # Direct delegation to build
        if is_overridden(type(self), Model, 'build'):
            return self.build().s(frequency, z0=z0)

        # Fetch primary
        primary_domain = self.primary_domain
        kwargs = {'z0': z0} if primary_domain == 's' else {}
        val = self.primary_matrix(frequency, **kwargs)

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
        # Direct delegation to build
        if is_overridden(type(self), Model, 'build'):
            return self.build().a(frequency)

        # Fetch primary
        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(frequency, **kwargs)

        # Return or Convert
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
        # Direct delegation to build
        if is_overridden(type(self), Model, 'build'):
            return self.build().z(frequency)

        # Fetch primary
        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(frequency, **kwargs)

        # Return or convert
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
        # Direct delegation to build
        if is_overridden(type(self), Model, 'build'):
            return self.build().y(frequency)

        # Fetch primary
        primary_domain = self.primary_domain
        kwargs = {'z0': HUB_Z0} if primary_domain == 's' else {}
        val = self.primary_matrix(frequency, **kwargs)

        # Return or convert
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
        
        If the model does not explicitly define an MNA stamp, this automatically 
        delegates to the appropriate conversion utility (`s2mna`, `z2mna`, etc.). 
        Explicitly defined Y-matrices are prioritized to maximize matrix sparsity, 
        while other domains fall back to auxiliary variables to guarantee stability.
        """
        # Direct delegation to build
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
                # Convert to scikit-rf Network at the specified frequency
                ntwk = self.to_skrf(freq)
                
                # Check if the generated Network actually supports this plot type
                if not hasattr(ntwk, name):
                    raise AttributeError(f"scikit-rf Network object has no attribute '{name}'")
                
                # Call the scikit-rf plot method with remaining args (e.g. labels, colors)
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
        
        Note that if a model is tied that has already been tied,
        the target and source location/name refers to the original, untied model. 

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
        
        if isinstance(self, Tied):
            model = self.model
        else:
            model = self
        
        name_to_path = tree_param_names_to_path(model)
        resolved_target = resolve_target(target, name_to_path)
        resolved_source = resolve_target(source, name_to_path)
        
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
    

def is_model(x: Any) -> TypeGuard[Model]:
    """
    Returns if `x` is an instance of :class:`pmrf.Model`.
    """
    return isinstance(x, Model)
