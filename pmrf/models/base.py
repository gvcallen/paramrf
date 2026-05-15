"""
Base class for RF models.
"""

from typing import Any
import dataclasses

import jax
import jax.numpy as jnp
import equinox as eqx
import skrf
from refrax import Lens
import parax as prx

from pmrf.frequency import Frequency
from pmrf.rf import a2s, s2a, s2z, z2s, s2y, y2s
from pmrf.math import CONVERSION_LOOKUP
from pmrf.constants import PRIMARY_PROPERTIES
from pmrf.utils.type import is_overridden
from pmrf.jax_utils import field, unwrap, unwrap_self

Z0_WARNING = \
r"""
WARNING: You have created a model with characteristic impedance other than 50 ohm.
Working with multiple models in ParamRF with differing characteristic impedances
is not yet officially supported and you may encounter subtle bugs. For now, it is
recommended to keep the default z0 and convert your results at the end.
"""

class Model(eqx.Module):
    """
    Base class for RF models.

    Derived from this class to define your own, custom model.

    This class should not be instantiated directly. It is created internally in ParamRF when models are
    built compositionally, or can be inherited from, in which case at least one of
    :meth:`pmrf.Model.s`, :meth:`pmrf.Model.a`, :meth:`pmrf.Model.y`, :meth:`pmrf.Model.z`,
    :meth:`pmrf.Model.build`, or :meth:`pmrf.Model.primary_matrix` should be overridden.

    The model is a Equinox `Module <https://gvcallen.github.io/parax/api/#parax.Module>`_
    (an immutable dataclass) and a JAX PyTree. Parameters are declared using standard dataclass
    field syntax and should be annotated with type :type:`pmrf.Param` and field specifier :func:`pmrf.param`.
    For more details in parameter definitions, see :mod:`pmrf.parameters`.

    Usage
    -----
    - Define new models by sub-classing the model and adding custom parameters and/or sub-models
    - Construct models by passing parameters and/or submodels to the initializer (like a dataclass).
    - Use :meth:`pmrf.Model.at` and methods such as :meth:`.terminated` and :meth:`.flipped` to create modified versions of your model.

    Methods & Properties Summary
    ----------------------------

    **Core Methods**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`s`                         Scattering (S) parameter matrix.
    :meth:`a`                         ABCD parameter matrix.
    :meth:`z`                         Impedance (Z) parameter matrix.
    :meth:`y`                         Admittance (Y) parameter matrix.
    :meth:`build`                     Build the model. Can be overridden for advanced models.
    :meth:`primary_matrix`            Return the primary matrix. Can be overridden for dynamic dispatch.
    :attr:`primary_property`          The primary property (e.g. ``"s"``, ``"a"``) as a string.
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
    :attr:`at`                        Modify a parameter at some path in the model.
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
            C1: prf.Param = prf.param()
            L:  prf.Param = prf.param()
            C2: prf.Param = prf.param()

            def a(self, freq: prf.Frequency) -> jnp.ndarray:
                w = freq.w
                Y1, Y2, Y3 = (1j * w * self.C1), (1j * w * self.C2), 1 / (1j * w * self.L)
                return jnp.array([
                    [1 + Y2 / Y3,        1 / Y3],
                    [Y1 + Y2 + Y1*Y2/Y3, 1 + Y1 / Y3],
                ]).transpose(2, 0, 1)

    An ``RLC`` network built in `build` using cascading:

    .. code-block:: python

        import pmrf as prf
        from pmrf.models import Resistor, Capacitor, Inductor
        from pmrf.parameters import Bounded

        class RLC(prf.Model):
            res: Resistor = Resistor(Bounded(9.0, 11.0))
            ind: Inductor = Inductor(Bounded(0.0, 10.0, scale=1e-12))
            cap: Capacitor = Capacitor(Bounded(0.0, 10.0, scale=1e-12))

            def build(self) -> prf.Model:
                return self.res ** self.ind ** self.cap.terminated()
            
    """
    #: The characteristic impedance of the model.
    #: NB: Mixing impedances across models is not fully supported.
    z0: complex = field(default=50.0+0j, kw_only=True, static=True)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        for name in PRIMARY_PROPERTIES:
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
            
        for prop in PRIMARY_PROPERTIES:
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
    
    @property
    def number_of_ports(self) -> int:
        """Number of ports.

        Returns
        -------
        int
        """
        freq = Frequency(1, 2, 2)
        eval = jax.eval_shape(lambda: self.s(freq))
        return eval.shape[1]

    @property
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
    
    
    def named_params(self, include_fixed: bool = False, unwrap: bool = True) -> dict[str, Any]:
        """
        Returns a dictionary of all parameters in the model mapped to their paths.

        This is a convenience method for inspection and debugging. It traverses the
        model's internal PyTree structure and returns a flat dictionary where the 
        keys are the Python-style attribute paths to each parameter.
        
        Each path can be passed to :attr:`pmrf.Model.at` via `.path` or `.select`.

        Parameters
        ----------
        include_fixed : bool
            Whether to include fixed parameters in the returned dictionary.
            Defaults to False.
        unwrap : bool
            Unwraps the parameters into raw floats/arrays. Defaults to True.
            To inspect internal parameter states (e.g. distributions, fixed etc.)
            pass `unwrap=False`.
        
        Returns
        -------
        dict[str, Any]
            A dictionary mapping string paths (e.g., '.ind.value') to their 
            corresponding JAX arrays or parameter objects.
            
        """
        if not include_fixed:
            leaves_with_path, _ = jax.tree.flatten_with_path(self, is_leaf=lambda x: prx.is_param(x) or prx.is_constant(x))
        else:
            leaves_with_path, _ = jax.tree.flatten_with_path(self, is_leaf=lambda x: prx.is_param(x))
        
        params = {jax.tree_util.keystr(path): leaf for path, leaf in leaves_with_path if prx.is_param(leaf)}
        
        if not include_fixed:
            params = {k: v for k, v in params.items() if not prx.is_constant(v)}
        
        if unwrap:
            params = prx.unwrap(params)
            params = {k: float(v) if jnp.isscalar(v) else v for k, v in params.items()}
            
        return params
    
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
    
    def primary_matrix(self, freq: Frequency) -> jnp.ndarray:
        """The primary matrix (e.g. ``s``, ``a`` etc.) as a function of frequency.

        The primary matrix represents the matrix returned by :attr:`pmrf.Model.primary_property`,
        which is either overridden by sub-classes, or is the first proprerty directly overriden
        out of :meth:`pmrf.Model.s`, :meth:`pmrf.Model.a`, :meth:`pmrf.Model.y`, :meth:`pmrf.Model.z`
        (in that order), unless :meth:``pmrf.Model.build`` is overridden, in which case the primary matrix
        of the built model is returned.
        
        This method can also be overriden itself in order to to dynamically
        implement one of the matrices as opposed to overriding it explicitly. 

        Returns
        -------
        jnp.ndarray

        Raises
        ------
        NotImplementedError
            If no primary property is overridden.
        """      
        primary_function = getattr(self, self.primary_property)
        return primary_function(freq)
    
    @property
    def primary_property(self) -> str:
        """The primary property (e.g. ``"s"``, ``"a"``) as a string.

        The primary property is the first overridden among
        :data:`PRIMARY_PROPERTIES`, unless ``build`` is overridden,
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
        unprioritized = tuple(p for p in PRIMARY_PROPERTIES if p not in prioritized)

        if is_overridden(type(self), Model, 'build'):
            return self.build().primary_property
        
        for property in prioritized:
            if is_overridden(type(self), Model, property):
                return property
        for property in unprioritized:
            if is_overridden(type(self), Model, property):
                return property
        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overridden, which are the only ones supported currently")     
    
    @eqx.filter_jit
    @unwrap_self
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Scattering parameter matrix.

        If a different parameter type (a, z, y) is primary, this converts it to S.
        
        Note that, in ParamRF, the **power wave** definition of S-parameters
        should be used. If you have a formulation in terms of another definition
        (such as traveling waves), simply use :meth:`pmrf.rf.s2s`
        (or :meth:`pmrf.rf.renormalize_s` if you need to change
        impedance too).

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
            return self.build().s(freq)

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary_matrix(freq)

        # 2. Return or Convert
        if primary_prop == 's':
            return val
        elif primary_prop == 'a':
            return a2s(val, self.z0)
        elif primary_prop == 'z':
            return z2s(val, self.z0)
        elif primary_prop == 'y':
            return y2s(val, self.z0)
        
        raise NotImplementedError(f"Conversion from '{primary_prop}' to 's' is not implemented.")
    
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
        
        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary_matrix(freq)

        # 2. Return or Convert
        if primary_prop == 'a':
            return val
        
        # Convert via S parameters (Hub strategy)
        if primary_prop == 's':
            s = val
        elif primary_prop == 'z':
            s = z2s(val, self.z0)
        elif primary_prop == 'y':
            s = y2s(val, self.z0)
        else:
            raise NotImplementedError(f"Conversion from '{primary_prop}' to 'a' is not implemented.")
            
        return s2a(s, self.z0)

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

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary_matrix(freq)

        # 2. Return or Convert
        if primary_prop == 'z':
            return val

        # Convert via S parameters (Hub strategy)
        if primary_prop == 's':
            s = val
        elif primary_prop == 'a':
            s = a2s(val, self.z0)
        elif primary_prop == 'y':
            s = y2s(val, self.z0)
        else:
            raise NotImplementedError(f"Conversion from '{primary_prop}' to 'z' is not implemented.")

        return s2z(s, self.z0)

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

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary_matrix(freq)

        # 2. Return or Convert
        if primary_prop == 'y':
            return val

        # Convert via S parameters (Hub strategy)
        if primary_prop == 's':
            s = val
        elif primary_prop == 'a':
            s = a2s(val, self.z0)
        elif primary_prop == 'z':
            s = z2s(val, self.z0)
        else:
            raise NotImplementedError(f"Conversion from '{primary_prop}' to 'y' is not implemented.")

        return s2y(s, self.z0)            
    
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
    
    @property
    def at(self) -> Lens:
        """Provides a fluent, lens-based interface for immutable PyTree updates.

        This property exposes a chainable API for safely mutating deeply nested
        models and parameters.
        
        Note that if your goal is to update static fields (not parameters e.g. strings),
        you should use :func:`pmrf.replace`.
        
        For more advanced manipulation, refer to the :mod:`refrax` documentation.
        
        Returns
        -------
        Lens
            A lens object focused on the root of the current instance.

        Examples
        --------
        Update a single attribute using `.set()` or `.apply()`:

        >>> new_model = model.at.R.set(20.0)
        >>> new_model = model.at.length.apply(lambda x: x * 2)

        Target multiple attributes simultaneously using `.select()`:

        >>> new_model = model.at.select('L', 'C').set(2.0)

        Apply a function over every item in a collection using `.each()`:

        >>> new_model = model.at.array_params.each().apply(jnp.abs)

        Select attributes dynamically based on a condition using `.where()`:

        >>> is_model = lambda x: isinstance(x, Model)
        >>> new_model = model.at.where(is_model).apply(modify_model)

        Select leaves (like parameters) and modify them (e.g. freeze them):
        >>> frozen_model = model.at.leaves(prf.is_param).apply(prf.freeze)
        """
        return Lens(self)
        
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
    
    def to_skrf(self, frequency: Frequency | Any, sigma=0.0, **kwargs) -> skrf.Network:
        """Convert the model at frequencies to an :class:`skrf.Network`.

        The active primary property (``self.primary_property``) is used.

        Parameters
        ----------
        frequency : pmrf.frequency.Frequency | skrf.Frequency
            Frequency grid.
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
        
        fval, fname = self.primary_matrix(model_freq), self.primary_property
        kwargs = kwargs or {}
        kwargs.update({
            fname: np.array(fval),
            'frequency': measured_freq,
            'z0': self.z0,
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
            # Inspect the dataclass fields of our custom RF models
            for f in dataclasses.fields(node):
                val = getattr(node, f.name)
                
                # Check for the silent corruption hazard
                is_array = isinstance(val, (jnp.ndarray, np.ndarray))
                is_static = f.metadata.get("static", False)
                
                if is_array and not is_static and jnp.issubdtype(val.dtype, jnp.inexact):
                    raise TypeError(
                        f"Field '{f.name}' in '{node.__class__.__name__}' is a raw JAX/NumPy array, "
                        f"which can be updated during optimization/inference.\n\n"
                        f"To make your intention clear, you must either:\n"
                        f"  1. Use the `pmrf.param` specifier (or a factory in `pmrf.parameters`) to indicate the value is a parameter\n"
                        f"  2. Explicitly mark the field as frozen using `{f.name}: jnp.ndarray = prf.field(converter=prf.freeze)` "
                        f"and then unwrap the frozen field when you need it using `prf.unwrap`."
                    )
                
                # Manually recurse into the field's value to catch nested Models, 
                # lists of Models, dicts of Models, etc.
                validate(val)