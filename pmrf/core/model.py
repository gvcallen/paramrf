"""
Abstract base class for RF models.
"""

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import equinox as eqx
import parax as prx
import skrf

from pmrf.core.frequency import Frequency
from pmrf.rf import a2s, s2a, s2z, z2s, s2y, y2s
from pmrf.math import CONVERSION_LOOKUP
from pmrf.constants import PRIMARY_PROPERTIES
from pmrf.utils import is_overridden

Z0_WARNING = \
r"""
WARNING: You have created a model with characteristic impedance other than 50 ohm.
Working with multiple models in ParamRF with differing characteristic impedances
is not yet officially supported and you may encounter subtle bugs. For now, it is
recommended to keep the default z0 and convert your results at the end.
"""

class Model(prx.Module, ABC):
    """
    Abstract base class for RF models.

    This base class is used to represent any computable RF network, referred to in
    **ParamRF** as a "Model". This class can be overridden for defining complex models,
    or can be utilized indirectly by combining models already provided in :mod:`pmrf.models`.

    Note that, since this inherits from `parax.Module <https://gvcallen.github.io/parax/api/#parax.Module>`_,
    models can easily be manipulated using e.g. ``mymodel.with_params(xxx)``. See the
    `Parax <https://gvcallen.github.io/parax>`_ documentation for more details.
    
    This class is abstract and should not be instantiated directly. Derive from :class:`Model`
    and override one of the primary property functions (e.g. :meth:`pmrf.Model.__call__`, :meth:`pmrf.Model.s`,
    :meth:`pmrf.Model.a`).

    The model is a Parax/Equinox `Module <https://gvcallen.github.io/parax/api/#parax.Module>`_
    (immutable, dataclass-like) and is treated as a JAX PyTree. Parameters are declared using standard dataclass
    field syntax using `parax.Parameter <https://gvcallen.github.io/parax/api/#parax.Parameter>`_.

    Usage
    -----
    - Define new models by sub-classing the model and adding custom parameters and/or sub-models
    - Construct models by passing parameters and/or submodels to the initializer (like a dataclass).
    - Use "past tense" functions to modify the model in conjunction with another model or data e.g. :meth:`.terminated`, :meth:`.flipped`.
    - Retrieve parameter information via Parax methods such as :meth:`.named_params`, :meth:`.param_names`, :meth:`.flat_params`, etc.
    - Use Parax ``with_xxx`` functions to modify fields, models and parameters within the model e.g. :meth:`.with_params`, :meth:`.with_fields`.    

    Methods & Properties Summary
    ----------------------------

    **Core API**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`__call__`                  Build the model. Should be overridden by sub-classes.
    :meth:`s`                         Scattering (S) parameter matrix.
    :meth:`a`                         ABCD parameter matrix.
    :meth:`z`                         Impedance (Z) parameter matrix.
    :meth:`y`                         Admittance (Y) parameter matrix.
    :meth:`primary`                   Dispatch to the primary function for the given frequency.
    :attr:`primary_function`          The primary function (``s`` or ``a``) as a callable.
    :attr:`primary_property`          The primary property (e.g. ``"s"``, ``"a"``) as a string.
    :attr:`number_of_ports`           Number of ports.
    :attr:`nports`                    Alias of :attr:`number_of_ports`.
    :attr:`port_tuples`               All (m, n) port index pairs.
    ================================= ====================================================================

    **Model Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`flipped`                   Return a version of the model with ports flipped.
    :meth:`renumbered`                Return a version of the model with ports renumbered.
    :meth:`terminated`                Return a new model terminated by another (e.g. load).
    ================================= ====================================================================

    **Plotting, File, & Conversion Utilities**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`plot_func`                 Evaluate and plot an arbitrary function of the model.
    :meth:`plot_func_samples`         Evaluate and plot a function over parameter samples.
    :meth:`to_skrf`                   Convert the model at frequencies to an :class:`skrf.Network`.
    :meth:`export_touchstone`         Export the model response to a Touchstone file.
    ================================= ====================================================================    

    Examples
    --------
    A ``PiCLC`` network ("foundational" model with fixed parameters and equations):

    .. code-block:: python

        import jax.numpy as jnp
        import parax as prx
        import pmrf as prf        

        class PiCLC(prf.Model):
            C1: prx.Parameter = 1.0e-12
            L:  prx.Parameter = 1.0e-9
            C2: prx.Parameter = 1.0e-12

            def a(self, freq: prf.Frequency) -> jnp.ndarray:
                w = freq.w
                Y1, Y2, Y3 = (1j * w * self.C1), (1j * w * self.C2), 1 / (1j * w * self.L)
                return jnp.array([
                    [1 + Y2 / Y3,        1 / Y3],
                    [Y1 + Y2 + Y1*Y2/Y3, 1 + Y1 / Y3],
                ]).transpose(2, 0, 1)

    An ``RLC`` network ("circuit" model with free parameters built using cascading)

    .. code-block:: python

        import pmrf as prf
        from pmrf.core import Resistor, Capacitor, Inductor, Cascade
        from parax.parameters import Uniform

        class RLC(prf.Model):
            res: Resistor = Resistor(Uniform(9.0, 11.0))
            ind: Inductor = Inductor(Uniform(0.0, 10.0, scale=1e-9))
            cap: Capacitor = Capacitor(Uniform(0.0, 10.0, scale=1e-12))

            def __call__(self) -> prf.Model:
                return self.res ** self.ind ** self.cap.terminated()
            
    """
    #: The characteristic impedance of the model.
    #: NB: Mixing impedances across models is not fully supported.
    z0: complex = prx.field(default=50.0+0j, kw_only=True, static=True)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)        
            
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

    # ---- Defaults / Primary ---------------------------------------------------    
    
    @property
    def primary_function(self) -> Callable[[Frequency], jnp.ndarray]:
        """The primary function (``s`` or ``a``) as a callable.

        The primary function is the first overridden among
        :data:`PRIMARY_PROPERTIES`, unless ``__call__`` is overridden,
        in which case the primary function of the built model is returned.

        Returns
        -------
        Callable[[Frequency], jnp.ndarray]

        Raises
        ------
        NotImplementedError
            If no primary property is overridden.
        """
        return getattr(self, self.primary_property)
            
    @property
    def primary_property(self) -> str:
        """The primary property (e.g. ``"s"``, ``"a"``) as a string.

        The primary property is the first overridden among
        :data:`PRIMARY_PROPERTIES`, unless ``__call__`` is overridden,
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

        if is_overridden(type(self), Model, '__call__'):
            return self().primary_property
        
        for property in prioritized:
            if is_overridden(type(self), Model, property):
                return property
        for property in unprioritized:
            if is_overridden(type(self), Model, property):
                return property
        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overridden, which are the only ones supported currently")    

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
    
    # ---- Core API -------------------------------------------------------------
    
    def __call__(self) -> 'Model':
        """Build the model.

        This function should be over-ridden by sub-classes.
        It is useful in defining complex models that are built
        using several sub-models (as opposed to equation-based models).

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
    
    @eqx.filter_jit
    def primary(self, freq: Frequency) -> jnp.ndarray:
        """Dispatch to the primary function for the given frequency."""        
        primary_function = self.primary_function
        return primary_function(freq)
    
    @eqx.filter_jit
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
        if is_overridden(type(self), Model, '__call__'):
            return self().s(freq)

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary(freq)

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
        if is_overridden(type(self), Model, '__call__'):
            return self().a(freq)
        
        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary(freq)

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
        if is_overridden(type(self), Model, '__call__'):
            return self().z(freq)

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary(freq)

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
        if is_overridden(type(self), Model, '__call__'):
            return self().y(freq)

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary(freq)

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
        return self.cascaded(other)
    
    def __matmul__(self, other: 'Model') -> 'Model':
        """Termination operator `@`."""        
        return self.terminated(other)
        
    def cascaded(self, other, **kwargs) -> 'Model':
        """Cascade this model with another, returning a new model.
        
        See :class:`pmrf.models.composite.transformed.Flipped`.

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
    
    # ---- Plotting --------------------------------------------------    

    def plot_func(
        self,
        func: Callable[['Model', Frequency], jnp.ndarray],
        freq: Frequency,
        *,
        ax = None,
        label: str | None = None,
        color: str | None = None,
        **kwargs
    ):
        """Evaluate and plot an arbitrary function of the current model.

        This method evaluates the provided function using the model's current 
        parameter values and plots the resulting response over frequency.

        Parameters
        ----------
        func : Callable[[Model, Frequency], jnp.ndarray]
            Function to evaluate. Must take a Model and a Frequency object and 
            return a jnp.ndarray of shape (n_freqs,).
        freq : Frequency
            Frequency grid to evaluate over.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, the current axes (`plt.gca()`) are used.
        label : str, optional
            Label for the plotted line (used in legends).
        color : str, optional
            Color for the line. If None, uses the matplotlib color cycle.
        **kwargs : dict
            Additional keyword arguments forwarded to `matplotlib.pyplot.plot` 
            (e.g., `linestyle`, `linewidth`, `alpha`).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            ax = plt.gca()

        # 1. Evaluate the function on the current model
        y_val = func(self, freq)
        y_val = np.asarray(y_val)
        
        # Extract the x-axis automatically from the frequency object
        x_axis = np.asarray(freq.f_scaled) 

        # 2. Plotting logic
        # Assemble kwargs safely to avoid passing multiple 'color' or 'label' arguments
        plot_kwargs = kwargs.copy()
        if label is not None:
            plot_kwargs['label'] = label
        if color is not None:
            plot_kwargs['color'] = color

        ax.plot(x_axis, y_val, **plot_kwargs)
        
        return ax    

    def plot_func_samples(
        self,
        func: Callable[['Model', Frequency], jnp.ndarray],
        freq: Frequency,
        *,
        key: jax.Array,
        num_samples: int = 1000,
        contours: bool = True,
        ax = None,
        label: str | None = None,
        color: str = 'C0',
        alpha: float = 0.1,
    ):
        """Evaluate and plot a function over samples from the parameter distribution.

        This method draws samples from the model's joint parameter distribution, 
        evaluates the provided function for each sample, and plots the resulting 
        responses over frequency.

        Parameters
        ----------
        func : Callable[[Model, Frequency], jnp.ndarray]
            Function to evaluate. Must take a Model and a Frequency object and 
            return a jnp.ndarray of shape (n_freqs,).
        freq : Frequency
            Frequency grid to evaluate over.
        key : jax.Array
            PRNG key for sampling the distribution.
        num_samples : int, default=1000
            Number of samples to draw.
        contours : bool, default=True
            If True, plots the mean response and filled contours corresponding 
            to 1, 2, and 3 standard deviations. If False, plots all individual 
            sample responses as transparent lines.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, the current axes (`plt.gca()`) are used.
        label : str, optional
            Label for the mean line (used in legends).
        color : str, default='C0'
            Base color for the lines and shaded regions.
        alpha : float, default=0.1
            Transparency of the individual lines (when `contours=False`). 

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            ax = plt.gca()

        # 1. Evaluate the ensemble
        y_samples = self.func_samples(func, freq, key=key, num_samples=num_samples)
        y_samples = np.asarray(y_samples)
        
        # Extract the x-axis automatically from the frequency object
        x_axis = np.asarray(freq.f_scaled) 

        # 2. Calculate central tendency
        y_mean = np.mean(y_samples, axis=0)

        # 3. Plotting logic
        if not contours:
            # Plot all individual samples
            # Transpose y_samples so matplotlib interprets columns as individual lines
            ax.plot(x_axis, y_samples.T, color=color, alpha=alpha)
            # Plot the mean as a solid line on top
            ax.plot(x_axis, y_mean, color=color, label=label, linewidth=2)
            
        else:
            # Plot mean line
            ax.plot(x_axis, y_mean, color=color, label=label, linewidth=2)
            
            # Plot contours for 1, 2, and 3 standard deviations
            y_std = np.std(y_samples, axis=0)
            
            # Decreasing opacity for outer standard deviations
            for i, sig_alpha in zip([1, 2, 3], [0.3, 0.2, 0.1]):
                ax.fill_between(
                    x_axis, 
                    y_mean - i * y_std, 
                    y_mean + i * y_std, 
                    color=color, 
                    alpha=sig_alpha, 
                    linewidth=0
                )
        
        return ax
        
    
    # ---- File and conversion utilities  --------------------------------------------------            
    
    def to_skrf(self, frequency: Frequency | Any, sigma=0.0, **kwargs) -> skrf.Network:
        """Convert the model at frequencies to an :class:`skrf.Network`.

        The active primary property (``self.primary_property``) is used.

        Parameters
        ----------
        frequency : pmrf.core.frequency | skrf.Frequency
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
        
        fval, fname = self.primary(model_freq), self.primary_property
        kwargs = kwargs or {}
        kwargs.update({
            fname: np.array(fval),
            'frequency': measured_freq,
            'name': kwargs.get('name', self.name),
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
        import skrf

        if not isinstance(filename, str):
            raise Exception('Filename must be a string')
        
        ntwk = self.to_skrf(frequency, sigma=sigma)
        return ntwk.write_touchstone(filename, **skrf_kwargs)