from __future__ import annotations

import skrf
import equinox as eqx

import jax.numpy as jnp
from pmrf._util import field
from pmrf.constants import NumberLike, FrequencyUnitT, UNIT_DICT, MULTIPLIER_DICT

class Frequency(eqx.Module):
    """
    Represents a frequency axis for **paramrf** models.

    This class provides a container for a frequency band, defining the points
    at which network parameters are evaluated. The source code has been
    derived from the scikit-rf Frequency class, but with added JAX compatibility.

    The primary purpose is to hold a vector of frequency points (`f`) and the
    corresponding frequency unit (`unit`). It provides numerous properties
    for accessing different representations of the frequency axis, such as
    angular frequency (`w`) and scaled frequency (`f_scaled`).

    Attributes
    ----------
    _f : jnp.ndarray
        The frequency vector in Hz.
    _unit : str
        The frequency unit (e.g., 'hz', 'ghz'). Marked as static for JAX/Equinox.

    Examples
    --------
    .. code-block:: python

        import pmrf as prf
        import skrf as rf

        # Create a frequency axis from 1 to 2 GHz with 101 points
        freq = prf.Frequency(start=1, stop=2, npoints=101, unit='ghz')

        # Access properties like the frequency vector in Hz or radians/sec
        print(f"Frequency points in Hz: {freq.f[:5]}...")
        print(f"Angular frequency in rad/s: {freq.w[:5]}...")

        # Convert to a scikit-rf Frequency object
        skrf_freq = freq.to_skrf()
        print(f"Type after conversion: {type(skrf_freq)}")

        # Create from a scikit-rf Frequency object
        freq_from_skrf = prf.Frequency.from_skrf(skrf_freq)
    """
    _f: jnp.array
    _unit: str = field(static=True)

    def __init__(self, start: float = 0, stop: float = 0, npoints: int = 0, unit: FrequencyUnitT | None = 'Hz') -> None:
        """
        Frequency initializer.

        Creates a Frequency object from start/stop/npoints and a unit.
        Alternatively, the class method :func:`from_f` can be used to
        create a Frequency object from a frequency vector instead.

        Parameters
        ----------
        start : number, optional
            Start frequency in units of `unit`. Default is 0.
        stop : number, optional
            Stop frequency in units of `unit`. Default is 0.
        npoints : int, optional
            Number of points in the band. Default is 0.
        unit : string, optional
            Frequency unit of the band: 'Hz', 'kHz', 'MHz', 'GHz', 'THz'.
            This is used to create the attribute :attr:`f_scaled`.
            It is also used by the :class:`~skrf.network.Network` class
            for plots vs. frequency. Default is 'Hz'.

        Notes
        -----
        The attribute `unit` sets the frequency multiplier, which is used
        to scale the frequency when `f_scaled` is referenced.

        The attribute `unit` is not case sensitive.
        Hence, for example, 'GHz' or 'ghz' is the same.

        See Also
        --------
        from_f : constructs a Frequency object from a frequency
            vector instead of start/stop/npoints.
        unit : frequency unit of the band

        Examples
        --------
        >>> wr1p5band = Frequency(start=500, stop=750, npoints=401, unit='ghz')
        >>> logband = Frequency(1, 1e9, 301, sweep_type='log')
        """
        self._unit = unit.lower()
        start =  self.multiplier * start
        stop = self.multiplier * stop
        self._f = jnp.linspace(start, stop, npoints)

    @classmethod
    def from_f(cls, f: NumberLike, unit: FrequencyUnitT | None = None) -> Frequency:
        """
        Construct Frequency object from a frequency vector.

        The unit is set by kwarg 'unit'.

        Parameters
        ----------
        f : scalar or array-like
            Frequency vector.
        unit : FrequencyUnitT, optional
            Frequency unit of the band. Default is None (defaults to 'Hz').

        Returns
        -------
        Frequency
            The instantiated Frequency object.

        Examples
        --------
        >>> f = np.linspace(75,100,101)
        >>> rf.Frequency.from_f(f, unit='GHz')
        """
        unit = unit or 'Hz'
        if jnp.isscalar(f):
            f = [f]
        temp_freq =  cls(0,0,0,unit=unit)
        new_freq = eqx.tree_at(lambda freq: freq._f, temp_freq, jnp.asarray(f) * temp_freq.multiplier)

        return new_freq
        
    @staticmethod
    def from_skrf(skrf_frequency: skrf.Frequency) -> 'Frequency':
        """
        Create a `pmrf.Frequency` from a `skrf.Frequency` object.

        Parameters
        ----------
        skrf_frequency : skrf.Frequency
            The scikit-rf Frequency object.

        Returns
        -------
        Frequency
            The equivalent pmrf Frequency object.
        """
        return Frequency.from_f(skrf_frequency.f_scaled, unit=skrf_frequency.unit)
    
    def to_skrf(self) -> skrf.Frequency:
        """
        Convert this `pmrf.Frequency` object to a `skrf.Frequency` object.

        Returns
        -------
        skrf.Frequency
            The equivalent scikit-rf Frequency object.
        """
        import numpy as np
        return skrf.Frequency.from_f(np.array(self.f_scaled), self._unit)
    
    def __hash__(self):
        return hash((self._f, self.unit))

    def __eq__(self, other: Frequency):
        return jnp.array_equal(self._f, other._f) and self.unit == other.unit

    def __len__(self) -> int:
        """
        Return the number of frequency points.

        Returns
        -------
        int
            Length of the frequency vector.
        """
        return self.npoints

    def __add__(self, other: Frequency | NumberLike) -> Frequency:
        """
        Elementwise addition on frequency values.

        Parameters
        ----------
        other : Frequency or NumberLike
            The addend. If a :class:`Frequency`, frequencies are added elementwise;
            otherwise ``other`` is broadcast as needed.

        Returns
        -------
        Frequency
            A new object with updated frequency vector.
        """
        out = self.copy()
        out._f = self.f + (other.f if isinstance(other, Frequency) else other)
        return out

    def __sub__(self, other: Frequency | NumberLike) -> Frequency:
        """
        Elementwise subtraction on frequency values.

        Parameters
        ----------
        other : Frequency or NumberLike
            The subtrahend. If a :class:`Frequency`, frequencies are subtracted
            elementwise; otherwise ``other`` is broadcast.

        Returns
        -------
        Frequency
            A new object with updated frequency vector.
        """
        out = self.copy()
        out._f = self.f - (other.f if isinstance(other, Frequency) else other)
        return out

    def __mul__(self, other: Frequency | NumberLike) -> Frequency:
        """
        Elementwise multiplication on frequency values.

        Parameters
        ----------
        other : Frequency or NumberLike
            The multiplier. If a :class:`Frequency`, multiply elementwise;
            otherwise ``other`` is broadcast.

        Returns
        -------
        Frequency
            A new object with updated frequency vector.
        """
        out = self.copy()
        out._f = self.f * (other.f if isinstance(other, Frequency) else other)
        return out

    def __rmul__(self, other: Frequency | NumberLike) -> Frequency:
        """
        Reflected elementwise multiplication on frequency values.

        Parameters
        ----------
        other : Frequency or NumberLike
            The multiplier.

        Returns
        -------
        Frequency
            A new object with updated frequency vector.
        """
        out = self.copy()
        out._f = self.f * (other.f if isinstance(other, Frequency) else other)
        return out

    def __div__(self, other: Frequency | NumberLike) -> Frequency:
        """
        Elementwise division on frequency values (Python 2 style alias).

        Parameters
        ----------
        other : Frequency or NumberLike
            The divisor. If a :class:`Frequency`, divide elementwise;
            otherwise ``other`` is broadcast.

        Returns
        -------
        Frequency
            A new object with updated frequency vector.
        """
        out = self.copy()
        out._f = self.f / (other.f if isinstance(other, Frequency) else other)
        return out

    def __truediv__(self, other: Frequency | NumberLike) -> Frequency:
        """
        Elementwise true division on frequency values.

        Parameters
        ----------
        other : Frequency or NumberLike
            The divisor. If a :class:`Frequency`, divide elementwise;
            otherwise ``other`` is broadcast.

        Returns
        -------
        Frequency
            A new object with updated frequency vector.
        """
        out = self.copy()
        out._f = self.f / (other.f if isinstance(other, Frequency) else other)
        return out

    def __floordiv__(self, other: Frequency | NumberLike) -> Frequency:
        """
        Elementwise floor division on frequency values.

        Parameters
        ----------
        other : Frequency or NumberLike
            The divisor.

        Returns
        -------
        Frequency
            A new object with updated frequency vector.
        """
        out = self.copy()
        out._f = self.f // (other.f if isinstance(other, Frequency) else other)
        return out

    def __mod__(self, other: Frequency | NumberLike) -> Frequency:
        """
        Elementwise modulo on frequency values.

        Parameters
        ----------
        other : Frequency or NumberLike
            The modulus.

        Returns
        -------
        Frequency
            A new object with updated frequency vector.
        """
        out = self.copy()
        out._f = self.f % (other.f if isinstance(other, Frequency) else other)
        return out

    @property
    def start(self) -> float:
        """
        The starting frequency in Hz.

        Returns
        -------
        float
            Start frequency.
        """
        return self.f[0]

    @property
    def start_scaled(self) -> float:
        """
        The starting frequency in the specified `unit`.

        Returns
        -------
        float
            Scaled start frequency.
        """
        return self.f_scaled[0]
    @property
    def stop_scaled(self) -> float:
        """
        The stop frequency in the specified `unit`.

        Returns
        -------
        float
            Scaled stop frequency.
        """
        return self.f_scaled[-1]

    @property
    def stop(self) -> float:
        """
        The stop frequency in Hz.

        Returns
        -------
        float
            Stop frequency.
        """
        return self.f[-1]

    @property
    def npoints(self) -> int:
        """
        The number of points in the frequency axis.

        Returns
        -------
        int
            Number of points.
        """
        return len(self.f)

    @property
    def center(self) -> float:
        """
        The center frequency in Hz.

        Returns
        -------
        float
            The exact center frequency in Hz.
        """
        return self.start + (self.stop-self.start)/2.

    @property
    def center_idx(self) -> int:
        """
        The index of the frequency point closest to the center.

        Returns
        -------
        int
            Index of the center frequency.
        """
        return self.npoints // 2

    @property
    def center_scaled(self) -> float:
        """
        The center frequency in the specified `unit`.

        Returns
        -------
        float
            The exact center frequency in the specified `unit`.
        """
        return self.start_scaled + (self.stop_scaled-self.start_scaled)/2.

    @property
    def step(self) -> float:
        """
        The frequency step size in Hz for evenly spaced sweeps.

        Returns
        -------
        float
            Step size in Hz.
        """
        if self.span == 0:
            return 0.
        else:
            return self.span / (self.npoints - 1.)

    @property
    def step_scaled(self) -> float:
        """
        The frequency step size in the specified `unit` for evenly spaced sweeps.

        Returns
        -------
        float
            Step size in units.
        """
        if self.span_scaled == 0:
            return 0.
        else:
            return self.span_scaled / (self.npoints - 1.)

    @property
    def span(self) -> float:
        """
        The frequency span (stop - start) in Hz.

        Returns
        -------
        float
            Span in Hz.
        """
        return abs(self.stop-self.start)

    @property
    def span_scaled(self) -> float:
        """
        The frequency span (stop - start) in the specified `unit`.

        Returns
        -------
        float
            Span in units.
        """
        return abs(self.stop_scaled-self.start_scaled)

    @property
    def f(self) -> jnp.ndarray:
        """
        The frequency vector in Hz.

        Returns
        -------
        jnp.ndarray
            The frequency vector in Hz.
        """
        return self._f

    @property
    def f_scaled(self) -> jnp.ndarray:
        """
        The frequency vector in the specified `unit`.

        Returns
        -------
        jnp.ndarray
            A frequency vector in the specified `unit`.
        """
        return self.f/self.multiplier

    @property
    def w(self) -> jnp.ndarray:
        r"""
        The angular frequency vector in radians/s.

        Angular frequency is defined as :math:`\omega=2\pi f`.

        Returns
        -------
        jnp.ndarray
            Angular frequency in rad/s.
        """
        return 2*jnp.pi*self.f

    @property
    def df(self) -> jnp.ndarray:
        """
        The gradient of the frequency vector, in Hz.

        Returns
        -------
        jnp.ndarray
            Gradient of frequency.
        """
        return jnp.gradient(self.f)

    @property
    def df_scaled(self) -> jnp.ndarray:
        """
        The gradient of the scaled frequency vector.

        Returns
        -------
        jnp.ndarray
            Gradient of scaled frequency.
        """
        return jnp.gradient(self.f_scaled)

    @property
    def dw(self) -> jnp.ndarray:
        """
        The gradient of the angular frequency vector, in rad/s.

        Returns
        -------
        jnp.ndarray
            Gradient of angular frequency.
        """
        return jnp.gradient(self.w)

    @property
    def unit(self) -> FrequencyUnitT:
        """
        The frequency unit.

        Possible values are 'Hz', 'kHz', 'MHz', 'GHz', 'THz'.
        Setting this attribute is not case-sensitive.

        Returns
        -------
        str
            String representing the frequency unit.
        """
        return UNIT_DICT[self._unit]

    @unit.setter
    def unit(self, unit: FrequencyUnitT) -> None:
        self._unit = unit.lower()

    @property
    def multiplier(self) -> float:
        """
        The multiplier to convert from the specified `unit` back to Hz.

        Returns
        -------
        float
            Multiplier for this frequency's unit.
        """
        return MULTIPLIER_DICT[self._unit]