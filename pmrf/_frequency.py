"""
The following file is immediate derivative work of the scikit-rf library. The scikit-rf copyright license is attached.

---------------------------------------------------------------

Copyright (c) 2010, Alexander Arsenovic All rights reserved.

Copyright (c) 2017, scikit-rf Developers All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the scikit-rf nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---------------------------------------------------------------
"""

from __future__ import annotations

from numbers import Number

import skrf
import equinox as eqx

import pmrf.numpy as np
from pmrf._misc import field

from skrf.constants import FREQ_UNITS, FrequencyUnitT, NumberLike
UNIT_DICT: dict[str] = {k.lower(): k for k in FREQ_UNITS}
MULTIPLIER_DICT = {k.lower(): v for k,v in FREQ_UNITS.items()}

class Frequency(eqx.Module):
    """
    A frequency band.

    This is a stripped-down version of the `skrf.Frequency` class, mainly designed
    to be used alongside `jax` in `pmrf`.
    """
    _f: np.array
    _unit: str = field(static=True)

    def __init__(self, *args, frequency=None, **kwargs) -> None:
        """The main frequency initializer.
        
        Arguments are forward to the initializer for `skrf.Frequency`. To initialize directly from `skrf`, use `from_skrf(..)`.
        """
        frequency = frequency or skrf.Frequency(*args, **kwargs)
        self._unit = frequency._unit
        self._f = np.asarray(frequency._f)
        
    @staticmethod
    def from_skrf(skrf_frequency: skrf.Frequency) -> 'Frequency':
        return Frequency(frequency=skrf_frequency)

    def __len__(self) -> int:
        """
        The number of frequency points
        """
        return self.npoints

    def __add__(self, other: Frequency | NumberLike) -> Frequency:
        out = self.copy()
        out._f = self.f + (other.f if isinstance(other, Frequency) else other)
        return out

    def __sub__(self, other: Frequency | NumberLike) -> Frequency:
        out = self.copy()
        out._f = self.f - (other.f if isinstance(other, Frequency) else other)
        return out

    def __mul__(self, other: Frequency | NumberLike) -> Frequency:
        out = self.copy()
        out._f = self.f * (other.f if isinstance(other, Frequency) else other)
        return out

    def __rmul__(self, other: Frequency | NumberLike) -> Frequency:
        out = self.copy()
        out._f = self.f * (other.f if isinstance(other, Frequency) else other)
        return out

    def __div__(self, other: Frequency | NumberLike) -> Frequency:
        out = self.copy()
        out._f = self.f / (other.f if isinstance(other, Frequency) else other)
        return out

    def __truediv__(self, other: Frequency | NumberLike) -> Frequency:
        out = self.copy()
        out._f = self.f / (other.f if isinstance(other, Frequency) else other)
        return out

    def __floordiv__(self, other: Frequency | NumberLike) -> Frequency:
        out = self.copy()
        out._f = self.f // (other.f if isinstance(other, Frequency) else other)
        return out

    def __mod__(self, other: Frequency | NumberLike) -> Frequency:
        out = self.copy()
        out._f = self.f % (other.f if isinstance(other, Frequency) else other)
        return out

    @property
    def start(self) -> float:
        """
        Starting frequency in Hz.
        """
        return self.f[0]

    @property
    def start_scaled(self) -> float:
        """
        Starting frequency in :attr:`unit`'s.
        """
        return self.f_scaled[0]
    @property
    def stop_scaled(self) -> float:
        """
        Stop frequency in :attr:`unit`'s.
        """
        return self.f_scaled[-1]

    @property
    def stop(self) -> float:
        """
        Stop frequency in Hz.
        """
        return self.f[-1]

    @property
    def npoints(self) -> int:
        """
        Number of points in the frequency.
        """
        return len(self.f)

    @property
    def center(self) -> float:
        """
        Center frequency in Hz.

        Returns
        -------
        center : number
            the exact center frequency in units of Hz
        """
        return self.start + (self.stop-self.start)/2.

    @property
    def center_idx(self) -> int:
        """
        Closes idx of :attr:`f` to the center frequency.
        """
        return self.npoints // 2

    @property
    def center_scaled(self) -> float:
        """
        Center frequency in :attr:`unit`'s.

        Returns
        -------
        center : number
            the exact center frequency in units of :attr:`unit`'s
        """
        return self.start_scaled + (self.stop_scaled-self.start_scaled)/2.

    @property
    def step(self) -> float:
        """
        The inter-frequency step size (in Hz) for evenly-spaced
        frequency sweeps

        See Also
        --------
        df : for general case
        """
        if self.span == 0:
            return 0.
        else:
            return self.span / (self.npoints - 1.)

    @property
    def step_scaled(self) -> float:
        """
        The inter-frequency step size (in :attr:`unit`) for evenly-spaced
        frequency sweeps.

        See Also
        --------
        df : for general case
        """
        if self.span_scaled == 0:
            return 0.
        else:
            return self.span_scaled / (self.npoints - 1.)

    @property
    def span(self) -> float:
        """
        The frequency span.
        """
        return abs(self.stop-self.start)

    @property
    def span_scaled(self) -> float:
        """
        The frequency span.
        """
        return abs(self.stop_scaled-self.start_scaled)

    @property
    def f(self) -> np.ndarray:
        """
        Frequency vector in Hz.

        Returns
        ----------
        f : :class:`numpy.ndarray`
            The frequency vector  in Hz

        See Also
        ----------
        f_scaled : frequency vector in units of :attr:`unit`
        w : angular frequency vector in rad/s
        """

        return self._f

    @property
    def f_scaled(self) -> np.ndarray:
        """
        Frequency vector in units of :attr:`unit`.

        Returns
        -------
        f_scaled : numpy.ndarray
            A frequency vector in units of :attr:`unit`

        See Also
        --------
        f : frequency vector in Hz
        w : frequency vector in rad/s
        """
        return self.f/self.multiplier

    @property
    def w(self) -> np.ndarray:
        r"""
        Angular frequency in radians/s.

        Angular frequency is defined as :math:`\omega=2\pi f` [#]_

        Returns
        -------
        w : :class:`numpy.ndarray`
            Angular frequency in rad/s

        References
        ----------
        .. [#] https://en.wikipedia.org/wiki/Angular_frequency

        See Also
        --------
        f_scaled : frequency vector in units of :attr:`unit`
        f : frequency vector in Hz
        """
        return 2*np.pi*self.f

    @property
    def df(self) -> np.ndarray:
        """
        The gradient of the frequency vector.

        Note
        ----
        The gradient is calculated using::

            `np.gradient(self.f)`

        """
        return np.gradient(self.f)

    @property
    def df_scaled(self) -> np.ndarray:
        """
        The gradient of the frequency vector (in unit of :attr:`unit`).

        Note
        ----
        The gradient is calculated using::

            `np.gradient(self.f_scaled)`
        """
        return np.gradient(self.f_scaled)

    @property
    def dw(self) -> np.ndarray:
        """
        The gradient of the frequency vector (in radians).

        Note
        ----
        The gradient is calculated using::

            `np.gradient(self.w)`
        """
        return np.gradient(self.w)

    @property
    def unit(self) -> FrequencyUnitT:
        """
        Unit of this frequency band.

        Possible strings for this attribute are:
        'Hz', 'kHz', 'MHz', 'GHz', 'THz'

        Setting this attribute is not case sensitive.

        Returns
        -------
        unit : string
            String representing the frequency unit
        """
        return UNIT_DICT[self._unit]

    @unit.setter
    def unit(self, unit: FrequencyUnitT) -> None:
        self._unit = unit.lower()

    @property
    def multiplier(self) -> float:
        """
        Multiplier for formatting axis.

        Returns
        -------
        multiplier : number
            multiplier for this Frequencies unit
        """
        return MULTIPLIER_DICT[self._unit]

    def _t_padded(self, *, pad: int = 0, n: int | None = None, bandpass: bool | None = None) -> np.ndarray:
        if bandpass is None:
            bandpass = self.f[0] != 0

        if n is None:
            n = self.npoints + pad
            n = n if bandpass else n * 2 - 1

        if bandpass:
            dt = 1 / (n * self.step)
            t_stop = (n - 1) // 2 * dt
            t_start = -t_stop - dt if n % 2 == 0 else (-n // 2 + 1) * dt

            t = np.linspace(t_start, t_stop, num=n, endpoint=True)
        else:
            dt = 1 / (n * self.step)
            t = np.linspace(-dt * (n // 2), dt * (n // 2), num=n, endpoint=True)

        return t

    @property
    def t(self) -> np.ndarray:
        """
        Time vector in s.

        t_period = 2*(n-1)/f_step
        """
        return self._t_padded(bandpass=True)

    @property
    def t_ns(self) -> np.ndarray:
        """
        Time vector in ns.

        t_period = 2*(n-1)/f_step
        """
        return self.t*1e9

    def round_to(self, val: FrequencyUnitT | Number = 'Hz') -> None:
        """
        Round off frequency values to a specified precision.

        This is useful for dealing with finite precision limitations of
        VNA's and/or other software

        Parameters
        ----------
        val : string or number
            if val is a string it should be a frequency :attr:`unit`
            (ie 'Hz', 'MHz',etc). if its a number, then this returns
            f = f-f%val

        Examples
        --------
        >>> f = pmrf.Frequency.from_f([.1,1.2,3.5],unit='Hz')
        >>> f.round_to('Hz')

        """
        if isinstance(val, str):
            val = MULTIPLIER_DICT[val.lower()]

        self.f = np.round(self.f/val)*val

    def overlap(self,f2: Frequency) -> Frequency:
        """
        Calculates overlapping frequency  between self and f2.

        See Also
        --------
        overlap_freq

        """
        return overlap_freq(self, f2)

    @property
    def sweep_type(self) -> str:
        """
        Frequency sweep type.

        Returns
        -------
        sweep_type: str
            'lin' if linearly increasing, 'log' or 'unknown'.

        """
        if np.allclose(self.f, np.linspace(self.f[0], self.f[-1], self.npoints), rtol=0.05):
            sweep_type = 'lin'
        elif self.f[0] and np.allclose(self.f, np.geomspace(self.f[0], self.f[-1], self.npoints), rtol=0.05):
            sweep_type = 'log'
        else:
            sweep_type = 'unknown'
        return sweep_type

def overlap_freq(f1: Frequency,f2: Frequency) -> Frequency:
    """
    Calculates overlapping frequency between f1 and f2.

    Or, put more accurately, this returns a Frequency that is the part
    of f1 that is overlapped by f2. The resultant start frequency is
    the smallest f1.f that is greater than f2.f.start, and the stop
    frequency is the largest f1.f that is smaller than f2.f.stop.

    This way the new frequency overlays onto f1.


    Parameters
    ----------
    f1 : :class:`Frequency`
        a frequency object
    f2 : :class:`Frequency`
        a frequency object

    Returns
    -------
    f3 : :class:`Frequency`
        part of f1 that is overlapped by f2

    """
    if f1.start > f2.stop:
        raise ValueError('Out of bounds. f1.start > f2.stop')
    elif f2.start > f1.stop:
        raise ValueError('Out of bounds. f2.start > f1.stop')

    start = max(f1.start, f2.start)
    stop = min(f1.stop, f2.stop)
    f = f1.f[(f1.f>=start) & (f1.f<=stop)]
    freq =  Frequency.from_f(f, unit = 'Hz')
    freq.unit = f1.unit
    return freq
