"""
Frequency-dependent dielectric materials.

A dielectric owns the relative permittivity, loss tangent and static
conductivity of a medium. Every model returns a *total* complex relative
permittivity, so callers never need to know how the loss was specified.
"""
from __future__ import annotations

from abc import abstractmethod

import jax.numpy as jnp
from scipy.constants import epsilon_0

from pmrf.constraints import Positive, GreaterThan, Interval
from pmrf.frequency import Frequency
from pmrf.modules.base import Module
from pmrf.parameters import Param, param
from pmrf.utils import field


class AbstractDielectric(Module):
    r"""Abstract base class for a dielectric material.

    Subclasses return the *total* complex relative permittivity, using the
    convention $\varepsilon_r = \varepsilon' - j\varepsilon''$ with
    $\varepsilon'' \geq 0$ for a passive medium. The static conductivity of the
    medium is included, so callers never see the split.

    **Reference frequencies are material constants.** Fields such as
    :attr:`DjordjevicSarkar.flow`, :attr:`DjordjevicSarkar.fhigh` and
    :attr:`DjordjevicSarkar.fref` describe the medium, never the sweep, and must
    never be defaulted from the analysis band. Deriving them that way would make
    the same physical material give different answers depending on the band
    asked about, re-trigger a JIT compilation on every new frequency grid, and
    stop results being comparable across sweeps — which breaks any fitting
    workflow that calibrates on one grid and validates on another.
    """

    @abstractmethod
    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        """Total complex relative permittivity.

        Parameters
        ----------
        freq : Frequency
            The frequency axis to evaluate over.

        Returns
        -------
        jnp.ndarray
            Complex relative permittivity of shape ``(freq.npoints,)``, including
            any static conductivity contribution.
        """

    def with_conduction(self, eps: jnp.ndarray, freq: Frequency) -> jnp.ndarray:
        r"""Add the static conduction term $-j\sigma/(\omega\varepsilon_0)$ to `eps`.

        A constant loss tangent gives a conductance proportional to $\omega$,
        whereas a constant conductivity gives a frequency-independent one;
        neither can express the other, so `sigma` is carried separately by every
        concrete dielectric and folded in here.

        The term is guarded at DC with the double-``where`` pattern, so both the
        value and its gradient stay finite when the axis includes $\omega = 0$.
        """
        w = jnp.asarray(freq.w)
        safe_w = jnp.where(w > 0, w, 1.0)
        term = jnp.where(w > 0, self.sigma / (safe_w * epsilon_0), 0.0)
        return eps - 1j * term


class ConstantDielectric(AbstractDielectric):
    r"""
    Non-dispersive dielectric with a constant permittivity and loss tangent.

    This is the default material. It is not causal — a constant loss tangent
    cannot satisfy the Kramers-Kronig relations — but it is the conventional
    choice in a frequency-domain package. For a causal wideband alternative see
    :class:`DjordjevicSarkar`.

    **Mathematical Formulation**

    $$\varepsilon_r(\omega) = \varepsilon_r' \left(1 - j\tan\delta\right)
    - j\frac{\sigma}{\omega\varepsilon_0}$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.materials import ConstantDielectric

        fr4 = ConstantDielectric(epr=4.3, tand=0.02)
        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        eps = fr4.epsilon_r(freq)

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 1.3. Wiley.

    Parameters
    ----------
    epr : Param, default=1.0
        Real relative permittivity.
    tand : Param, default=0.0
        Dielectric loss tangent.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: Real relative permittivity
    epr: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Dielectric loss tangent
    tand: Param = param(default=0.0, constraint=Positive())

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        ones = jnp.ones(freq.npoints)
        eps = self.epr * ones - 1j * self.epr * self.tand * ones
        return self.with_conduction(eps, freq)


class DjordjevicSarkar(AbstractDielectric):
    r"""
    Causal wideband Debye (Djordjevic-Sarkar / Svensson-Dermer) dielectric.

    A continuous distribution of relaxation times between `flow` and `fhigh`
    gives a permittivity that falls slowly with frequency while keeping an
    almost constant loss tangent, and which satisfies the Kramers-Kronig
    relations.

    **Mathematical Formulation**

    With $k = \ln\left(\frac{f_{high} + jf_{ref}}{f_{low} + jf_{ref}}\right)$ and
    $f_d(f) = \ln\left(\frac{f_{high} + jf}{f_{low} + jf}\right)$:

    $$\varepsilon_d = \frac{-\tan\delta \cdot \varepsilon_r'}{\mathrm{Im}(k)}
    \qquad
    \varepsilon_\infty = \varepsilon_r'\left(1 + \tan\delta\frac{\mathrm{Re}(k)}{\mathrm{Im}(k)}\right)$$

    $$\varepsilon_r(f) = \varepsilon_\infty + \varepsilon_d f_d(f)
    - j\frac{\sigma}{\omega\varepsilon_0}$$

    Here $\varepsilon_r'$ and $\tan\delta$ are the values measured at the
    reference frequency `fref`.

    References
    ----------
    Djordjevic, A. R., Biljic, R. M., Likar-Smiljanic, V. D., & Sarkar, T. K. (2001).
    Wideband frequency-domain characterization of FR-4 and time-domain causality.
    IEEE Transactions on Electromagnetic Compatibility, 43(4), 662-667.

    Svensson, C., & Dermer, G. E. (2001). Time domain modeling of lossy interconnects.
    IEEE Transactions on Advanced Packaging, 24(2), 191-196.

    Parameters
    ----------
    epr : Param, default=1.0
        Real relative permittivity at `fref`.
    tand : Param, default=0.0
        Dielectric loss tangent at `fref`.
    flow : Param, default=1e3
        Lower bound of the relaxation-time distribution in Hz.
    fhigh : Param, default=1e12
        Upper bound of the relaxation-time distribution in Hz.
    fref : Param, default=1e9
        Frequency in Hz at which `epr` and `tand` were measured.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: Real relative permittivity at `fref`
    epr: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Dielectric loss tangent at `fref`
    tand: Param = param(default=0.0, constraint=Positive())

    #: Lower bound of the relaxation-time distribution
    flow: Param = param(default=1e3, constraint=Positive())

    #: Upper bound of the relaxation-time distribution
    fhigh: Param = param(default=1e12, constraint=Positive())

    #: Frequency at which `epr` and `tand` were measured
    fref: Param = param(default=1e9, constraint=Positive())

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        f = freq.f
        k = jnp.log((self.fhigh + 1j * self.fref) / (self.flow + 1j * self.fref))
        fd = jnp.log((self.fhigh + 1j * f) / (self.flow + 1j * f))

        eps_d = -self.tand * self.epr / jnp.imag(k)
        eps_inf = self.epr * (1.0 + self.tand * jnp.real(k) / jnp.imag(k))

        return self.with_conduction(eps_inf + eps_d * fd, freq)


class DebyePole(Module):
    r"""
    A single Debye relaxation pole.

    **Mathematical Formulation**

    $$\Delta\varepsilon_r(f) = \frac{\Delta\varepsilon_r}{1 + jf/f_{r}}$$

    References
    ----------
    Debye, P. (1929). Polar Molecules. Chemical Catalog Company.

    Parameters
    ----------
    depr : Param, default=0.0
        Permittivity increment (relaxation strength) of the pole.
    frelax : Param, default=1e9
        Relaxation frequency of the pole in Hz.
    """
    #: Permittivity increment of the pole
    depr: Param = param(default=0.0, constraint=Positive())

    #: Relaxation frequency of the pole
    frelax: Param = param(default=1e9, constraint=Positive())

    def contribution(self, freq: Frequency) -> jnp.ndarray:
        """The pole's complex permittivity contribution."""
        return self.depr / (1.0 + 1j * freq.f / self.frelax)


def _as_poles(poles) -> tuple[DebyePole, ...]:
    """Coerce a sequence of poles or ``(depr, frelax)`` pairs into modules."""
    return tuple(
        pole if isinstance(pole, DebyePole) else DebyePole(*pole) for pole in poles
    )


class MultipoleDebye(AbstractDielectric):
    r"""
    N-pole Debye dielectric, with every coefficient fittable.

    **Mathematical Formulation**

    $$\varepsilon_r(f) = \varepsilon_\infty
    + \sum_{n} \frac{\Delta\varepsilon_{r,n}}{1 + jf/f_{r,n}}
    - j\frac{\sigma}{\omega\varepsilon_0}$$

    Example
    --------
    .. code-block:: python

        from pmrf.materials import MultipoleDebye

        material = MultipoleDebye(epinf=2.0, poles=[(1.0, 1e9), (0.5, 1e10)])

    References
    ----------
    Debye, P. (1929). Polar Molecules. Chemical Catalog Company.

    Parameters
    ----------
    epinf : Param, default=1.0
        High-frequency limit of the relative permittivity.
    poles : tuple of DebyePole, default=()
        The Debye poles. ``(depr, frelax)`` pairs are coerced automatically.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: High-frequency limit of the relative permittivity
    epinf: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: The Debye poles
    poles: tuple[DebyePole, ...] = field(default=(), converter=_as_poles)

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        eps = self.epinf * jnp.ones(freq.npoints, dtype=complex)
        for pole in self.poles:
            eps = eps + pole.contribution(freq)
        return self.with_conduction(eps, freq)


class ColeCole(AbstractDielectric):
    r"""
    Cole-Cole dielectric: a single relaxation broadened by the exponent $\alpha$.

    **Mathematical Formulation**

    $$\varepsilon_r(f) = \varepsilon_\infty
    + \frac{\Delta\varepsilon_r}{1 + \left(jf/f_r\right)^{1 - \alpha}}
    - j\frac{\sigma}{\omega\varepsilon_0}$$

    With $\alpha = 0$ this reduces exactly to a single Debye pole.

    References
    ----------
    Cole, K. S., & Cole, R. H. (1941). Dispersion and absorption in dielectrics.
    Journal of Chemical Physics, 9(4), 341-351.

    Parameters
    ----------
    epinf : Param, default=1.0
        High-frequency limit of the relative permittivity.
    depr : Param, default=0.0
        Permittivity increment (relaxation strength).
    frelax : Param, default=1e9
        Relaxation frequency in Hz.
    alpha : Param, default=0.0
        Broadening exponent, in [0, 1).
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: High-frequency limit of the relative permittivity
    epinf: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Permittivity increment
    depr: Param = param(default=0.0, constraint=Positive())

    #: Relaxation frequency
    frelax: Param = param(default=1e9, constraint=Positive())

    #: Broadening exponent
    alpha: Param = param(default=0.0, constraint=Interval(0.0, 1.0))

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        # Guard f = 0, where (jf/f_r)**(1-alpha) is a branch point.
        f = jnp.asarray(freq.f)
        safe_f = jnp.where(f > 0, f, 1.0)
        ratio = jnp.where(f > 0, (1j * safe_f / self.frelax) ** (1.0 - self.alpha), 0.0)

        eps = self.epinf + self.depr / (1.0 + ratio)
        return self.with_conduction(eps, freq)


class TabulatedDielectric(AbstractDielectric):
    r"""
    Dielectric linearly interpolated from tabulated vendor or measured data.

    **Mathematical Formulation**

    The real and imaginary parts are interpolated independently, and values
    outside the tabulated band are clamped to the end points:

    $$\varepsilon_r(f) = \mathrm{interp}(f, f_k, \mathrm{Re}\,\varepsilon_{r,k})
    + j\,\mathrm{interp}(f, f_k, \mathrm{Im}\,\varepsilon_{r,k})
    - j\frac{\sigma}{\omega\varepsilon_0}$$

    Interpolating the parts independently does not guarantee causality; the
    tabulated data is trusted as given.

    Parameters
    ----------
    f : jnp.ndarray
        Tabulated frequency points in Hz, strictly increasing.
    epr : jnp.ndarray
        Complex relative permittivity at each tabulated frequency. Real input is
        promoted to a lossless complex value.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m, added on top of the tabulated data.
    """
    #: Tabulated frequency points in Hz
    f: jnp.ndarray = field(converter=jnp.asarray)

    #: Complex relative permittivity at each tabulated frequency
    epr: jnp.ndarray = field(converter=lambda x: jnp.asarray(x, dtype=complex))

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def __post_init__(self):
        if self.f.shape != self.epr.shape:
            raise ValueError(
                "`f` and `epr` must have the same shape, got "
                f"{self.f.shape} and {self.epr.shape}"
            )

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        real = jnp.interp(freq.f, self.f, jnp.real(self.epr))
        imag = jnp.interp(freq.f, self.f, jnp.imag(self.epr))
        return self.with_conduction(real + 1j * imag, freq)


def as_dielectric(value) -> AbstractDielectric:
    """
    Coerce a value into a dielectric material.

    Accepts an existing :class:`AbstractDielectric`, a scalar permittivity, or an
    ``(epr, tand)`` or ``(epr, tand, sigma)`` tuple, which build a
    :class:`ConstantDielectric`.

    Parameters
    ----------
    value : Any
        The value to coerce.

    Returns
    -------
    AbstractDielectric
        The resulting dielectric material.
    """
    if isinstance(value, AbstractDielectric):
        return value
    if isinstance(value, (tuple, list)):
        return ConstantDielectric(*value)
    return ConstantDielectric(value)
