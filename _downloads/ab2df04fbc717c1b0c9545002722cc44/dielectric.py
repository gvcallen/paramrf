"""
Frequency-dependent dielectric materials.

A dielectric owns the relative permittivity, loss tangent and static
conductivity of a medium. Every model returns a *total* complex relative
permittivity, so callers never need to know how the loss was specified.
"""
from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from pmrf.parameters.constraints import Positive, NonNegative, GreaterThan, Interval
from pmrf.frequency import Frequency
from pmrf.modules.base import Module
from pmrf.parameters import Param, param
from pmrf.materials.properties import DielectricProperties
from pmrf.utils import field


class AbstractDielectric(Module):
    r"""Abstract base class for a dielectric material.

    Subclasses return the *total* complex relative permittivity, using the
    convention $\varepsilon_r = \varepsilon' - j\varepsilon''$ with
    $\varepsilon'' \geq 0$ for a passive medium.
    
    Static conductivity of the medium is included.
    """

    @abstractmethod
    def properties(self, freq: Frequency) -> DielectricProperties:
        """Evaluate the dielectric properties over a frequency axis.

        Parameters
        ----------
        freq : Frequency
            The frequency axis to evaluate over.

        Returns
        -------
        jnp.ndarray
            Complex permittivity, permeability, and static conductivity.
        """


class ConstantDielectric(AbstractDielectric):
    r"""
    Non-dispersive dielectric with a constant permittivity and loss tangent.

    This is the default material. It is not causal - a constant loss tangent
    cannot satisfy the Kramers-Kronig relations - but it is the conventional
    choice in a frequency-domain package. For a popular causal wideband alternative,
    see :class:`DjordjevicSarkarDielectric`.

    **Mathematical Formulation**

    $$\varepsilon_r(\omega) = \varepsilon_r' \left(1 - j\tan\delta\right)
    - j\frac{\sigma}{\omega\varepsilon_0}
    \qquad \mu_r(\omega) = \mu_r$$

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.materials import ConstantDielectric

        fr4 = ConstantDielectric(ep_r=4.3, tand=0.02)
        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        eps = fr4.properties(freq).ep_r

    References
    ----------
    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 1.3. Wiley.

    Parameters
    ----------
    ep_r : Param, default=1.0
        Real relative permittivity.
    tand : Param, default=0.0
        Dielectric loss tangent.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    mu_r : Param, default=1.0
        Relative permeability of the medium.
    """
    #: Real relative permittivity
    ep_r: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Dielectric loss tangent
    tand: Param = param(default=0.0, constraint=NonNegative())

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=NonNegative())

    #: Relative permeability of the medium
    mu_r: Param = param(default=1.0, constraint=Positive())

    def properties(self, freq: Frequency) -> DielectricProperties:
        ones = jnp.ones(freq.npoints)
        eps = self.ep_r * ones - 1j * self.ep_r * self.tand * ones
        return DielectricProperties(eps, self.mu_r * ones, self.sigma * ones)


class DjordjevicSarkarDielectric(AbstractDielectric):
    r"""
    Causal wideband Debye (Djordjevic-Sarkar / Svensson-Dermer) dielectric.

    A continuous distribution of relaxation times between `f_low` and `f_high`
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
    reference frequency `f_ref`.

    References
    ----------
    Djordjevic, A. R., Biljic, R. M., Likar-Smiljanic, V. D., & Sarkar, T. K. (2001).
    Wideband frequency-domain characterization of FR-4 and time-domain causality.
    IEEE Transactions on Electromagnetic Compatibility, 43(4), 662-667.

    Svensson, C., & Dermer, G. E. (2001). Time domain modeling of lossy interconnects.
    IEEE Transactions on Advanced Packaging, 24(2), 191-196.

    Parameters
    ----------
    ep_r : Param, default=1.0
        Real relative permittivity at `f_ref`.
    tand : Param, default=0.0
        Dielectric loss tangent at `f_ref`.
    f_low : Param, default=1e3
        Lower bound of the relaxation-time distribution in Hz.
    f_high : Param, default=1e12
        Upper bound of the relaxation-time distribution in Hz.
    f_ref : Param, default=1e9
        Frequency in Hz at which `ep_r` and `tand` were measured.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: Real relative permittivity at `f_ref`
    ep_r: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Dielectric loss tangent at `f_ref`
    tand: Param = param(default=0.0, constraint=NonNegative())

    #: Lower bound of the relaxation-time distribution
    f_low: Param = param(default=1e3, constraint=Positive())

    #: Upper bound of the relaxation-time distribution
    f_high: Param = param(default=1e12, constraint=Positive())

    #: Frequency at which `ep_r` and `tand` were measured
    f_ref: Param = param(default=1e9, constraint=Positive())

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=NonNegative())

    def properties(self, freq: Frequency) -> DielectricProperties:
        f = freq.f
        k = jnp.log((self.f_high + 1j * self.f_ref) / (self.f_low + 1j * self.f_ref))
        fd = jnp.log((self.f_high + 1j * f) / (self.f_low + 1j * f))

        eps_d = -self.tand * self.ep_r / jnp.imag(k)
        eps_inf = self.ep_r * (1.0 + self.tand * jnp.real(k) / jnp.imag(k))

        ones = jnp.ones(freq.npoints)
        return DielectricProperties(eps_inf + eps_d * fd, ones, self.sigma * ones)


class AbstractPole(Module):
    r"""Abstract base class for a single pole of a :class:`PoleDielectric`.

    A pole returns a dimensionless *increment* to a relative material constant.
    It adds to the relative permittivity when listed in
    :attr:`PoleDielectric.poles`, and to the relative permeability when listed in
    :attr:`PoleDielectric.mu_poles`. Increments use the same
    $\varepsilon' - j\varepsilon''$ convention as :class:`AbstractDielectric`.
    """

    @abstractmethod
    def contribution(self, freq: Frequency) -> jnp.ndarray:
        """The pole's dimensionless complex increment over a frequency axis.

        Parameters
        ----------
        freq : Frequency
            The frequency axis to evaluate over.

        Returns
        -------
        jnp.ndarray
            Complex increment, one value per frequency point.
        """


class DebyePole(AbstractPole):
    r"""
    A single Debye relaxation pole.

    **Mathematical Formulation**

    $$\Delta(f) = \frac{\Delta\varepsilon_r}{1 + jf/f_{relax}}$$

    In :attr:`PoleDielectric.mu_poles`, `dep_r` is a permeability increment.

    References
    ----------
    Debye, P. (1929). Polar Molecules. Chemical Catalog Company.

    Parameters
    ----------
    dep_r : Param, default=0.0
        Increment (relaxation strength) of the pole.
    f_relax : Param, default=1e9
        Relaxation frequency of the pole in Hz.
    """
    #: Increment (relaxation strength) of the pole
    dep_r: Param = param(default=0.0, constraint=NonNegative())

    #: Relaxation frequency of the pole
    f_relax: Param = param(default=1e9, constraint=Positive())

    def contribution(self, freq: Frequency) -> jnp.ndarray:
        return self.dep_r / (1.0 + 1j * freq.f / self.f_relax)


class LorentzPole(AbstractPole):
    r"""
    A single Lorentz (damped harmonic oscillator) resonance pole.

    **Mathematical Formulation**

    $$\Delta(f) = \frac{\Delta\varepsilon_r}
    {1 - \left(f/f_{res}\right)^2 + 2j\zeta f/f_{res}}$$

    The pole gives $\Delta(0) = \Delta\varepsilon_r$ at DC. With damping
    $\zeta > 0$ it is passive: $\mathrm{Im}\,\Delta(f) \leq 0$ for $f \geq 0$.
    With $\zeta = 0$ it is lossless,

    $$\Delta(f) = \frac{\Delta\varepsilon_r f_{res}^2}{f_{res}^2 - f^2},$$

    and singular at $f = f_{res}$, so an undamped pole must resonate outside the
    evaluated band. :meth:`contribution` raises if it does not.

    In :attr:`PoleDielectric.mu_poles`, `dep_r` is a permeability increment.

    References
    ----------
    Jackson, J. D. (1999). Classical Electrodynamics (3rd ed.), Section 7.5. Wiley.

    Parameters
    ----------
    dep_r : Param
        Increment (oscillator strength) of the pole, its value at DC.
    f_res : Param
        Resonant frequency of the pole in Hz.
    zeta : Param, default=0.0
        Dimensionless damping ratio.
    """
    #: Increment (oscillator strength) of the pole
    dep_r: Param = param(constraint=NonNegative())

    #: Resonant frequency of the pole
    f_res: Param = param(constraint=Positive())

    #: Dimensionless damping ratio
    zeta: Param = param(default=0.0, constraint=NonNegative())

    def contribution(self, freq: Frequency) -> jnp.ndarray:
        f = jnp.asarray(freq.f)
        f_res, zeta = jnp.asarray(self.f_res), jnp.asarray(self.zeta)
        # f_res is a Param, so the singularity can only be checked at run time.
        in_band = (zeta == 0) & (f_res >= jnp.min(f)) & (f_res <= jnp.max(f))
        f = eqx.error_if(
            f, in_band,
            "an undamped LorentzPole (zeta = 0) must resonate outside the evaluated band",
        )
        x = f / f_res
        return self.dep_r / (1.0 - x**2 + 2j * zeta * x)


def _as_poles(poles) -> tuple[AbstractPole, ...]:
    """Coerce a sequence of poles or ``(dep_r, f_relax)`` Debye pairs into modules."""
    return tuple(
        pole if isinstance(pole, AbstractPole) else DebyePole(*pole) for pole in poles
    )


class PoleDielectric(AbstractDielectric):
    r"""
    A base dielectric plus pole increments on its permittivity and permeability.

    The base supplies the high-frequency permittivity, permeability and static
    conductivity. Each pole adds a dimensionless increment, so the base's own
    constants are counted once. A Debye-pole base of
    ``ConstantDielectric(ep_r=ep_inf, sigma=sigma)`` is the classic N-pole Debye
    dielectric.

    **Mathematical Formulation**

    $$\varepsilon_r(f) = \varepsilon_{r,base}(f) + \sum_{n} \Delta_n(f)
    - j\frac{\sigma_{base}}{\omega\varepsilon_0}
    \qquad
    \mu_r(f) = \mu_{r,base}(f) + \sum_{m} \Delta_m(f)$$

    where $\Delta_n$ runs over `poles` and $\Delta_m$ over `mu_poles`.

    Example
    --------
    .. code-block:: python

        from pmrf.materials import ConstantDielectric, LorentzPole, PoleDielectric

        material = PoleDielectric(
            ConstantDielectric(ep_r=2.0),
            poles=[(1.0, 1e9), LorentzPole(dep_r=0.2, f_res=50e9, zeta=0.1)],
        )

    References
    ----------
    Debye, P. (1929). Polar Molecules. Chemical Catalog Company.

    Jackson, J. D. (1999). Classical Electrodynamics (3rd ed.), Section 7.5. Wiley.

    Parameters
    ----------
    base : AbstractDielectric, default=ConstantDielectric()
        The dielectric the poles add to. Scalars and tuples are coerced as by
        :func:`as_dielectric`.
    poles : tuple of AbstractPole, default=()
        Permittivity poles. ``(dep_r, f_relax)`` pairs are coerced to
        :class:`DebyePole`.
    mu_poles : tuple of AbstractPole, default=()
        Permeability poles, coerced as `poles`.
    """
    #: The dielectric the poles add to
    base: AbstractDielectric = field(
        default_factory=ConstantDielectric, converter=lambda x: as_dielectric(x)
    )

    #: Permittivity poles
    poles: tuple[AbstractPole, ...] = field(default=(), converter=_as_poles)

    #: Permeability poles
    mu_poles: tuple[AbstractPole, ...] = field(default=(), converter=_as_poles)

    def properties(self, freq: Frequency) -> DielectricProperties:
        base = self.base.properties(freq)
        ep_r = base.ep_r + sum(pole.contribution(freq) for pole in self.poles)
        mu_r = base.mu_r + sum(pole.contribution(freq) for pole in self.mu_poles)
        return DielectricProperties(ep_r, mu_r, base.sigma)


class ColeColeDielectric(AbstractDielectric):
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
    ep_inf : Param, default=1.0
        High-frequency limit of the relative permittivity.
    dep_r : Param, default=0.0
        Permittivity increment (relaxation strength).
    f_relax : Param, default=1e9
        Relaxation frequency in Hz.
    alpha : Param, default=0.0
        Broadening exponent, in [0, 1).
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: High-frequency limit of the relative permittivity
    ep_inf: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Permittivity increment
    dep_r: Param = param(default=0.0, constraint=NonNegative())

    #: Relaxation frequency
    f_relax: Param = param(default=1e9, constraint=Positive())

    #: Broadening exponent
    alpha: Param = param(default=0.0, constraint=Interval(0.0, 1.0))

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=NonNegative())

    def properties(self, freq: Frequency) -> DielectricProperties:
        # Guard f = 0, where (jf/f_r)**(1-alpha) is a branch point.
        f = jnp.asarray(freq.f)
        safe_f = jnp.where(f > 0, f, 1.0)
        ratio = jnp.where(f > 0, (1j * safe_f / self.f_relax) ** (1.0 - self.alpha), 0.0)

        eps = self.ep_inf + self.dep_r / (1.0 + ratio)
        ones = jnp.ones(freq.npoints)
        return DielectricProperties(eps, ones, self.sigma * ones)


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
    ep_r : jnp.ndarray
        Complex relative permittivity at each tabulated frequency. Real input is
        promoted to a lossless complex value.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m, added on top of the tabulated data.
    """
    #: Tabulated frequency points in Hz
    f: jnp.ndarray = field(converter=jnp.asarray)

    #: Complex relative permittivity at each tabulated frequency
    ep_r: jnp.ndarray = field(converter=lambda x: jnp.asarray(x, dtype=complex))

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=NonNegative())

    def __check_init__(self):
        if self.f.ndim != 1 or self.ep_r.ndim != 1:
            raise ValueError("`f` and `ep_r` must be one-dimensional")
        if self.f.size == 0:
            raise ValueError("tabulated dielectric data must be nonempty")
        if self.f.shape != self.ep_r.shape:
            raise ValueError(
                "`f` and `ep_r` must have the same shape, got "
                f"{self.f.shape} and {self.ep_r.shape}"
            )
        if not bool(jnp.all(jnp.diff(self.f) > 0)):
            raise ValueError("`f` must be strictly increasing")

    def properties(self, freq: Frequency) -> DielectricProperties:
        real = jnp.interp(freq.f, self.f, jnp.real(self.ep_r))
        imag = jnp.interp(freq.f, self.f, jnp.imag(self.ep_r))
        ones = jnp.ones(freq.npoints)
        return DielectricProperties(real + 1j * imag, ones, self.sigma * ones)


def as_dielectric(value) -> AbstractDielectric:
    """
    Coerce a value into a dielectric material.

    Accepts an existing :class:`AbstractDielectric`, a scalar permittivity, or an
    ``(ep_r, tand)`` or ``(ep_r, tand, sigma)`` tuple, which build a
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
