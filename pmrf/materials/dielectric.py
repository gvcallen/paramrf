"""
Frequency-dependent dielectric materials.

A dielectric owns the relative permittivity, loss tangent and static
conductivity of a medium. Every model returns a *total* complex relative
permittivity, so callers never need to know how the loss was specified.
"""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from scipy.constants import epsilon_0

from pmrf.constraints import Positive, GreaterThan, Interval
from pmrf.frequency import Frequency
from pmrf.modules.base import Module
from pmrf.parameters import Param, param, as_param
from pmrf.utils import field


def _conductivity_term(sigma, w) -> jnp.ndarray:
    r"""Imaginary permittivity contribution $\sigma / (\omega \varepsilon_0)$.

    Guarded at DC with the double-``where`` pattern, so both the value and its
    gradient stay finite when the frequency axis includes $\omega = 0$.
    """
    w = jnp.asarray(w)
    safe_w = jnp.where(w > 0, w, 1.0)
    return jnp.where(w > 0, sigma / (safe_w * epsilon_0), 0.0)


class AbstractDielectric(Module):
    r"""Abstract base class for a dielectric material.

    Subclasses return the *total* complex relative permittivity, using the
    convention $\varepsilon_r = \varepsilon' - j\varepsilon''$ with
    $\varepsilon'' \geq 0$ for a passive medium.
    """

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
        raise NotImplementedError

    def loss_tangent(self, freq: Frequency) -> jnp.ndarray:
        r"""Effective loss tangent $-\mathrm{Im}(\varepsilon_r)/\mathrm{Re}(\varepsilon_r)$."""
        eps = self.epsilon_r(freq)
        return -jnp.imag(eps) / jnp.real(eps)


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

        fr4 = ConstantDielectric(eps_r=4.3, tand=0.02)
        freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')
        eps = fr4.epsilon_r(freq)

    Parameters
    ----------
    eps_r : Param, default=1.0
        Real relative permittivity.
    tand : Param, default=0.0
        Dielectric loss tangent.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: Real relative permittivity
    eps_r: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Dielectric loss tangent
    tand: Param = param(default=0.0, constraint=Positive())

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        ones = jnp.ones(freq.npoints)
        eps_real = self.eps_r * ones
        eps_imag = self.eps_r * self.tand * ones + _conductivity_term(self.sigma, freq.w)
        return eps_real - 1j * eps_imag


class DjordjevicSarkar(AbstractDielectric):
    r"""
    Causal wideband Debye (Djordjevic-Sarkar / Svensson-Dermer) dielectric.

    A continuous distribution of relaxation times between ``f_low`` and
    ``f_high`` gives a permittivity that falls slowly with frequency while
    keeping an almost constant loss tangent, and which satisfies the
    Kramers-Kronig relations.

    **Mathematical Formulation**

    With $k = \ln\left(\frac{f_{high} + jf_{ref}}{f_{low} + jf_{ref}}\right)$ and
    $f_d(f) = \ln\left(\frac{f_{high} + jf}{f_{low} + jf}\right)$:

    $$\varepsilon_d = \frac{-\tan\delta \cdot \varepsilon_r'}{\mathrm{Im}(k)}
    \qquad
    \varepsilon_\infty = \varepsilon_r'\left(1 + \tan\delta\frac{\mathrm{Re}(k)}{\mathrm{Im}(k)}\right)$$

    $$\varepsilon_r(f) = \varepsilon_\infty + \varepsilon_d f_d(f)
    - j\frac{\sigma}{\omega\varepsilon_0}$$

    Here $\varepsilon_r'$ and $\tan\delta$ are the values measured at the
    reference frequency ``f_ref``.

    References
    ----------
    Djordjevic, A. R., Biljic, R. M., Likar-Smiljanic, V. D., & Sarkar, T. K. (2001).
    Wideband frequency-domain characterization of FR-4 and time-domain causality.
    IEEE Transactions on Electromagnetic Compatibility, 43(4), 662-667.

    Svensson, C., & Dermer, G. E. (2001). Time domain modeling of lossy interconnects.
    IEEE Transactions on Advanced Packaging, 24(2), 191-196.

    Parameters
    ----------
    eps_r : Param, default=1.0
        Real relative permittivity at ``f_ref``.
    tand : Param, default=0.0
        Dielectric loss tangent at ``f_ref``.
    f_low : Param, default=1e3
        Lower bound of the relaxation-time distribution in Hz.
    f_high : Param, default=1e12
        Upper bound of the relaxation-time distribution in Hz.
    f_ref : Param, default=1e9
        Frequency in Hz at which ``eps_r`` and ``tand`` were measured.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: Real relative permittivity at `f_ref`
    eps_r: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Dielectric loss tangent at `f_ref`
    tand: Param = param(default=0.0, constraint=Positive())

    #: Lower bound of the relaxation-time distribution
    f_low: Param = param(default=1e3, constraint=Positive())

    #: Upper bound of the relaxation-time distribution
    f_high: Param = param(default=1e12, constraint=Positive())

    #: Frequency at which `eps_r` and `tand` were measured
    f_ref: Param = param(default=1e9, constraint=Positive())

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        f = freq.f
        k = jnp.log((self.f_high + 1j * self.f_ref) / (self.f_low + 1j * self.f_ref))
        fd = jnp.log((self.f_high + 1j * f) / (self.f_low + 1j * f))

        eps_d = -self.tand * self.eps_r / jnp.imag(k)
        eps_inf = self.eps_r * (1.0 + self.tand * jnp.real(k) / jnp.imag(k))

        eps = eps_inf + eps_d * fd
        return eps - 1j * _conductivity_term(self.sigma, freq.w)


class DebyePole(Module):
    r"""
    A single Debye relaxation pole.

    Parameters
    ----------
    delta_eps : Param, default=0.0
        Permittivity increment (relaxation strength) of the pole.
    f_relax : Param, default=1e9
        Relaxation frequency of the pole in Hz.
    """
    #: Permittivity increment of the pole
    delta_eps: Param = param(default=0.0, constraint=Positive())

    #: Relaxation frequency of the pole
    f_relax: Param = param(default=1e9, constraint=Positive())

    def contribution(self, freq: Frequency) -> jnp.ndarray:
        r"""Complex permittivity contribution $\Delta\varepsilon / (1 + jf/f_r)$."""
        return self.delta_eps / (1.0 + 1j * freq.f / self.f_relax)


def _as_poles(poles) -> tuple[DebyePole, ...]:
    """Coerce a sequence of poles or ``(delta_eps, f_relax)`` pairs into modules."""
    return tuple(
        pole if isinstance(pole, DebyePole) else DebyePole(*pole) for pole in poles
    )


class MultipoleDebye(AbstractDielectric):
    r"""
    N-pole Debye dielectric, with every coefficient fittable.

    **Mathematical Formulation**

    $$\varepsilon_r(f) = \varepsilon_\infty
    + \sum_{n} \frac{\Delta\varepsilon_n}{1 + jf/f_{r,n}}
    - j\frac{\sigma}{\omega\varepsilon_0}$$

    Example
    --------
    .. code-block:: python

        from pmrf.materials import MultipoleDebye

        material = MultipoleDebye(eps_inf=2.0, poles=[(1.0, 1e9), (0.5, 1e10)])

    References
    ----------
    Debye, P. (1929). Polar Molecules. Chemical Catalog Company.

    Parameters
    ----------
    eps_inf : Param, default=1.0
        High-frequency limit of the relative permittivity.
    poles : tuple of DebyePole, default=()
        The Debye poles. ``(delta_eps, f_relax)`` pairs are coerced automatically.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: High-frequency limit of the relative permittivity
    eps_inf: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: The Debye poles
    poles: tuple[DebyePole, ...] = field(default=(), converter=_as_poles)

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        eps = self.eps_inf * jnp.ones(freq.npoints, dtype=complex)
        for pole in self.poles:
            eps = eps + pole.contribution(freq)
        return eps - 1j * _conductivity_term(self.sigma, freq.w)


class ColeCole(AbstractDielectric):
    r"""
    Cole-Cole dielectric: a single relaxation broadened by the exponent $\alpha$.

    **Mathematical Formulation**

    $$\varepsilon_r(f) = \varepsilon_\infty
    + \frac{\Delta\varepsilon}{1 + \left(jf/f_r\right)^{1 - \alpha}}
    - j\frac{\sigma}{\omega\varepsilon_0}$$

    With $\alpha = 0$ this reduces exactly to a single Debye pole.

    References
    ----------
    Cole, K. S., & Cole, R. H. (1941). Dispersion and absorption in dielectrics.
    Journal of Chemical Physics, 9(4), 341-351.

    Parameters
    ----------
    eps_inf : Param, default=1.0
        High-frequency limit of the relative permittivity.
    delta_eps : Param, default=0.0
        Permittivity increment (relaxation strength).
    f_relax : Param, default=1e9
        Relaxation frequency in Hz.
    alpha : Param, default=0.0
        Broadening exponent, in [0, 1).
    sigma : Param, default=0.0
        Static bulk conductivity in S/m.
    """
    #: High-frequency limit of the relative permittivity
    eps_inf: Param = param(default=1.0, constraint=GreaterThan(1.0))

    #: Permittivity increment
    delta_eps: Param = param(default=0.0, constraint=Positive())

    #: Relaxation frequency
    f_relax: Param = param(default=1e9, constraint=Positive())

    #: Broadening exponent
    alpha: Param = param(default=0.0, constraint=Interval(0.0, 1.0))

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        # Guard f = 0, where (jf/f_r)**(1-alpha) is a branch point.
        f = jnp.asarray(freq.f)
        safe_f = jnp.where(f > 0, f, 1.0)
        ratio = jnp.where(f > 0, (1j * safe_f / self.f_relax) ** (1.0 - self.alpha), 0.0)

        eps = self.eps_inf + self.delta_eps / (1.0 + ratio)
        return eps - 1j * _conductivity_term(self.sigma, freq.w)


class TabulatedDielectric(AbstractDielectric):
    r"""
    Dielectric interpolated from tabulated vendor or measured data.

    Values outside the tabulated band are clamped to the end points. The real
    and imaginary parts are interpolated independently.

    Parameters
    ----------
    f : jnp.ndarray
        Tabulated frequency points in Hz, strictly increasing.
    eps_r_values : jnp.ndarray
        Complex relative permittivity at each tabulated frequency. Real input is
        promoted to a lossless complex value.
    sigma : Param, default=0.0
        Static bulk conductivity in S/m, added on top of the tabulated data.
    method : {'linear'}, default='linear'
        Interpolation method.
    """
    #: Tabulated frequency points in Hz
    f: jnp.ndarray = field(converter=jnp.asarray)

    #: Complex relative permittivity at each tabulated frequency
    eps_r_values: jnp.ndarray = field(converter=lambda x: jnp.asarray(x, dtype=complex))

    #: Static bulk conductivity in S/m
    sigma: Param = param(default=0.0, constraint=Positive())

    #: Interpolation method
    method: Literal["linear"] = field(default="linear", static=True)

    def __post_init__(self):
        if self.method != "linear":
            raise ValueError(f"Unknown interpolation method: {self.method!r}")
        if self.f.shape != self.eps_r_values.shape:
            raise ValueError(
                "`f` and `eps_r_values` must have the same shape, got "
                f"{self.f.shape} and {self.eps_r_values.shape}"
            )

    def epsilon_r(self, freq: Frequency) -> jnp.ndarray:
        real = jnp.interp(freq.f, self.f, jnp.real(self.eps_r_values))
        imag = jnp.interp(freq.f, self.f, jnp.imag(self.eps_r_values))
        return real + 1j * imag - 1j * _conductivity_term(self.sigma, freq.w)


def as_dielectric(value) -> AbstractDielectric:
    """
    Coerce a value into a dielectric material.

    Accepts an existing :class:`AbstractDielectric`, a scalar permittivity, or an
    ``(eps_r, tand)`` or ``(eps_r, tand, sigma)`` tuple, which build a
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


