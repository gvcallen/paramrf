"""
Closed-form physics for transmission lines (coaxial, microstrip).

A formulation is pure numerics. Materials are evaluated by the line, so a
formulation never sees a :class:`~pmrf.Param` or a :class:`~pmrf.Module`
parameter: it can be checked directly against the equations of the paper it
comes from, and contributed without learning the material taxonomy.
"""
from abc import abstractmethod

from scipy.constants import c, mu_0, epsilon_0
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.materials import AbstractConductor
from pmrf.models.components.lines.base import ImmittanceResult


class QuasiStaticResult(eqx.Module):
    r"""
    Quasi-static solution of a planar line.

    A quasi-static formulation stops at the effective permittivity, the
    characteristic impedance it implies and the effective conductor width; the
    line turns those into an immittance, and a dispersion model may correct them
    first.

    Parameters
    ----------
    eps_eff : jnp.ndarray
        Complex effective relative permittivity, shape ``(npoints,)``.
    zc : jnp.ndarray
        Quasi-static characteristic impedance in ohms, $Z_a/\sqrt{\varepsilon_e}$.
    w_eff : jnp.ndarray
        Effective conductor width in meters, carrying the series loss.
    """
    #: Complex effective relative permittivity
    eps_eff: jnp.ndarray

    #: Quasi-static characteristic impedance in ohms
    zc: jnp.ndarray

    #: Effective conductor width in meters
    w_eff: jnp.ndarray

    def to_immittance(self, freq: Frequency, zs: jnp.ndarray) -> ImmittanceResult:
        r"""
        Converts the quasi-static solution into a per-unit-length immittance.

        The external inductance and the shunt admittance follow from the
        quasi-static impedance and effective permittivity, and the signal
        conductor and its return path each add their surface impedance over the
        effective width:
        $$Z = \frac{j\omega Z_c \sqrt{\varepsilon_e}}{c} + \frac{2 Z_s}{W_{eff}}
        \qquad
        Y = \frac{j\omega \sqrt{\varepsilon_e}}{Z_c c}$$

        $\varepsilon_e$ is complex, so $Y$ already carries the dielectric loss
        as its real part and needs no separate loss-tangent term.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        zs : jnp.ndarray
            Complex surface impedance of the conductor in ohm per square.

        Returns
        -------
        ImmittanceResult
            The series impedance and shunt admittance.

        References
        ----------
        Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.8. Wiley.
        """
        w = freq.w
        sqrt_eps_eff = jnp.sqrt(self.eps_eff)

        Z = 1j * w * self.zc * sqrt_eps_eff / c + 2 * zs / self.w_eff
        Y = 1j * w * sqrt_eps_eff / (self.zc * c)

        return ImmittanceResult(Z=Z, Y=Y, w=w)


class AbstractCoaxialFormulation(eqx.Module):
    """
    Abstract base class for a closed-form coaxial line formulation.

    A formulation is pure numerics: every argument other than `conductor`
    arrives as an already-evaluated array, so it can be exercised directly
    against its published equations with no ParamRF objects in sight. Coaxial
    lines are the documented exception on `conductor`, because the internal
    impedance of a rod or tube depends on the conductor radius as well as on the
    surface impedance.
    """
    @abstractmethod
    def run(self, freq: Frequency, *, din, dout, eps_r, mu_r, conductor: AbstractConductor) -> ImmittanceResult:
        r"""
        Calculates the per-unit-length immittance of the line.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        din : ArrayLike
            Inner conductor diameter in meters.
        dout : ArrayLike
            Outer conductor inner diameter in meters.
        eps_r : jnp.ndarray
            Complex relative permittivity of the dielectric, shape ``(npoints,)``.
        mu_r : ArrayLike
            Relative permeability of the dielectric.
        conductor : AbstractConductor
            The conductor material of both conductors.

        Returns
        -------
        ImmittanceResult
            The series impedance and shunt admittance.
        """
        raise NotImplementedError


class TescheCoaxialFormulation(AbstractCoaxialFormulation):
    r"""
    Coaxial line formulation using the Tesche high-frequency approximation.

    **Mathematical Formulation**

    The external per-unit-length inductance and the shunt admittance follow from
    the coaxial geometry and the complex permittivity
    $\varepsilon = \varepsilon_0 \varepsilon_r$:
    $$L' = \frac{\mu_0 \mu_r}{2\pi} \ln\left(\frac{b}{a}\right)
    \qquad
    Y = \frac{j\omega 2\pi\varepsilon}{\ln(b/a)}$$

    which is $G + j\omega C$ with $C = 2\pi\Re(\varepsilon)/\ln(b/a)$ and
    $G = -2\pi\omega\Im(\varepsilon)/\ln(b/a)$. The internal impedance of the two
    conductors adds their surface impedance $Z_s$ over their circumferences:
    $$Z = j\omega L' + Z_s\left(\frac{1}{2\pi a} + \frac{1}{2\pi b}\right)$$

    Where $a$ is the inner radius and $b$ is the outer radius. In the strong
    skin-effect regime $Z_s = \sqrt{\omega\mu\rho/2}\,(1 + j)$, so the real part
    is Tesche's skin resistance and the imaginary part is $\omega$ times his skin
    inductance.

    References
    ----------
    Tesche, F. M. (2007). A Simple Model for the Line Parameters of a Lossy Coaxial
    Cable Filled With a Nondispersive Dielectric. IEEE Transactions on Electromagnetic
    Compatibility, 49(1), 12-17.

    Schelkunoff, S. A. (1934). The Electromagnetic Theory of Coaxial Transmission Lines
    and Cylindrical Shields. Bell System Technical Journal, 13(4), 532-579.
    """
    def run(self, freq: Frequency, *, din, dout, eps_r, mu_r, conductor: AbstractConductor) -> ImmittanceResult:
        eps = epsilon_0 * eps_r
        mu = mu_0 * mu_r
        w = freq.w

        a, b = din / 2, dout / 2
        lnbOvera = jnp.log(b / a)

        L_ext = jnp.ones(freq.npoints) * mu / (2 * jnp.pi) * lnbOvera

        # Internal impedance of the two conductors, per unit length: the surface
        # impedance in ohm per square spread over each circumference.
        zs = conductor.surface_impedance(freq)
        Z_int = zs * (1 / (2 * jnp.pi * a) + 1 / (2 * jnp.pi * b))

        Z = 1j * w * L_ext + Z_int
        Y = 1j * w * 2 * jnp.pi * eps / lnbOvera

        return ImmittanceResult(Z=Z, Y=Y, w=w)


class AbstractMicrostripFormulation(eqx.Module):
    """
    Abstract base class for a closed-form microstrip line formulation.

    A formulation is pure numerics: every argument arrives as an
    already-evaluated array, so it can be exercised directly against its
    published equations with no ParamRF objects in sight.
    """
    @abstractmethod
    def run(self, freq: Frequency, *, w, h, t, eps_r, zs) -> QuasiStaticResult:
        r"""
        Calculates the quasi-static solution of the line.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        w : ArrayLike
            Width of the microstrip trace in meters.
        h : ArrayLike
            Height of the dielectric substrate in meters.
        t : ArrayLike | None
            Thickness of the trace in meters, or None for a zero-thickness trace.
        eps_r : jnp.ndarray
            Complex relative permittivity of the substrate, shape ``(npoints,)``.
        zs : jnp.ndarray
            Complex surface impedance of the conductor in ohm per square,
            shape ``(npoints,)``.

        Returns
        -------
        QuasiStaticResult
            The effective permittivity, impedance and effective width.
        """
        raise NotImplementedError


class WheelerMicrostripFormulation(AbstractMicrostripFormulation):
    r"""
    Microstrip line formulation using the standard Wheeler approximations.

    **Mathematical Formulation**

    With ratio $u = W/H$, the effective relative permittivity ($\varepsilon_e$)
    and the impedance of the same geometry in air ($Z_a$) are:
    $$\varepsilon_e = \frac{\varepsilon_r + 1}{2} + \frac{\varepsilon_r - 1}{2} \frac{1}{\sqrt{1 + 12/u}}$$
    $$Z_a = \frac{120\pi}{u + 1.393 + 0.667 \ln(u + 1.444)} \quad (u > 1)
    \qquad
    Z_a = 60 \ln\left(\frac{8}{u} + \frac{u}{4}\right) \quad (u \leq 1)$$
    $$Z_c = \frac{Z_a}{\sqrt{\varepsilon_e}}$$

    $\varepsilon_r$ is complex, and $\varepsilon_e$ is linear in it, so the
    dielectric loss carries through the same filling factor as the real part and
    needs no separate loss-tangent term. The effective width is $W$: the
    approximation is derived for a zero-thickness strip, so `zs` does not enter
    here — a thickness-aware formulation uses it to widen $W_{eff}$ for the
    current distribution, and the line applies the series loss either way.

    References
    ----------
    Wheeler, H. A. (1977). Transmission-Line Properties of a Strip on a Dielectric Sheet on a Plane.
    IEEE Transactions on Microwave Theory and Techniques.
    """
    def run(self, freq: Frequency, *, w, h, t, eps_r, zs) -> QuasiStaticResult:
        if t is not None:
            raise ValueError("Wheeler microstrip approximation does not support finite thickness")

        W, H = w, h
        u = W / H

        # Shared base terms
        t1 = (eps_r + 1) / 2
        t2 = (eps_r - 1) / 2
        t3 = 1 / jnp.sqrt(1 + 12 / u)

        # Piecewise effective permittivity (eps_eff)
        eps_eff_le1 = t1 + t2 * (t3 + 0.04 * (1 - u)**2)
        eps_eff_gt1 = t1 + t2 * t3
        eps_eff = jnp.where(u <= 1.0, eps_eff_le1, eps_eff_gt1) * jnp.ones(freq.npoints)

        # Piecewise characteristic impedance in air (Za)
        Za_le1 = 60 * jnp.log(8 / u + 0.25 * u)
        Za_gt1 = (120 * jnp.pi) / (u + 1.393 + 0.667 * jnp.log(u + 1.444))
        Za = jnp.where(u <= 1.0, Za_le1, Za_gt1)

        zc = Za / jnp.sqrt(eps_eff)
        w_eff = W * jnp.ones(freq.npoints)

        return QuasiStaticResult(eps_eff=eps_eff, zc=zc, w_eff=w_eff)

