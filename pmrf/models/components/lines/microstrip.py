"""Microstrip models, formulations, and current distributions."""
from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import c, epsilon_0, mu_0

from pmrf.constraints import Positive
from pmrf.frequency import Frequency
from pmrf.materials import BulkConductor, ConstantDielectric, Substrate, as_substrate
from pmrf.materials.conductor_shape import AbstractConductorShape, HalfSpaceShape, RootSumSquareSlabShape
from pmrf.models.components.lines.base import AbstractImmittanceLine, ImmittanceResult
from pmrf.models.components.lines.planar import AbstractCurrentDistribution, AbstractPlanarCrossSection, PlanarQuasiStaticResult
from pmrf.parameters import Param, param
from pmrf.utils import field

_DEFAULT_DISPERSION = object()

class MicrostripCrossSection(AbstractPlanarCrossSection):
    """Cross-section of a strip on a grounded dielectric sheet.

    Parameters
    ----------
    w : ArrayLike
        Width of the strip in meters.
    h : ArrayLike
        Substrate height in meters.
    t : ArrayLike | None, default=None
        Strip thickness in meters, or ``None`` when it is unspecified.
    """

    #: Width of the strip in meters
    w: jnp.ndarray

    #: Substrate height in meters
    h: jnp.ndarray

    #: Strip thickness in meters, or ``None`` when unspecified
    t: jnp.ndarray | None = None

    def dimensions(self) -> dict:
        return {"w": self.w, "t": self.t}


class WheelerCurrentDistribution(AbstractCurrentDistribution[MicrostripCrossSection]):
    r"""Wheeler's incremental-inductance current distribution.

    This implements Wheeler's 1942 skin-effect rule. It is distinct from the
    1977 quasi-static impedance approximation implemented by
    :class:`WheelerMicrostripFormulation`.

    **Mathematical Formulation**

    $$k_c = \frac{2}{W}\exp\left[-1.2\left(\frac{\Re(Z_c)}{Z_0}\right)^{0.7}\right]$$

    Here $W$ is the physical trace width and $Z_c$ is the solved
    characteristic impedance. An unspecified thickness uses
    :class:`HalfSpaceShape`; otherwise, the distribution uses
    :attr:`slab_shape`. The returned weight is in inverse metres.

    References
    ----------
    Wheeler, H. A. (1942). Formulas for the Skin Effect. Proceedings of the
    IRE, 30(9), 412-424.
    """

    cross_section_type: ClassVar[type] = MicrostripCrossSection

    #: Finite-thickness conductor shape. The default matches the dc and
    #: strong-skin limits under Wheeler's geometry weight. See
    #: :class:`~pmrf.materials.conductor_shape.AbstractConductorShape` for
    #: normalisation details.
    slab_shape: AbstractConductorShape = eqx.field(
        default_factory=RootSumSquareSlabShape
    )

    def _distribute(self, freq, cross_section, quasi_static):
        z0 = jnp.sqrt(mu_0 / epsilon_0)
        zc = quasi_static.zc
        weight = 2 / cross_section.w * jnp.exp(-1.2 * (jnp.real(zc) / z0) ** 0.7)
        shape = HalfSpaceShape() if cross_section.t is None else self.slab_shape
        return ((shape, weight),)


class AbstractMicrostripFormulation(eqx.Module):
    """Abstract base class for a closed-form microstrip formulation."""
    @abstractmethod
    def quasi_static(self, *, w, h, t, ep_r) -> PlanarQuasiStaticResult:
        r"""
        Calculate the quasi-static solution.

        Parameters
        ----------
        w : ArrayLike
            Width of the microstrip trace in meters.
        h : ArrayLike
            Height of the dielectric substrate in meters.
        t : ArrayLike | None
            Trace thickness in meters, or ``None`` when unspecified.
        ep_r : jnp.ndarray
            Complex relative permittivity of the substrate, shape ``(npoints,)``.

        Returns
        -------
        PlanarQuasiStaticResult
            The effective permittivity, impedance and effective width.
        """
        raise NotImplementedError


class WheelerMicrostripFormulation(AbstractMicrostripFormulation):
    r"""
    Microstrip line formulation using the standard Wheeler approximations.

    This implements Wheeler's 1977 quasi-static approximation. It is distinct
    from the 1942 conductor-loss rule in
    :class:`WheelerCurrentDistribution`
    and does not calculate conductor loss.

    **Mathematical Formulation**

    With ratio $u = W/H$, the effective relative permittivity ($\varepsilon_e$)
    and the impedance of the same geometry in air ($Z_a$) are:
    $$\varepsilon_e = \frac{\varepsilon_r + 1}{2} + \frac{\varepsilon_r - 1}{2} \frac{1}{\sqrt{1 + 12/u}}$$
    $$Z_a = \frac{120\pi}{u + 1.393 + 0.667 \ln(u + 1.444)} \quad (u > 1)
    \qquad
    Z_a = 60 \ln\left(\frac{8}{u} + \frac{u}{4}\right) \quad (u \leq 1)$$
    $$Z_c = \frac{Z_a}{\sqrt{\varepsilon_e}}$$

    Complex $\varepsilon_r$ carries dielectric loss through the same filling
    factor. The effective width is $W$.

    **Validity**

    Derived for a zero-thickness strip on an isotropic, non-magnetic substrate.
    Finite thickness is ignored by this formulation but remains available to
    the current distribution. Modal dispersion requires a separate
    :class:`AbstractMicrostripDispersion`.

    References
    ----------
    Wheeler, H. A. (1977). Transmission-Line Properties of a Strip on a Dielectric Sheet on a Plane.
    IEEE Transactions on Microwave Theory and Techniques.
    """
    def quasi_static(self, *, w, h, t, ep_r) -> PlanarQuasiStaticResult:
        # t is accepted and ignored: the 1977 result is derived for a
        # zero-thickness strip, so thickness enters neither ep_eff nor zc. It
        # still reaches the current distribution through the line's
        # cross-section, where it refines conductor loss.
        W, H = w, h
        u = W / H

        # Shared base terms
        t1 = (ep_r + 1) / 2
        t2 = (ep_r - 1) / 2
        t3 = 1 / jnp.sqrt(1 + 12 / u)

        # Piecewise effective permittivity (ep_eff)
        ep_eff_le1 = t1 + t2 * (t3 + 0.04 * (1 - u)**2)
        ep_eff_gt1 = t1 + t2 * t3
        ep_eff = jnp.where(u <= 1.0, ep_eff_le1, ep_eff_gt1) * jnp.ones_like(ep_r)

        # Piecewise characteristic impedance in air (Za)
        Za_le1 = 60 * jnp.log(8 / u + 0.25 * u)
        Za_gt1 = (120 * jnp.pi) / (u + 1.393 + 0.667 * jnp.log(u + 1.444))
        Za = jnp.where(u <= 1.0, Za_le1, Za_gt1)

        zc = Za / jnp.sqrt(ep_eff)
        w_eff = W * jnp.ones_like(ep_r)
        conductance_factor = _microstrip_conductance_factor(ep_r, ep_eff, zc)
        return PlanarQuasiStaticResult(ep_eff, zc, w_eff, conductance_factor)


class HammerstadJensenMicrostripFormulation(AbstractMicrostripFormulation):
    r"""
    Hammerstad--Jensen quasi-static microstrip formulation.

    **Mathematical Formulation**

    For normalized width $u=W/H$, the impedance of the homogeneous geometry is
    $$Z_L(u) = \frac{Z_0}{2\pi}\ln\left[\frac{F(u)}{u}
    + \sqrt{1 + \left(\frac{2}{u}\right)^2}\right]$$
    $$F(u) = 6 + (2\pi - 6)\exp\left[-(30.666/u)^{0.7528}\right].$$

    With normalized thickness $t_n=T/H$, finite conductor thickness changes the
    normalized widths in air and in the dielectric:
    $$\Delta u_1 = \frac{t_n}{\pi}\ln\left[1 + \frac{4e}{t_n}
    \tanh^2\left(\sqrt{6.517u}\right)\right]$$
    $$\Delta u_r = \frac{\Delta u_1}{2}\left[1 +
    \operatorname{sech}\left(\sqrt{\varepsilon_r-1}\right)\right],
    \qquad u_1=u+\Delta u_1,\quad u_r=u+\Delta u_r.$$

    The empirical exponents are
    $$a = 1 + \frac{1}{49}\ln\left[\frac{u_r^4+(u_r/52)^2}
    {u_r^4+0.432}\right] + \frac{1}{18.7}\ln\left[1+(u_r/18.1)^3\right]$$
    $$b = 0.564\left(\frac{\varepsilon_r-0.9}{\varepsilon_r+3}\right)^{0.053}.$$
    Defining
    $$e = \frac{\varepsilon_r + 1}{2}
    + \frac{\varepsilon_r - 1}{2}(1 + 10/u_r)^{-ab},$$
    the returned quantities are
    $$Z_c=\frac{Z_L(u_r)}{\sqrt{e}},\qquad
    \varepsilon_e=e\left[\frac{Z_L(u_1)}{Z_L(u_r)}\right]^2,
    \qquad W_{eff}=u_rH,$$
    $W_{eff}$ is passed to the dispersion formulation. Conductor loss is
    calculated separately by the current distribution.

    The fractional power in $b$ is evaluated through the principal logarithm.
    The dielectric constraint $\Re(\varepsilon_r)>1$ keeps its argument away
    from the negative-real branch cut for passive materials.

    **Validity**

    The quoted error in $\varepsilon_e$ is below 0.2% for
    $\varepsilon_r<128$ and $0.01\leq u\leq100$. The quoted error in $Z_L$
    is below 0.01% for $u\leq1$ and 0.03% for $u\leq1000$. The formulation
    is quasi-static and requires a non-magnetic substrate. Inputs outside the
    fitted ranges are extrapolated.

    References
    ----------
    Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
    Computer-Aided Design. IEEE MTT-S International Microwave Symposium Digest,
    407-409.
    """

    def quasi_static(self, *, w, h, t, ep_r) -> PlanarQuasiStaticResult:
        u = w / h
        thickness = None if t is None else t / h

        du1 = 0.0
        if thickness is not None:
            du1 = thickness / jnp.pi * jnp.log(
                1 + 4 * jnp.e / thickness * jnp.tanh(jnp.sqrt(6.517 * u)) ** 2
            )

        dur = du1 * (1 + 1 / jnp.cosh(jnp.sqrt(ep_r - 1))) / 2
        u1 = u + du1
        ur = u + dur

        zr = self._homogeneous_impedance(ur)
        z1 = self._homogeneous_impedance(u1)

        a = (
            1
            + jnp.log((ur**4 + (ur / 52) ** 2) / (ur**4 + 0.432)) / 49
            + jnp.log(1 + (ur / 18.1) ** 3) / 18.7
        )
        b_argument = (ep_r - 0.9) / (ep_r + 3)
        b = 0.564 * jnp.exp(0.053 * jnp.log(b_argument))
        e = (ep_r + 1) / 2 + (ep_r - 1) / 2 * (1 + 10 / ur) ** (-a * b)

        zc = zr / jnp.sqrt(e)
        ep_eff = e * (z1 / zr) ** 2
        w_eff = ur * h
        conductance_factor = _microstrip_conductance_factor(ep_r, ep_eff, zc)
        return PlanarQuasiStaticResult(ep_eff, zc, w_eff, conductance_factor)

    @staticmethod
    def _homogeneous_impedance(u):
        z0 = jnp.sqrt(mu_0 / epsilon_0)
        f_u = 6 + (2 * jnp.pi - 6) * jnp.exp(-(30.666 / u) ** 0.7528)
        return z0 / (2 * jnp.pi) * jnp.log(f_u / u + jnp.sqrt(1 + (2 / u) ** 2))


class AbstractMicrostripDispersion(eqx.Module):
    """Abstract modal-dispersion formulation for an inhomogeneous microstrip."""

    @abstractmethod
    def disperse(
        self, freq: Frequency, *, ep_eff_0, zc_0, ep_r, w_eff, h
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Return frequency-dependent $(\varepsilon_e, Z_c)$.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        ep_eff_0 : jnp.ndarray
            Quasi-static complex effective relative permittivity.
        zc_0 : jnp.ndarray
            Quasi-static characteristic impedance in ohms.
        ep_r : jnp.ndarray
            Complex relative permittivity of the substrate.
        w_eff : jnp.ndarray
            Electromagnetic effective conductor width in meters.
        h : ArrayLike
            Substrate height in meters.

        Returns
        -------
        tuple of jnp.ndarray
            The dispersed effective permittivity and characteristic impedance.
        """
        raise NotImplementedError


class KirschningJansenMicrostripDispersion(AbstractMicrostripDispersion):
    r"""
    Kirschning--Jansen modal dispersion for microstrip.

    **Mathematical Formulation**

    The model expresses the dispersed effective permittivity as
    $$\varepsilon_e(f) = \varepsilon_r
    - \frac{\varepsilon_r-\varepsilon_e(0)}{1 + P(f)},$$
    where $P=P_1P_2(0.1844+P_3P_4)^{1.5763}f_n^{1.5763}$ and the $P_i$
    are
    $$P_1=0.27488+\left[0.6315+\frac{0.525}{(1+0.0157f_n)^{20}}\right]u
    -0.065683e^{-8.7513u}$$
    $$P_2=0.33622(1-e^{-0.03442\varepsilon_r}),\quad
    P_3=0.0363e^{-4.6u}[1-e^{-(f_n/38.7)^{4.97}}]$$
    $$P_4=1+2.751[1-e^{-(\varepsilon_r/15.916)^8}].$$
    The normalized frequency is
    $f_n=f[\mathrm{Hz}]H[\mathrm{m}]10^{-6}$ (GHz-mm).
    The normalized width is the thickness-corrected $u = W_{eff}/H$.

    Characteristic impedance is corrected by
    $$Z_c(f)=Z_c(0)\left(\frac{R_{13}}{R_{14}}\right)^{R_{17}},$$
    with
    $$R_1=\min(0.03891\varepsilon_r^{1.4},20),\quad
    R_2=\min(0.2671u^7,20),\quad R_3=4.766e^{-3.228u^{0.641}}$$
    $$R_4=0.016+(0.0514\varepsilon_r)^{4.524},\quad
    R_5=(f_n/28.843)^{12},\quad R_6=\min(22.20u^{1.92},20)$$
    $$R_7=1.206-0.3144e^{-R_1}(1-e^{-R_2})$$
    $$R_8=1+1.275\left[1-e^{-0.004625R_3\varepsilon_r^{1.674}
    (f_n/18.365)^{2.745}}\right]$$
    $$R_9=\frac{5.086R_4R_5e^{-R_6}}{(0.3838+0.386R_4)(1+1.2992R_5)}
    \frac{(\varepsilon_r-1)^6}{1+10(\varepsilon_r-1)^6}$$
    $$R_{10}=0.00044\varepsilon_r^{2.136}+0.0184,\quad
    R_{11}=\frac{(f_n/19.47)^6}{1+0.0962(f_n/19.47)^6},\quad
    R_{12}=\frac{1}{1+0.00245u^2}$$
    $$R_{13}=0.9408\varepsilon_e(f)^{R_8}-0.9603,\quad
    R_{14}=(0.9408-R_9)\varepsilon_e(0)^{R_8}-0.9603$$
    $$R_{15}=0.707R_{10}(f_n/12.3)^{1.097},\quad
    R_{16}=1+0.0503\varepsilon_r^2R_{11}[1-e^{-(u/15)^6}]$$
    $$R_{17}=R_7\left[1-\frac{1.1241R_{12}}{R_{16}}
    e^{-0.026f_n^{1.15656}-R_{15}}\right].$$

    The papers validate the fit from $\varepsilon_r=2.2$, while modal
    dispersion must vanish at the homogeneous $\varepsilon_r=1$ limit. In the
    extrapolation interval ParamRF therefore applies the smooth weight
    $$x=\operatorname{clip}\left(\frac{\Re(\varepsilon_r)-1}{1.2},0,1\right),
    \qquad q=x^2(3-2x),$$
    $$\varepsilon_e=(1-q)\varepsilon_e(0)+q\varepsilon_{e,KJ},\qquad
    Z_c=(1-q)Z_c(0)+qZ_{c,KJ}.$$
    This smooth extension to the homogeneous limit is specific to ParamRF.

    **Validity**

    The published fits cover $1\leq\varepsilon_r\leq20$,
    $0.1\leq W/H\leq100$, and $0\leq H/\lambda_0\leq0.13$, with numerical
    fits anchored at $\varepsilon_r\geq2.2$. ParamRF uses the extension above
    for $1<\varepsilon_r<2.2$ and extrapolates outside the remaining ranges.

    References
    ----------
    Kirschning, M., & Jansen, R. H. (1982). Accurate Model for Effective
    Dielectric Constant of Microstrip with Validity up to Millimeter-Wave
    Frequencies. Electronics Letters, 18(6), 272-273.

    Jansen, R. H., & Kirschning, M. (1983). Arguments and an Accurate Model for
    the Power-Current Formulation of Microstrip Characteristic Impedance.
    Archiv fuer Elektronik und Uebertragungstechnik, 37, 108-112.
    """

    def disperse(
        self, freq: Frequency, *, ep_eff_0, zc_0, ep_r, w_eff, h
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        u = w_eff / h
        fn = freq.f * h * 1e-6

        p1 = (
            0.27488
            + (0.6315 + 0.525 / (1 + 0.0157 * fn) ** 20) * u
            - 0.065683 * jnp.exp(-8.7513 * u)
        )
        p2 = 0.33622 * (1 - jnp.exp(-0.03442 * ep_r))
        p3 = 0.0363 * jnp.exp(-4.6 * u) * (1 - jnp.exp(-(fn / 38.7) ** 4.97))
        p4 = 1 + 2.751 * (1 - jnp.exp(-(ep_r / 15.916) ** 8))
        p_f = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
        ep_eff = ep_r - (ep_r - ep_eff_0) / (1 + p_f)

        r1_raw = 0.03891 * ep_r**1.4
        r1 = jnp.where(jnp.real(r1_raw) < 20, r1_raw, 20)
        r2 = jnp.minimum(0.2671 * u**7, 20)
        r3 = 4.766 * jnp.exp(-3.228 * u**0.641)
        r4 = 0.016 + (0.0514 * ep_r) ** 4.524
        r5 = (fn / 28.843) ** 12
        r6 = jnp.minimum(22.20 * u**1.92, 20)
        r7 = 1.206 - 0.3144 * jnp.exp(-r1) * (1 - jnp.exp(-r2))
        r8 = 1 + 1.275 * (
            1 - jnp.exp(-0.004625 * r3 * ep_r**1.674 * (fn / 18.365) ** 2.745)
        )
        r9 = (
            5.086
            * r4
            * r5
            / (0.3838 + 0.386 * r4)
            * jnp.exp(-r6)
            / (1 + 1.2992 * r5)
            * (ep_r - 1) ** 6
            / (1 + 10 * (ep_r - 1) ** 6)
        )
        r10 = 0.00044 * ep_r**2.136 + 0.0184
        r11 = (fn / 19.47) ** 6 / (1 + 0.0962 * (fn / 19.47) ** 6)
        r12 = 1 / (1 + 0.00245 * u**2)
        r13 = 0.9408 * ep_eff**r8 - 0.9603
        r14 = (0.9408 - r9) * ep_eff_0**r8 - 0.9603
        r15 = 0.707 * r10 * (fn / 12.3) ** 1.097
        r16 = 1 + 0.0503 * ep_r**2 * r11 * (1 - jnp.exp(-(u / 15) ** 6))
        r17 = r7 * (
            1
            - 1.1241
            * r12
            / r16
            * jnp.exp(-0.026 * fn**1.15656 - r15)
        )
        zc = zc_0 * (r13 / r14) ** r17

        # The published fit starts at epsilon_r=2.2, while modal dispersion must
        # vanish at the homogeneous epsilon_r=1 limit. Smoothly introduce the
        # empirical correction over that extrapolation interval. The smoothstep
        # has zero slope at both ends, keeping values and gradients continuous.
        fit_fraction = jnp.clip((jnp.real(ep_r) - 1) / 1.2, 0, 1)
        modal_weight = fit_fraction**2 * (3 - 2 * fit_fraction)
        ep_eff = ep_eff_0 + modal_weight * (ep_eff - ep_eff_0)
        zc = zc_0 + modal_weight * (zc - zc_0)
        return ep_eff, zc


def _microstrip_conductance_factor(ep_r, ep_eff, zc):
    """Convert static substrate conductivity through the quasi-static fill."""
    capacitance = jnp.sqrt(ep_eff) / (zc * c)
    delta = ep_r - 1
    safe_delta = jnp.where(jnp.abs(delta) > 1e-14, delta, 1.0)
    filling = jnp.where(
        jnp.abs(delta) > 1e-14,
        (capacitance - 1 / (zc * c * jnp.sqrt(ep_eff))) / safe_delta,
        0.0,
    )
    return jnp.real(filling) / epsilon_0


class MicrostripLine(AbstractImmittanceLine):
    r"""
    Microstrip line defined by geometry and materials.

    The defaults are :class:`HammerstadJensenMicrostripFormulation`,
    :class:`KirschningJansenMicrostripDispersion`, and
    :class:`WheelerCurrentDistribution`.
    :class:`WheelerMicrostripFormulation` is available as a zero-thickness
    quasi-static alternative.

    **Mathematical Formulation**

    Complex permittivity is propagated through the quasi-static and dispersion
    formulations, so $\varepsilon_e$ includes dielectric loss:
    $$\gamma_m=\frac{j\omega}{c}\sqrt{\varepsilon_e(f)}.$$

    Static conductivity contributes separately as $G=\sigma K_g$, avoiding a
    singular permittivity at dc. Microstrip formulations require
    $\mu_r=1$.

    Wheeler's current distribution gives
    $$\alpha_c=\frac{\Re(Z_s)}{\Re(Z_{c,loss})W}
    \exp\left[-1.2\left(\frac{\Re(Z_{c,loss})}{Z_0}\right)^{0.7}\right],$$
    using the physical width $W$ and the active quasi-static or dispersed
    $Z_c$. With finite thickness, the default slab shape adds
    $R_{dc}=1/(\sigma Wt)$ through
    $R=\sqrt{R_{dc}^2+R_{ac}^2}$. An unspecified thickness applies the
    half-space skin-effect model without a dc floor.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import MicrostripLine
        from pmrf.materials import BulkConductor, ConstantDielectric

        phys_microstrip = MicrostripLine(
            w=4e-3,
            h=2.0e-3,
            dielectric=ConstantDielectric(ep_r=4.6, tand=0.025),
            conductor=BulkConductor(sigma=5.8e7),
            length=0.5
        )

        freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
        s_phys = phys_microstrip.s(freq)    

    Supply either ``substrate`` or its individual fields, not both.

    Parameters
    ----------
    w : Param, default=3e-3
        Width of the microstrip trace in meters.
    substrate : Substrate, optional
        Substrate carrying the trace.
    h : Param, default=1.6e-3
        Height of the dielectric substrate in meters. Loose form of
        ``substrate.h``.
    dielectric : AbstractDielectric, default=ConstantDielectric(ep_r=4.3)
        The substrate material. A scalar permittivity or an ``(ep_r, tand)`` tuple
        is coerced into a :class:`~pmrf.materials.ConstantDielectric`.
    conductor : AbstractConductor, default=BulkConductor()
        The material of the trace and ground plane. A scalar conductivity in
        S/m is coerced into a :class:`~pmrf.materials.BulkConductor`.
    t : Param | None, default=None
        Conductor thickness. A positive value supplies a dc resistance floor
        and may refine the quasi-static geometry. ``None`` uses the half-space
        conductor model without a dc floor.
    formulation : AbstractMicrostripFormulation, default=HammerstadJensenMicrostripFormulation()
        The closed-form physics used to compute the quasi-static solution.
    dispersion : AbstractMicrostripDispersion | None, default=KirschningJansenMicrostripDispersion()
        The modal-dispersion correction. ``None`` disables modal dispersion and
        preserves the quasi-static immittance path.

    References
    ----------
    Wheeler, H. A. (1942). Formulas for the Skin Effect. Proceedings of the
    IRE, 30(9), 412-424.

    Schneider, M. V. (1969). Dielectric Loss in Integrated Microwave Circuits.
    Bell System Technical Journal, 48(7).

    Schneider, M. V. (1969). Microstrip Lines for Microwave Integrated Circuits.
    Bell System Technical Journal, 48(5), 1421-1444.

    Kirschning, M., & Jansen, R. H. (1982). Accurate Model for Effective
    Dielectric Constant of Microstrip with Validity up to Millimeter-Wave
    Frequencies. Electronics Letters, 18(6), 272-273.

    Jansen, R. H., & Kirschning, M. (1983). Arguments and an Accurate Model for
    the Power-Current Formulation of Microstrip Characteristic Impedance.
    Archiv fuer Elektronik und Uebertragungstechnik, 37, 108-112.
    """
    #: Width of the microstrip trace
    w: Param = param(default=3e-3, constraint=Positive())

    #: The substrate carrying the trace
    substrate: Substrate = field(default_factory=Substrate, converter=as_substrate)

    #: The underlying physics formulation
    formulation: AbstractMicrostripFormulation = field(
        default_factory=HammerstadJensenMicrostripFormulation
    )

    #: The modal-dispersion formulation, or None to disable it
    dispersion: AbstractMicrostripDispersion | None = field(
        default_factory=KirschningJansenMicrostripDispersion
    )

    #: The conductor current-distribution strategy
    current_distribution: AbstractCurrentDistribution = field(
        default_factory=WheelerCurrentDistribution
    )

    def __init__(
        self,
        w: Param = 3e-3,
        *,
        length: Param,
        substrate: Substrate | None = None,
        h: Param | None = None,
        dielectric=None,
        conductor=None,
        t: Param | None = None,
        formulation: AbstractMicrostripFormulation | None = None,
        dispersion: AbstractMicrostripDispersion | None = _DEFAULT_DISPERSION,
        current_distribution: AbstractCurrentDistribution | None = None,
        name: str | None = None,
        metadata=None,
    ):
        loose = {"h": h, "dielectric": dielectric, "conductor": conductor, "t": t}
        given = {key: value for key, value in loose.items() if value is not None}
        if substrate is not None and given:
            raise ValueError(
                "pass either substrate= or the loose substrate fields "
                f"({', '.join(sorted(given))}), not both"
            )

        self.w = w
        self.substrate = Substrate(**given) if substrate is None else substrate
        self.formulation = (
            HammerstadJensenMicrostripFormulation() if formulation is None else formulation
        )
        self.dispersion = (
            KirschningJansenMicrostripDispersion()
            if dispersion is _DEFAULT_DISPERSION else dispersion
        )
        self.current_distribution = (
            WheelerCurrentDistribution()
            if current_distribution is None else current_distribution
        )
        self.length = length
        self.name = name
        self.metadata = metadata

    def _resolved_quasi_static(self, freq: Frequency) -> PlanarQuasiStaticResult:
        """Return the quasi-static state after optional modal dispersion."""
        substrate = self.substrate
        dielectric = substrate.dielectric.properties(freq)
        ep_r = eqx.error_if(
            dielectric.ep_r,
            jnp.any(jnp.abs(dielectric.mu_r - 1) > 1e-12),
            "microstrip formulations require a nonmagnetic substrate (mu_r = 1)",
        )

        quasi_static = self.formulation.quasi_static(
            w=self.w,
            h=substrate.h,
            t=substrate.t,
            ep_r=ep_r,
        )

        if self.dispersion is None:
            return quasi_static

        ep_eff, zc = self.dispersion.disperse(
            freq,
            ep_eff_0=quasi_static.ep_eff,
            zc_0=quasi_static.zc,
            ep_r=ep_r,
            w_eff=quasi_static.w_eff,
            h=substrate.h,
        )

        # Route through the same to_immittance a quasi-static line uses, at
        # the dispersed (ep_eff, zc) rather than inverting the modal (zc,
        # gamma) through from_zc_gamma. That inversion made Zc = sqrt(Z/Y)
        # tautologically equal to the modal zc, so the conductor never
        # genuinely entered it; to_immittance instead builds Z and Y from
        # physical per-unit-length quantities, so Zc = sqrt(Z/Y) falls out
        # exact with the conductor contribution actually in it. This is a
        # ParamRF convention, not a correction of Kirschning-Jansen: K-J's Zc
        # is a power-current quasi-TEM modal quantity, chosen by Jansen &
        # Koster for its weak frequency dependence, and NIST (Williams,
        # Alpert, Arz et al., "Causal Characteristic Impedance of Planar
        # Transmission Lines") establishes that microstrip has no unique Zc.
        #
        return PlanarQuasiStaticResult(
            ep_eff=ep_eff,
            zc=zc,
            w_eff=quasi_static.w_eff,
            shunt_conductance_factor=quasi_static.shunt_conductance_factor,
        )

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        substrate = self.substrate
        conductor = substrate.conductor.properties(freq)
        dielectric = substrate.dielectric.properties(freq)
        cross_section = MicrostripCrossSection(w=self.w, h=substrate.h, t=substrate.t)
        return self._resolved_quasi_static(freq).to_immittance(
            freq, dielectric, conductor,
            current_distribution=self.current_distribution,
            cross_section=cross_section,
        )

    def ep_eff(self, freq: Frequency) -> jnp.ndarray:
        r"""
        Return the effective relative permittivity used by :meth:`immittance`.

        This includes modal dispersion when ``dispersion`` is set. Its
        imaginary part carries dielectric loss.

        Parameters
        ----------
        freq : Frequency
            Frequencies at which to evaluate the line.

        Returns
        -------
        jnp.ndarray
            Complex effective relative permittivity, shape ``(npoints,)``.
        """
        return self._resolved_quasi_static(freq).ep_eff

    def w_eff(self, freq: Frequency) -> jnp.ndarray:
        r"""
        Return the effective conductor width used by :meth:`immittance`.

        Parameters
        ----------
        freq : Frequency
            Frequencies at which to evaluate the line.

        Returns
        -------
        jnp.ndarray
            Effective conductor width in meters, shape ``(npoints,)``.
        """
        return self._resolved_quasi_static(freq).w_eff
