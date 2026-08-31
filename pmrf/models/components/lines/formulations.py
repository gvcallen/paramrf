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
    Quasi-static solution of a single-conductor planar line over a ground plane.

    A quasi-static formulation stops at the effective permittivity, the
    characteristic impedance it implies and the effective conductor width; the
    line turns those into an immittance, and a dispersion model may correct them
    first.

    Coupled lines are not this type: they carry a solution per mode, so they
    need their own result.

    Parameters
    ----------
    ep_eff : jnp.ndarray
        Complex effective relative permittivity, shape ``(npoints,)``.
    zc : jnp.ndarray
        Quasi-static characteristic impedance in ohms, $Z_a/\sqrt{\varepsilon_e}$.
    w_eff : jnp.ndarray
        Effective conductor width in meters, carrying the series loss.
    """
    #: Complex effective relative permittivity
    ep_eff: jnp.ndarray

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
        sqrt_ep_eff = jnp.sqrt(self.ep_eff)

        Z = 1j * w * self.zc * sqrt_ep_eff / c + 2 * zs / self.w_eff
        Y = 1j * w * sqrt_ep_eff / (self.zc * c)

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
    def immittance(self, freq: Frequency, *, d_in, d_out, ep_r, mu_r, conductor: AbstractConductor) -> ImmittanceResult:
        r"""
        Calculates the per-unit-length immittance of the line.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        d_in : ArrayLike
            Inner conductor diameter in meters.
        d_out : ArrayLike
            Outer conductor inner diameter in meters.
        ep_r : jnp.ndarray
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
    def immittance(self, freq: Frequency, *, d_in, d_out, ep_r, mu_r, conductor: AbstractConductor) -> ImmittanceResult:
        eps = epsilon_0 * ep_r
        mu = mu_0 * mu_r
        w = freq.w

        a, b = d_in / 2, d_out / 2
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
    def quasi_static(self, freq: Frequency, *, w, h, t, ep_r, zs) -> QuasiStaticResult:
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
        ep_r : jnp.ndarray
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
    def quasi_static(self, freq: Frequency, *, w, h, t, ep_r, zs) -> QuasiStaticResult:
        if t is not None:
            raise ValueError("Wheeler microstrip approximation does not support finite thickness")

        W, H = w, h
        u = W / H

        # Shared base terms
        t1 = (ep_r + 1) / 2
        t2 = (ep_r - 1) / 2
        t3 = 1 / jnp.sqrt(1 + 12 / u)

        # Piecewise effective permittivity (ep_eff)
        ep_eff_le1 = t1 + t2 * (t3 + 0.04 * (1 - u)**2)
        ep_eff_gt1 = t1 + t2 * t3
        ep_eff = jnp.where(u <= 1.0, ep_eff_le1, ep_eff_gt1) * jnp.ones(freq.npoints)

        # Piecewise characteristic impedance in air (Za)
        Za_le1 = 60 * jnp.log(8 / u + 0.25 * u)
        Za_gt1 = (120 * jnp.pi) / (u + 1.393 + 0.667 * jnp.log(u + 1.444))
        Za = jnp.where(u <= 1.0, Za_le1, Za_gt1)

        zc = Za / jnp.sqrt(ep_eff)
        w_eff = W * jnp.ones(freq.npoints)

        return QuasiStaticResult(ep_eff=ep_eff, zc=zc, w_eff=w_eff)


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
    where $e$ denotes the bracketed permittivity expression before its
    thickness correction.

    The fractional power in $b$ is evaluated through the principal logarithm.
    The dielectric constraint $\Re(\varepsilon_r)>1$ keeps its argument away
    from the negative-real branch cut for passive materials.

    References
    ----------
    Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
    Computer-Aided Design. IEEE MTT-S International Microwave Symposium Digest,
    407-409.
    """

    def quasi_static(self, freq: Frequency, *, w, h, t, ep_r, zs) -> QuasiStaticResult:
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
        return QuasiStaticResult(ep_eff=ep_eff, zc=zc, w_eff=w_eff)

    @staticmethod
    def _homogeneous_impedance(u):
        z0 = jnp.sqrt(mu_0 / epsilon_0)
        f_u = 6 + (2 * jnp.pi - 6) * jnp.exp(-(30.666 / u) ** 0.7528)
        return z0 / (2 * jnp.pi) * jnp.log(f_u / u + jnp.sqrt(1 + (2 / u) ** 2))


class AbstractMicrostripDispersion(eqx.Module):
    """Abstract modal-dispersion formulation for an inhomogeneous microstrip."""

    @abstractmethod
    def disperse(
        self, freq: Frequency, *, ep_eff_0, zc_0, ep_r, w, w_eff, h, t
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""Return frequency-dependent $(\varepsilon_e, Z_c)$."""
        raise NotImplementedError


class KirschningJansen(AbstractMicrostripDispersion):
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
    Following the ADS and QUCS conventions, the normalized width is
    $$u=\begin{cases}W_{eff}/H,&\text{complex permittivity},\\
    W/H,&\text{real permittivity}.\end{cases}$$

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
    This extension is not part of the cited empirical fit; it supplies a
    continuous, differentiable connection to its required homogeneous limit.

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
        self, freq: Frequency, *, ep_eff_0, zc_0, ep_r, w, w_eff, h, t
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # ADS applies the effective width in its complex-permittivity path;
        # QUCS applies the physical width in its real-permittivity path.
        dispersion_width = w_eff if jnp.iscomplexobj(ep_r) else w
        u = dispersion_width / h
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
