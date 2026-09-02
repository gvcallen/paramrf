"""Stripline models, formulations, and current distributions."""
from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import c, epsilon_0

from pmrf.constraints import Positive
from pmrf.frequency import Frequency
from pmrf.materials import AbstractConductor, AbstractDielectric, BulkConductor, ConstantDielectric, as_conductor, as_dielectric
from pmrf.materials.surface_impedance import HalfSpaceShape
from pmrf.models.components.lines.base import AbstractImmittanceLine, ImmittanceResult
from pmrf.models.components.lines.planar import AbstractCurrentDistribution, AbstractPlanarCrossSection, PlanarQuasiStaticResult
from pmrf.parameters import Param, as_param, param
from pmrf.utils import field

class StriplineCrossSection(AbstractPlanarCrossSection):
    """Cross-section of a centre strip between two ground planes.

    Parameters
    ----------
    w : ArrayLike
        Width of the centre strip in meters.
    b : ArrayLike
        Separation of the ground planes in meters.
    t : ArrayLike | None, default=None
        Strip thickness in meters, or ``None`` when it is unspecified.
    ep_r : jnp.ndarray | None, default=None
        Complex relative permittivity of the homogeneous filling, used by
        Cohn's attenuation model.
    """

    #: Width of the centre strip in meters
    w: jnp.ndarray

    #: Separation of the ground planes in meters
    b: jnp.ndarray

    #: Strip thickness in meters, or ``None`` when unspecified
    t: jnp.ndarray | None = None

    #: Complex relative permittivity of the filling
    ep_r: jnp.ndarray | None = None

    def dimensions(self) -> dict:
        return {"w": self.w, "t": self.t}


class CohnCurrentDistribution(AbstractCurrentDistribution[StriplineCrossSection]):
    r"""Cohn's stripline current distribution.

    **Mathematical Formulation**

    Cohn gives conductor attenuation per unit length. Inverting
    $$\alpha_c=\frac{\Re(Z_s k_c)}{2\Re(Z_c)}$$
    gives the geometry weight
    $$k_c=2(\alpha_c/R_s)\Re(Z_c).$$
    The model assigns zero conductor-loss weight when the strip thickness is
    unspecified.

    References
    ----------
    Cohn, S. B. (1955). Problems in Strip Transmission Lines. IRE Transactions
    on Microwave Theory and Techniques, 3(2), 119-126.
    """

    cross_section_type: ClassVar[type] = StriplineCrossSection

    def _distribute(self, freq, cross_section, quasi_static):
        w, b, t = cross_section.w, cross_section.b, cross_section.t
        if t is None:
            weight = jnp.asarray(0.0)
        else:
            ep_r = jnp.real(cross_section.ep_r)
            zc_real = jnp.real(quasi_static.zc)
            a = 1 + 2 * w / (b - t) + (b + t) / (jnp.pi * (b - t)) * jnp.log((2 * b - t) / t)
            alpha_low = 2.7e-3 * ep_r * zc_real / (30 * jnp.pi * (b - t)) * a
            beta = 1 + b / (0.5 * w + 0.7 * t) * (
                0.5 + 0.7 * t / w + jnp.log(4 * jnp.pi * w / t) / (2 * jnp.pi)
            )
            alpha_high = 0.16 / (zc_real * b) * beta
            alpha_over_rs = jnp.where(jnp.sqrt(ep_r) * zc_real < 120, alpha_low, alpha_high)
            weight = 2 * alpha_over_rs * zc_real
        return ((HalfSpaceShape(), weight),)


class AbstractStriplineFormulation(eqx.Module):
    """Abstract base class for a closed-form stripline formulation.

    Homogeneous filling gives $\varepsilon_e=\varepsilon_r$ without modal
    dispersion.
    """

    @abstractmethod
    def quasi_static(self, *, w, b, t, ep_r) -> PlanarQuasiStaticResult:
        r"""
        Calculate the quasi-static solution.

        Parameters
        ----------
        w : ArrayLike
            Width of the centre strip in meters.
        b : ArrayLike
            Ground-plane separation in meters.
        t : ArrayLike | None
            Thickness of the strip in meters, or None for a zero-thickness strip.
        ep_r : jnp.ndarray
            Complex relative permittivity of the filling, shape ``(npoints,)``.

        Returns
        -------
        PlanarQuasiStaticResult
            The effective permittivity, impedance and effective width.
        """
        raise NotImplementedError


class CohnStriplineFormulation(AbstractStriplineFormulation):
    r"""
    Cohn's stripline formulation, in the form tabulated by Pozar.

    **Mathematical Formulation**

    The filling is homogeneous, so
    $$\varepsilon_e = \varepsilon_r$$
    exactly, with no filling factor and no modal dispersion. With the fringing
    correction to the strip width,
    $$\frac{W_e}{b} = \frac{W}{b} -
    \begin{cases}0, & W/b > 0.35,\\ (0.35 - W/b)^2, & W/b \leq 0.35,\end{cases}$$
    the characteristic impedance of the zero-thickness strip is
    $$Z_c = \frac{30\pi}{\sqrt{\varepsilon_r}}\frac{b}{W_e + 0.441b}.$$

    Conductor loss is supplied by
    :class:`CohnCurrentDistribution`,
    and complex $\varepsilon_e$ carries dielectric loss.

    **Validity**

    The impedance expression assumes zero thickness and uses a continuous
    fringing correction at $W/b=0.35$. A supplied thickness must satisfy
    $0<T<b$, although it does not enter the impedance expression.

    References
    ----------
    Cohn, S. B. (1955). Problems in Strip Transmission Lines. IRE Transactions
    on Microwave Theory and Techniques, 3(2), 119-126.

    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.7. Wiley.
    """

    def quasi_static(self, *, w, b, t, ep_r) -> PlanarQuasiStaticResult:
        ones = jnp.ones_like(ep_r)
        if t is not None:
            t = eqx.error_if(t, t - b >= 0, "stripline thickness must satisfy 0 < t < b")
        ep_eff = ep_r * ones

        u = w / b
        w_e = b * (u - jnp.where(u > 0.35, 0.0, (0.35 - u) ** 2))
        zc = 30 * jnp.pi / jnp.sqrt(ep_eff) * b / (w_e + 0.441 * b)

        shunt_conductance_factor = jnp.sqrt(ep_eff) / (zc * c * epsilon_0 * ep_eff)
        return PlanarQuasiStaticResult(
            ep_eff, zc, w_e * ones, shunt_conductance_factor,
        )


class StriplineLine(AbstractImmittanceLine):
    r"""
    Stripline defined by its geometry and material modules.

    The default is :class:`CohnStriplineFormulation`. Homogeneous filling gives
    $\varepsilon_e=\varepsilon_r$ without a separate modal-dispersion model.
    Material dispersion remains available through the dielectric.

    **Mathematical Formulation**

    The quasi-static formulation returns $(\varepsilon_e, Z_c, W_{eff})$, and
    :meth:`PlanarQuasiStaticResult.to_immittance` converts them directly:
    $$Z = \frac{j\omega Z_c\sqrt{\varepsilon_e}}{c} + \frac{2Z_s}{W_{eff}}
    \qquad
    Y = \frac{j\omega\sqrt{\varepsilon_e}}{Z_c c}.$$
    See :class:`CohnStriplineFormulation` for the geometry.

    Example
    --------
    .. code-block:: python

        import pmrf as prf
        from pmrf.models import StriplineLine
        from pmrf.materials import BulkConductor, ConstantDielectric

        line = StriplineLine(
            w=2.655e-3,
            b=3.2e-3,
            t=35e-6,
            dielectric=ConstantDielectric(ep_r=2.2, tand=0.001),
            conductor=BulkConductor(sigma=5.8e7),
            length=0.1,
        )

        freq = prf.Frequency(start=1, stop=20, npoints=101, unit='ghz')
        s = line.s(freq)

    Parameters
    ----------
    w : Param, default=2.655e-3
        Width of the centre strip in meters.
    b : Param, default=3.2e-3
        Separation of the ground planes in meters.
    t : Param | None, default=35e-6
        Thickness of the centre strip in meters. ``None`` idealises it as
        zero-thickness, which has no finite conductor loss.
    dielectric : AbstractDielectric, default=ConstantDielectric(ep_r=4.3)
        The filling between the ground planes. A scalar permittivity or an
        ``(ep_r, tand)`` tuple is coerced into a
        :class:`~pmrf.materials.ConstantDielectric`.
    conductor : AbstractConductor, default=BulkConductor()
        The material of the strip and the ground planes. A scalar conductivity in
        S/m is coerced into a :class:`~pmrf.materials.BulkConductor`.
    formulation : AbstractStriplineFormulation, default=CohnStriplineFormulation()
        The closed-form physics used to compute the quasi-static solution.

    References
    ----------
    Cohn, S. B. (1955). Problems in Strip Transmission Lines. IRE Transactions
    on Microwave Theory and Techniques, 3(2), 119-126.

    Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.7. Wiley.
    """
    #: Width of the centre strip
    w: Param = param(default=2.655e-3, constraint=Positive())

    #: Separation of the ground planes
    b: Param = param(default=3.2e-3, constraint=Positive())

    #: Thickness of the centre strip
    t: Param | None = field(
        default=35e-6,
        converter=lambda x: as_param(x, constraint=Positive()) if x is not None else None,
    )

    #: The filling between the ground planes
    dielectric: AbstractDielectric = field(
        default_factory=lambda: ConstantDielectric(ep_r=4.3), converter=as_dielectric
    )

    #: The material of the strip and the ground planes
    conductor: AbstractConductor = field(
        default_factory=BulkConductor, converter=as_conductor
    )

    #: The underlying physics formulation
    formulation: AbstractStriplineFormulation = field(
        default_factory=CohnStriplineFormulation
    )

    #: The conductor current-distribution strategy
    current_distribution: AbstractCurrentDistribution = field(
        default_factory=CohnCurrentDistribution
    )

    def immittance(self, freq: Frequency) -> ImmittanceResult:
        dielectric = self.dielectric.properties(freq)
        conductor = self.conductor.properties(freq)
        quasi_static = self.formulation.quasi_static(
            w=self.w,
            b=self.b,
            t=self.t,
            ep_r=dielectric.ep_r,
        )
        return quasi_static.to_immittance(
            freq, dielectric, conductor,
            current_distribution=self.current_distribution,
            cross_section=StriplineCrossSection(
                w=self.w, b=self.b, t=self.t, ep_r=dielectric.ep_r
            ),
        )
