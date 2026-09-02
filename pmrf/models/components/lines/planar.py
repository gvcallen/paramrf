"""Shared machinery for planar transmission lines."""
from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import c

from pmrf.frequency import Frequency
from pmrf.materials import ConductorProperties, DielectricProperties
from pmrf.materials.surface_impedance import AbstractSurfaceImpedance
from pmrf.models.components.lines.base import ImmittanceResult

class AbstractPlanarCrossSection(eqx.Module):
    """Cross-section geometry for a planar line family."""

    def dimensions(self) -> dict:
        """Return dimensions accepted by conductor-shape formulations."""
        raise NotImplementedError


class PlanarQuasiStaticResult(eqx.Module):
    r"""
    Quasi-static solution of a single-conductor planar line over a ground plane.

    Contains the effective permittivity, characteristic impedance, effective
    conductor width, and static-conductivity geometry factor.

    Parameters
    ----------
    ep_eff : jnp.ndarray
        Complex effective relative permittivity, shape ``(npoints,)``.
    zc : jnp.ndarray
        Quasi-static characteristic impedance in ohms, $Z_a/\sqrt{\varepsilon_e}$.
    w_eff : jnp.ndarray
        Electromagnetic effective conductor width in meters.
    shunt_conductance_factor : jnp.ndarray
        Geometry factor multiplying static conductivity, in meters.
    """
    #: Complex effective relative permittivity
    ep_eff: jnp.ndarray

    #: Quasi-static characteristic impedance in ohms
    zc: jnp.ndarray

    #: Effective conductor width in meters
    w_eff: jnp.ndarray

    #: Geometry factor multiplying static conductivity
    shunt_conductance_factor: jnp.ndarray

    def to_immittance(
        self, freq: Frequency, dielectric: DielectricProperties,
        conductor: ConductorProperties,
        current_distribution: AbstractCurrentDistribution,
        cross_section: AbstractPlanarCrossSection,
    ) -> ImmittanceResult:
        r"""
        Convert the quasi-static solution to per-unit-length immittance.

        The external inductance and the shunt admittance follow from the
        quasi-static impedance and effective permittivity. Surface impedance
        is charged through the supplied current-distribution strategy:
        $$Z = \frac{j\omega Z_c \sqrt{\varepsilon_e}}{c} + Z_s K_c
        \qquad
        Y = \frac{j\omega \sqrt{\varepsilon_e}}{Z_c c}$$

        The current distribution supplies surface impedances and geometry
        weights. Complex $\varepsilon_e$ contributes dielectric loss through
        the real part of $Y$.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        dielectric : DielectricProperties
            Evaluated relative permittivity and permeability.
        conductor : ConductorProperties
            Evaluated conductor properties.
        current_distribution : AbstractCurrentDistribution
            The strategy that charges surface impedance into $Z$. It must be
            written for the family ``cross_section`` belongs to.
        cross_section : AbstractPlanarCrossSection
            The line's frozen cross-section record.

        Returns
        -------
        ImmittanceResult
            The series impedance and shunt admittance.

        References
        ----------
        Pozar, D. M. (2011). Microwave Engineering (4th ed.), Section 3.8. Wiley.
        """
        omega = freq.w
        sqrt_ep_eff_mu = jnp.sqrt(self.ep_eff * dielectric.mu_r)
        sqrt_ep_eff_over_mu = jnp.sqrt(self.ep_eff / dielectric.mu_r)

        Z = 1j * omega * self.zc * sqrt_ep_eff_mu / c
        # Cross-section dimensions reach the shape from the typed record, and
        # the weight the shape is about to be multiplied by travels with
        # them: an entry whose dc floor is fixed in per-unit-length terms
        # needs it to express that floor in this caller's normalisation.
        # Every other entry ignores it.
        dimensions = cross_section.dimensions()
        Z_cond = sum(
            shape.impedance(omega, conductor, weight=weight, **dimensions) * weight
            for shape, weight in current_distribution.distribute(
                freq, cross_section, self
            )
        )
        Z = Z + Z_cond
        Y = 1j * omega * sqrt_ep_eff_over_mu / (self.zc * c)
        Y = Y + dielectric.sigma * self.shunt_conductance_factor

        return ImmittanceResult(Z=Z, Y=Y, omega=omega)


CrossSectionT = TypeVar("CrossSectionT", bound=AbstractPlanarCrossSection)

class AbstractCurrentDistribution(eqx.Module, Generic[CrossSectionT]):
    r"""Surface-current distribution for a transmission-line cross-section.

    Returns ``(shape, weight)`` pairs used to calculate conductor series
    impedance. Each shape supplies a surface impedance; its weight converts
    that value to impedance per unit length. Weights may depend on frequency.

    :attr:`cross_section_type` identifies the supported line family.
    :meth:`distribute` validates the supplied cross-section before evaluating
    the distribution.
    """

    #: The cross-section record family this strategy is written for.
    cross_section_type: ClassVar[type] = AbstractPlanarCrossSection

    def distribute(
        self,
        freq: Frequency,
        cross_section: CrossSectionT,
        quasi_static: "PlanarQuasiStaticResult",
    ):
        r"""Return the conductor-shape and geometry-weight pairs.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.
        cross_section : AbstractPlanarCrossSection
            Cross-section record. It must be an instance of
            :attr:`cross_section_type`.
        quasi_static : PlanarQuasiStaticResult
            Solved quasi-static line properties.

        Returns
        -------
        tuple
            ``(shape, weight)`` pairs.

        Raises
        ------
        TypeError
            If the cross-section record belongs to another line family.
        """
        if not isinstance(cross_section, self.cross_section_type):
            raise TypeError(
                f"{type(self).__name__} is a "
                f"{self.cross_section_type.__name__} strategy, but it was "
                f"given a {type(cross_section).__name__}"
            )
        return self._distribute(freq, cross_section, quasi_static)

    @abstractmethod
    def _distribute(
        self,
        freq: Frequency,
        cross_section: CrossSectionT,
        quasi_static: "PlanarQuasiStaticResult",
    ):
        r"""Return the pairs for an already type-checked cross-section."""
        raise NotImplementedError
