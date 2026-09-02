"""Current-distribution strategies for conductor loss in planar lines."""
from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from scipy.constants import epsilon_0, mu_0

from pmrf.frequency import Frequency
from pmrf.materials.conductor_shape import (
    AbstractConductorShape,
    HalfSpaceShape,
    RootSumSquareSlabShape,
)
from pmrf.models.components.lines.cross_section import (
    AbstractPlanarCrossSection,
    MicrostripCrossSection,
    StriplineCrossSection,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from pmrf.models.components.lines.formulations import PlanarQuasiStaticResult

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


class WheelerCurrentDistribution(AbstractCurrentDistribution[MicrostripCrossSection]):
    r"""Wheeler's incremental-inductance current distribution.

    This implements Wheeler's 1942 skin-effect rule. It is distinct from the
    1977 quasi-static impedance approximation implemented by
    :class:`~pmrf.models.components.lines.formulations.WheelerMicrostripFormulation`.

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
