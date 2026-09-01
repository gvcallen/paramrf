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

    A strategy returns ``(shape, weight)`` pairs.  The shape supplies the
    material's surface impedance and the weight, in inverse metres, charges
    that impedance into the line's series impedance.  Weights are evaluated
    at the requested frequency because current crowding may be frequency
    dependent.  A strategy chooses shapes and weights only: the
    cross-section dimensions a shape needs reach it from the record at call
    time, so no strategy ever hands a shape a number derived from its own
    weight.

    A strategy is written for one planar family and declares it in
    :attr:`cross_section_type`.  It receives that family's frozen
    cross-section record together with the solved quasi-static state, so a
    model that consumes solved intermediates rather than dimensions needs no
    new record field.  :meth:`distribute` checks the record against the
    declared type, which is where a wrong-family pairing fails.
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
            The line's frozen cross-section record. It must be an instance of
            this strategy's :attr:`cross_section_type`.
        quasi_static : PlanarQuasiStaticResult
            The solved quasi-static state of the line, as the conversion to
            immittance has it.

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

    **Mathematical Formulation**

    $$k_c = \frac{2}{W}\exp\left[-1.2\left(\frac{\Re(Z_c)}{Z_0}\right)^{0.7}\right]$$

    The weight charges the sheet impedance over the physical trace width $W$,
    not a dispersion-widened one: the rule sums over every receded conductor
    surface, and those terms do not vanish as $t\to0$.  $Z_c$ is the
    characteristic impedance of the solved state it is given, so on a
    dispersive line it is the dispersed one; only its real part enters, since
    the exponent is a lossless current-crowding correction.

    The trace is represented by a half-space surface and the returned weight
    is Wheeler's geometry factor in inverse metres.

    A trace of unspecified thickness is a half-space; a trace of stated
    thickness is charged through :attr:`slab_shape`.

    References
    ----------
    Wheeler, H. A. (1942). Formulas for the Skin Effect. Proceedings of the
    IRE, 30(9), 412-424.
    """

    cross_section_type: ClassVar[type] = MicrostripCrossSection

    #: The finite-thickness cross-section entry used when a thickness is
    #: stated.  The default,
    #: :class:`~pmrf.materials.conductor_shape.RootSumSquareSlabShape`, is
    #: the only entry that is right at both asymptotes under this
    #: distribution's frequency-independent weight;
    #: :class:`~pmrf.materials.conductor_shape.HollowayKuesterSlabShape` is
    #: the exact strip-diffusion result but is normalised to the total strip
    #: current and so wants a weight of $1/(2W)$ rather than Wheeler's.
    #: Neither is the better entry in general -- see the normalisation note
    #: on :class:`~pmrf.materials.conductor_shape.AbstractConductorShape`.
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

    The returned weight is $k_c=2(\alpha_c/R_s)Z_c$: Cohn gives the
    attenuation per unit length, and the immittance the weight is charged
    into is a series impedance, so the low-loss inversion of
    $\alpha_c=\Re(Z_s k_c)/(2\Re(Z_c))$ puts the $Z_c$ back.  A zero-thickness
    strip has no finite conductor-loss weight in this model.

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
