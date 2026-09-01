"""
A convenience grouping for a dielectric sheet of known height, with its metallization.
"""
from __future__ import annotations

from pmrf.constraints import Positive
from pmrf.materials.conductor import AbstractConductor, BulkConductor, as_conductor
from pmrf.materials.dielectric import AbstractDielectric, ConstantDielectric, as_dielectric
from pmrf.modules.base import Module
from pmrf.parameters import Param, as_param, param
from pmrf.utils import field


class Substrate(Module):
    r"""
    A dielectric sheet of a given height, with the conductor printed on it.

    Substrates can be used as a shared base that is injected into other models.
    For example, when combined with :class:`~pmrf.models.AbstractBuilder`,
    one substrate can be shared between two lines within the same model.

    >>> import pmrf as prf
    >>> from pmrf.materials import Substrate
    >>> from pmrf.models import AbstractBuilder, MicrostripLine
    >>> class Board(AbstractBuilder):
    ...     substrate: Substrate
    ...     w1: prf.Param
    ...     w2: prf.Param
    ...     def build(self):
    ...         return (MicrostripLine(w=self.w1, substrate=self.substrate, length=0.1)
    ...              ** MicrostripLine(w=self.w2, substrate=self.substrate, length=0.2))
    >>> board = Board(substrate=Substrate(h=1.6e-3, dielectric=4.3), w1=1e-3, w2=2e-3)
    >>> [name for name in board.named_params() if name.endswith("ep_r")]
    ['substrate.dielectric.ep_r']

    Parameters
    ----------
    h : Param, default=1.6e-3
        Height of the dielectric sheet in meters.
    dielectric : AbstractDielectric, default=ConstantDielectric(ep_r=4.3)
        The sheet material. A scalar permittivity or an ``(ep_r, tand)`` tuple is
        coerced into a :class:`~pmrf.materials.ConstantDielectric`.
    conductor : AbstractConductor, default=BulkConductor()
        The metallization. A scalar conductivity in S/m is coerced into a
        :class:`~pmrf.materials.BulkConductor`.
    t : Param | None, default=None
        Thickness of the metallization in meters, or ``None`` when it is
        unspecified. Unspecified is not the same input as zero: it asserts
        skin effect in operation at every frequency, so the conductor loss
        gets no dc resistance floor. A positive value gets the floor
        $R_{dc}=1/(\sigma W t)$, and refines the geometry in those
        quasi-static formulations that are thickness-aware.
    """
    #: Height of the dielectric sheet
    h: Param = param(default=1.6e-3, constraint=Positive())

    #: The sheet material
    dielectric: AbstractDielectric = field(
        default_factory=lambda: ConstantDielectric(ep_r=4.3), converter=as_dielectric
    )

    #: The metallization
    conductor: AbstractConductor = field(
        default_factory=BulkConductor, converter=as_conductor
    )

    #: Thickness of the metallization
    t: Param | None = field(
        default=None,
        converter=lambda x: as_param(x, constraint=Positive()) if x is not None else None,
    )


def as_substrate(value) -> Substrate:
    """
    Coerce a value into a :class:`Substrate`.

    Accepts an existing :class:`Substrate` or a mapping of its fields.

    Parameters
    ----------
    value : Any
        The value to coerce.

    Returns
    -------
    Substrate
        The resulting substrate.
    """
    if isinstance(value, Substrate):
        return value
    if isinstance(value, dict):
        return Substrate(**value)
    raise TypeError(f"cannot interpret {value!r} as a Substrate")
