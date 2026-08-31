"""
Frequency-dependent materials.

Materials are :class:`pmrf.Module` sub-modules rather than string flags, so
every dispersion coefficient is a :class:`pmrf.Param`: fittable, constrainable
and sweepable. A dielectric owns the shunt terms (permittivity, loss tangent,
static conductivity); a conductor owns the series terms (surface impedance).
"""
from pmrf.materials.dielectric import (
    AbstractDielectric as AbstractDielectric,
    ConstantDielectric as ConstantDielectric,
    DjordjevicSarkar as DjordjevicSarkar,
    DebyePole as DebyePole,
    MultipoleDebye as MultipoleDebye,
    ColeCole as ColeCole,
    TabulatedDielectric as TabulatedDielectric,
    as_dielectric as as_dielectric,
)
from pmrf.materials.conductor import (
    AbstractConductor as AbstractConductor,
    Bulk as Bulk,
    RoughConductor as RoughConductor,
    as_conductor as as_conductor,
)

__all__ = [
    "AbstractDielectric",
    "ConstantDielectric",
    "DjordjevicSarkar",
    "DebyePole",
    "MultipoleDebye",
    "ColeCole",
    "TabulatedDielectric",
    "as_dielectric",
    "AbstractConductor",
    "Bulk",
    "RoughConductor",
    "as_conductor",
]
