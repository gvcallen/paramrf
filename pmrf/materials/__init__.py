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
from pmrf.materials.properties import (
    ConductorProperties as ConductorProperties,
    DielectricProperties as DielectricProperties,
)
from pmrf.materials.substrate import (
    Substrate as Substrate,
    as_substrate as as_substrate,
)
from pmrf.materials.conductor import (
    AbstractConductor as AbstractConductor,
    AbstractRoughness as AbstractRoughness,
    BulkConductor as BulkConductor,
    HammerstadRoughness as HammerstadRoughness,
    RoughConductor as RoughConductor,
    as_conductor as as_conductor,
)
from pmrf.materials.conductor_shape import (
    AbstractConductorShape as AbstractConductorShape,
    HalfSpaceShape as HalfSpaceShape,
    SchelkunoffRodShape as SchelkunoffRodShape,
    SchelkunoffTubeShape as SchelkunoffTubeShape,
    SchelkunoffCothTubeShape as SchelkunoffCothTubeShape,
    SchelkunoffInfiniteTubeShape as SchelkunoffInfiniteTubeShape,
    TescheRodShape as TescheRodShape,
    TescheTubeShape as TescheTubeShape,
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
    "DielectricProperties",
    "ConductorProperties",
    "AbstractConductor",
    "AbstractRoughness",
    "BulkConductor",
    "HammerstadRoughness",
    "RoughConductor",
    "as_conductor",
    "AbstractConductorShape",
    "HalfSpaceShape",
    "SchelkunoffRodShape",
    "SchelkunoffTubeShape",
    "SchelkunoffCothTubeShape",
    "SchelkunoffInfiniteTubeShape",
    "TescheRodShape",
    "TescheTubeShape",
    "Substrate",
    "as_substrate",
]
