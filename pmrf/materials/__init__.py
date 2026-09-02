"""
Frequency-dependent materials.

Materials are :class:`pmrf.Module` sub-modules.
Dielectrics refer to the shunt terms (permittivity, loss tangent, static conductivity),
while conductors refer to the series terms (surface impedance).
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
    BulkConductor as BulkConductor,
    RoughConductor as RoughConductor,
    as_conductor as as_conductor,
)
from pmrf.materials.roughness import (
    AbstractRoughness as AbstractRoughness,
    HammerstadRoughness as HammerstadRoughness,
)
from pmrf.materials.surface_impedance import (
    AbstractSurfaceImpedance as AbstractSurfaceImpedance,
    HalfSpaceShape as HalfSpaceShape,
    HollowayKuesterSlabShape as HollowayKuesterSlabShape,
    RootSumSquareSlabShape as RootSumSquareSlabShape,
    SchelkunoffRodShape as SchelkunoffRodShape,
    SchelkunoffTubeShape as SchelkunoffTubeShape,
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
    "AbstractSurfaceImpedance",
    "HalfSpaceShape",
    "HollowayKuesterSlabShape",
    "RootSumSquareSlabShape",
    "SchelkunoffRodShape",
    "SchelkunoffTubeShape",
    "TescheRodShape",
    "TescheTubeShape",
    "Substrate",
    "as_substrate",
]
