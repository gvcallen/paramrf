"""
Frequency-dependent materials.

Materials are :class:`pmrf.Module` sub-modules.
Dielectrics refer to the shunt terms (permittivity, loss tangent, static conductivity),
while conductors refer to the series terms (surface impedance).
"""
from pmrf.materials.dielectric import (
    AbstractDielectric as AbstractDielectric,
    ConstantDielectric as ConstantDielectric,
    DjordjevicSarkarDielectric as DjordjevicSarkarDielectric,
    DebyePole as DebyePole,
    MultipoleDebyeDielectric as MultipoleDebyeDielectric,
    ColeColeDielectric as ColeColeDielectric,
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
    HalfSpaceSurfaceImpedance as HalfSpaceSurfaceImpedance,
    HollowayKuesterSlabSurfaceImpedance as HollowayKuesterSlabSurfaceImpedance,
    RootSumSquareSlabSurfaceImpedance as RootSumSquareSlabSurfaceImpedance,
    SchelkunoffRodSurfaceImpedance as SchelkunoffRodSurfaceImpedance,
    SchelkunoffTubeSurfaceImpedance as SchelkunoffTubeSurfaceImpedance,
    TescheRodSurfaceImpedance as TescheRodSurfaceImpedance,
    TescheTubeSurfaceImpedance as TescheTubeSurfaceImpedance,
)

__all__ = [
    "AbstractDielectric",
    "ConstantDielectric",
    "DjordjevicSarkarDielectric",
    "DebyePole",
    "MultipoleDebyeDielectric",
    "ColeColeDielectric",
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
    "HalfSpaceSurfaceImpedance",
    "HollowayKuesterSlabSurfaceImpedance",
    "RootSumSquareSlabSurfaceImpedance",
    "SchelkunoffRodSurfaceImpedance",
    "SchelkunoffTubeSurfaceImpedance",
    "TescheRodSurfaceImpedance",
    "TescheTubeSurfaceImpedance",
    "Substrate",
    "as_substrate",
]
