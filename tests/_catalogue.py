"""
A hand-written instance of every concrete line, material and current distribution.

Each model needs a physically sensible geometry, so the instances are written by
hand. `concrete_subclasses` lets a coverage test fail until a new class is listed,
so every check parametrised over this catalogue reaches models added later.
"""
import jax.numpy as jnp

from pmrf.materials import (
    ColeColeDielectric,
    ConstantDielectric,
    DjordjevicSarkarDielectric,
    HammerstadRoughness,
    PoleDielectric,
    BulkConductor,
    DebyePole,
    LorentzPole,
    RoughConductor,
    TabulatedDielectric,
)
from pmrf.models import (
    CoaxialLine,
    DatasheetLine,
    DelayLine,
    FirstOrderEffectiveLineFormulation,
    FloatingLine,
    GeneralizedLine,
    LowLossLine,
    MicrostripLine,
    PhaseLine,
    PhysicalLine,
    RLGCLine,
    StriplineLine,
)
from pmrf.models.components.lines.microstrip import (
    HammerstadJensenMicrostripFormulation,
    IncrementalInductanceCurrentDistribution,
    KirschningJansenMicrostripDispersion,
    TraceGroundCurrentDistribution,
    WheelerMicrostripFormulation,
)


LINES = {
    "PhaseLine": PhaseLine(z0=50.0, theta=90.0, f0=5e9),
    "DelayLine": DelayLine(z0=50.0, td=1e-9),
    "RLGCLine": RLGCLine(R=0.1, L=250e-9, G=1e-6, C=100e-12, length=0.1),
    "RLGCLine (lossless)": RLGCLine(R=0.0, L=250e-9, G=0.0, C=100e-12, length=0.1),
    "LowLossLine": LowLossLine(zn=50.0, ep_eff=2.2, A=0.01, tand=0.001, length=1.0),
    "PhysicalLine": PhysicalLine(zn=50.0, ep_eff=2.2, A=0.01, tand=0.001, length=1.0),
    "PhysicalLine (first-order)": PhysicalLine(
        zn=50.0, ep_eff=2.2, A=0.01, tand=0.001, length=1.0,
        formulation=FirstOrderEffectiveLineFormulation(),
    ),
    "DatasheetLine": DatasheetLine(zn=50.0, vf=0.69, k1=0.2, k2=0.01, length=1.0),
    "DatasheetLine (first-order)": DatasheetLine(
        zn=50.0, vf=0.69, k1=0.2, k2=0.01, length=1.0,
        formulation=FirstOrderEffectiveLineFormulation(),
    ),
    # Every term family and the skin-effect onset. The undamped LC terms resonate above
    # the 0-10 GHz sweep; the damped ones may sit inside it.
    "GeneralizedLine": GeneralizedLine(
        zn=50.0, ep_eff=2.1, A=0.05, tand=2e-4, length=1.0, R_dc=0.01, f_s=1e6,
        conductor_terms=[(0.02, 1e6)],
        dielectric_terms=[(1e-3, 1e8)],
        inductive_dispersion=[(1e-2, 50e9), (5e-3, 2e9, 0.2)],
        capacitive_dispersion=[(1e-2, 60e9), (5e-3, 5e9, 0.1)],
    ),
    "GeneralizedLine (no onset)": GeneralizedLine(zn=50.0, ep_eff=2.1, A=0.05, tand=2e-4, length=1.0),
    "CoaxialLine": CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=ConstantDielectric(ep_r=1.5, tand=4e-4),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=0.5,
    ),
    "MicrostripLine": MicrostripLine(
        w=3e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=0.1,
    ),
    "MicrostripLine (no dispersion)": MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02), dispersion=None, length=0.1
    ),
    "MicrostripLine (explicit KirschningJansenMicrostripDispersion)": MicrostripLine(
        formulation=WheelerMicrostripFormulation(),
        dispersion=KirschningJansenMicrostripDispersion(),
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        length=0.1,
    ),
    "MicrostripLine (Wheeler, no dispersion)": MicrostripLine(
        formulation=WheelerMicrostripFormulation(),
        dispersion=None,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        length=0.1,
    ),
    "MicrostripLine (finite thickness)": MicrostripLine(
        formulation=HammerstadJensenMicrostripFormulation(),
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        t=35e-6,
        length=0.1,
    ),
    "MicrostripLine (IncrementalInductanceCurrentDistribution)": MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        t=35e-6,
        current_distribution=IncrementalInductanceCurrentDistribution(),
        length=0.1,
    ),
    "MicrostripLine (TraceGroundCurrentDistribution)": MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        t=35e-6,
        current_distribution=TraceGroundCurrentDistribution(),
        length=0.1,
    ),
    "StriplineLine": StriplineLine(
        w=2.655e-3,
        b=3.2e-3,
        t=35e-6,
        dielectric=ConstantDielectric(ep_r=2.2, tand=0.001),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=0.1,
    ),
    "StriplineLine (zero thickness)": StriplineLine(
        dielectric=ConstantDielectric(ep_r=2.2, tand=0.001), t=None, length=0.1
    ),
    "FloatingLine": FloatingLine(floating=PhaseLine(z0=50.0, theta=90.0, f0=5e9)),
}

MATERIALS = {
    "ConstantDielectric": ConstantDielectric(ep_r=4.3, tand=0.02, sigma=1e-3),
    "DjordjevicSarkarDielectric": DjordjevicSarkarDielectric(sigma=1e-3),
    # One undamped Lorentz pole resonates above the 0-10 GHz sweep; the damped
    # one may sit inside it.
    "PoleDielectric": PoleDielectric(
        ConstantDielectric(ep_r=2.0, sigma=1e-3),
        poles=(
            DebyePole(dep_r=0.5, f_relax=1e9),
            LorentzPole(dep_r=0.3, f_res=5e9, zeta=0.2),
            LorentzPole(dep_r=0.1, f_res=50e9),
        ),
        mu_poles=(LorentzPole(dep_r=0.2, f_res=3e9, zeta=0.5),),
    ),
    "ColeColeDielectric": ColeColeDielectric(dep_r=0.5, f_relax=1e9, alpha=0.3, sigma=1e-3),
    "TabulatedDielectric": TabulatedDielectric(
        f=jnp.array([0.0, 5e9, 10e9]),
        ep_r=jnp.array([4.3 - 0.1j, 4.2 - 0.1j, 4.1 - 0.1j]),
        sigma=1e-3,
    ),
    "BulkConductor": BulkConductor(sigma=1 / 1.72e-8),
    "RoughConductor": RoughConductor(sigma=1 / 1.72e-8, roughness=HammerstadRoughness(1e-6)),
}


def concrete_subclasses(base):
    """Every instantiable class under `base`, so a new model cannot slip the net.

    A concrete class that is itself subclassed still counts -- `BulkConductor`
    is directly usable as well as being the base of `RoughConductor`.
    """
    found = set()
    pending = [base]
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if (
            cls is not base
            and cls.__module__.startswith("pmrf.")
            and not getattr(cls, "__abstractmethods__", None)
        ):
            found.add(cls)
    return found
