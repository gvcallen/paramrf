"""
RF models, components, surrogates, and adapters, and their solvers.

This module contains various components, surrogate models, composite building models, and model adapters.

Note that all models are re-exported under `pmrf.models`.
"""
from pmrf.models.base import (
    Model as Model,
    is_model as is_model,
    validate as validate,
)

# Adapters
from pmrf.models.adapters.base import (
    AbstractDiscrete as AbstractDiscrete,
    AbstractSingleDomain as AbstractSingleDomain,
    AbstractSingleDiscreteDomain as AbstractSingleDiscreteDomain,
)

from pmrf.models.adapters.bridge import (
    Host as Host
)

from pmrf.models.adapters.static import (
    SkrfNetwork as SkrfNetwork,
    Touchstone as Touchstone,
    SModel as SModel,
    AModel as AModel,
    YModel as YModel,
    ZModel as ZModel,
)

from pmrf.models.adapters.callable import (
    ContinuousCallable as ContinuousCallable,
    DiscreteCallable as DiscreteCallable,
)

# Components
from pmrf.models.components.ideal import (
    Load as Load,
    Short as Short,
    Open as Open,
    Match as Match,
    Port as Port,
    Ground as Ground,
    Transformer as Transformer,
    SourceConverter as SourceConverter,
    Isolator as Isolator,
    Splitter as Splitter,
    Tee as Tee,
    Attenuator as Attenuator,
    Amplifier as Amplifier,
    DirectionalCoupler as DirectionalCoupler,
)

from pmrf.models.components.lines import (
    TransmissionLine as TransmissionLine,
    FloatingLine as FloatingLine,
    RLGCLine as RLGCLine,
    PhaseLine as PhaseLine,
    ConstantRLGCLine as ConstantRLGCLine,
    PhysicalLine as PhysicalLine,
    DatasheetLine as DatasheetLine,
    CoaxialLine as CoaxialLine,
    MicrostripLine as MicrostripLine,
    WheelerMicrostripSolver as WheelerMicrostripSolver,
    TescheCoaxialSolver as TescheCoaxialSolver,
)

from pmrf.models.components.lumped import (
    Resistor as Resistor,
    Capacitor as Capacitor,
    Inductor as Inductor,
    Impedance as Impedance,
    Admittance as Admittance,
    ShuntResistor as ShuntResistor,
    ShuntCapacitor as ShuntCapacitor,
    ShuntInductor as ShuntInductor,
    CapacitorQ as CapacitorQ,
    InductorQ as InductorQ,
)

from pmrf.models.components.sections import (
    PiSection as PiSection,
    TSection as TSection,
    LSection as LSection,
    BoxSection as BoxSection,
    PiSectionCLC as PiSectionCLC,
    BoxSectionCLCC as BoxSectionCLCC,
    TSectionLCL as TSectionLCL,
    LSectionLC as LSectionLC,
)


# Composite
from pmrf.models.composite.interconnected.base import (
    AbstractCircuitSolver as AbstractCircuitSolver,
    AbstractScatteringCircuitSolver as AbstractScatteringCircuitSolver,
    AbstractAdmittanceCircuitSolver as AbstractAdmittanceCircuitSolver,
    AbstractMNACircuitSolver as AbstractMNACircuitSolver,
    ScatteringResult as ScatteringResult,
    AdmittanceResult as AdmittanceResult,
    PortRepresentation as PortRepresentation,
    NodalRepresentation as NodalRepresentation,
    MNARepresentation as MNARepresentation,
)

from pmrf.models.composite.interconnected.solvers.scattering import (
    GlobalScatteringCircuitSolver as GlobalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver as SequentialScatteringCircuitSolver,
    HierarchicalScatteringCircuitSolver as HierarchicalScatteringCircuitSolver,
)

from pmrf.models.composite.interconnected.solvers.nodal import (
    GlobalMNACircuitSolver as GlobalMNACircuitSolver,
    GlobalNodalCircuitSolver as GlobalNodalCircuitSolver,
)

from pmrf.models.composite.interconnected.circuit import Circuit as Circuit
from pmrf.models.composite.interconnected.cascade import Cascade as Cascade
from pmrf.models.composite.interconnected.terminated import Terminated as Terminated


from pmrf.models.composite.transformed import (
    Renumbered as Renumbered,
    Flipped as Flipped,
)

from pmrf.models.composite.nodal import (
    Shunt as Shunt,
    GroundLifted as GroundLifted,
    GroundExposed as GroundExposed,
    CoupledOnePorts as CoupledOnePorts,
    CoupledTwoPorts as CoupledTwoPorts,
)

from pmrf.models.composite.topological import (
    PiTopology as PiTopology,
    TTopology as TTopology,
    LTopology as LTopology,
)

from pmrf.models.composite.wrapped import (
    Tied as Tied,
    Probabilistic as Probabilistic,
)

# Surrogates
from pmrf.models.surrogates.expansion import (
    VectorExpansion as VectorExpansion
)

from pmrf.models.surrogates.rational import (
    PolynomialRatio as PolynomialRatio,
    PoleResidue as PoleResidue,
    StateSpace as StateSpace,
    BarycentricRational as BarycentricRational,
)

from pmrf.models import adapters, components, composite, surrogates

__all__ = [
    "adapters",
    "components",
    "composite",
    "surrogates",
]