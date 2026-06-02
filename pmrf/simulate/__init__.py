"""
Frequency-domain circuit simulation, such as reductions and cascades.

This is a lower-level module, and is used by the composite models in :mod:`pmrf.models`.
"""

from pmrf.simulate.base import (
    ScatteringResult,
    AdmittanceResult,
    ABCDResult,
    AbstractAdmittanceReducer,
    AbstractMNAReducer,
    AbstractScatteringReducer,
    AbstractScatteringCascader,
    AbstractABCDCascader,
    AbstractABCDTerminator,
    AbstractTerminator,
    AbstractReducer,
    AbstractCascader,
    NodalRepresentation,
    MNARepresentation,
    PortRepresentation,
)
from pmrf.simulate.reduce import reduce
from pmrf.simulate.cascade import cascade
from pmrf.simulate.terminate import terminate
from pmrf.simulate.result import SimulateResult

# Reducers
from pmrf.simulate.solvers.scattering import (
    GlobalScatteringReducer,
    SequentialScatteringReducer,
    HierarchicalScatteringReducer,
)
from pmrf.simulate.solvers.nodal import (
    GlobalNodalReducer,
    GlobalMNAReducer,
)

# Cascaders
from pmrf.simulate.solvers.scattering import SequentialScatteringCascader
from pmrf.simulate.solvers.abcd import SequentialABCDCascader

# Terminators
from pmrf.simulate.solvers.scattering import ScatteringTerminator
from pmrf.simulate.solvers.abcd import ABCDTerminator

__all__ = [
    'ScatteringResult',
    'AdmittanceResult',
    'ABCDResult',
    'AbstractAdmittanceReducer',
    'AbstractMNAReducer',
    'AbstractScatteringReducer',
    'AbstractReducer',
    'AbstractCascader',
    'reduce',
    'cascade',
    'terminate',
    'PortRepresentation',
    'NodalRepresentation',
    'MNARepresentation',
    'SimulateResult',
    'GlobalScatteringReducer',
    'GlobalNodalReducer',
    'GlobalMNAReducer',
    'HierarchicalScatteringReducer',
    'SequentialScatteringReducer',
    'SequentialScatteringCascader',
    'ABCDTerminator',
    'ScatteringTerminator',
    'SequentialABCDCascader',
    'AbstractScatteringCascader',
    'AbstractABCDCascader',
    'AbstractABCDTerminator',
    'AbstractTerminator',
]