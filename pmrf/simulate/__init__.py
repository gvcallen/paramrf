"""
Frequency-domain circuit simulation, such as reductions and cascades.

This is a lower-level module, and is used by the composite models in :mod:`pmrf.models`.

It is likely only really convenient to use to test different algorithms,
or to investigate convergence or accuracy issues.
"""

from pmrf.simulate.base import (
    ScatteringResult,
    AdmittanceResult,
    TransferResult,
    AbstractAdmittanceReducer,
    AbstractScatteringReducer,
    AbstractTransferCascader,
    AbstractTransferTerminator,
    AbstractTerminator,
    AbstractReducer,
    AbstractCascader,
    NodalRepresentation,
    PortRepresentation,
)
from pmrf.simulate.reduce import reduce
from pmrf.simulate.cascade import cascade
from pmrf.simulate.terminate import terminate
from pmrf.simulate.result import SimulateResult

# Reducers
from pmrf.simulate.solvers.hallbjorner import HallbjornerReducer
from pmrf.simulate.solvers.kron import KronReducer
from pmrf.simulate.solvers.modified_kron import ModifiedKronReducer
from pmrf.simulate.solvers.subnetwork_growth import SubnetworkGrowthReducer

# Cascaders
from pmrf.simulate.solvers.redheffer import RedhefferCascader
from pmrf.simulate.solvers.transfer import TransferCascader

# Terminators
from pmrf.simulate.solvers.linear_fractional import LinearFractionalTerminator
from pmrf.simulate.solvers.mobius import MobiusTerminator

__all__ = [
    'ScatteringResult',
    'AdmittanceResult',
    'TransferResult',
    'AbstractAdmittanceReducer',
    'AbstractScatteringReducer',
    'AbstractReducer',
    'AbstractCascader',
    'reduce',
    'cascade',
    'terminate',
    'PortRepresentation',
    'NodalRepresentation',
    'SimulateResult',
    'HallbjornerReducer',
    'KronReducer',
    'ModifiedKronReducer',
    'SubnetworkGrowthReducer',
    'RedhefferCascader',
    'MobiusTerminator',
    'LinearFractionalTerminator',
    'TransferCascader',
    'AbstractTransferCascader',
    'AbstractTransferTerminator',
    'AbstractTerminator',
]