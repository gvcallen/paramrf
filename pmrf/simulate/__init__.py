"""
A module for simulating different circuits, for example general port reductions and cascades.

This is a lower-level module, and is used by the composite models in :mod:`pmrf.models`.

It is likely only really convenient to use to test different algorithms,
or to investigate convergenceor accuracy issues.
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

from pmrf.simulate.solvers.hallbjorner import Hallbjorner
from pmrf.simulate.solvers.kron import Kron
from pmrf.simulate.solvers.redheffer import Redheffer
from pmrf.simulate.solvers.transfer_cascader import TransferCascader
from pmrf.simulate.solvers.linear_fractional_terminator import LinearFractionalTerminator
from pmrf.simulate.solvers.mobius_terminator import MobiusTerminator

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
    'Hallbjorner',
    'Kron',
    'Redheffer',
    'MobiusTerminator',
    'LinearFractionalTerminator',
    'TransferCascader',
    'AbstractTransferCascader',
    'AbstractTransferTerminator',
    'AbstractTerminator',
]