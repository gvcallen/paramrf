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
    AbstractABCDCascader,
    AbstractABCDTerminator,
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
from pmrf.simulate.solvers.global_schur_scattering import GlobalSchurScatteringReducer
from pmrf.simulate.solvers.subnetwork_growth import SequentialSchurScatteringReducer
from pmrf.simulate.solvers.hierarchical_tree import BlockSchurScatteringReducer
from pmrf.simulate.solvers.nodal import NodalAdmittanceReducer
from pmrf.simulate.solvers.modified_nodal import ModifiedNodalAdmittanceReducer

# Cascaders
from pmrf.simulate.solvers.redheffer import RedhefferScatteringCascader
from pmrf.simulate.solvers.analytic import AnalyticABCDCascader

# Terminators
from pmrf.simulate.solvers.analytic import AnalyticScatteringTerminator
from pmrf.simulate.solvers.analytic import BilinearABCDTerminator

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
    'GlobalSchurScatteringReducer',
    'NodalAdmittanceReducer',
    'ModifiedNodalAdmittanceReducer',
    'BlockSchurScatteringReducer',
    'SequentialSchurScatteringReducer',
    'RedhefferScatteringCascader',
    'BilinearABCDTerminator',
    'AnalyticScatteringTerminator',
    'AnalyticABCDCascader',
    'AbstractABCDCascader',
    'AbstractABCDTerminator',
    'AbstractTerminator',
]