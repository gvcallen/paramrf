"""pmrf/simulate/reduce.py"""

from typing import TypeVar

import jax
from jaxtyping import ArrayLike

from pmrf.frequency import Frequency
from pmrf.simulate.topology import Topology
from pmrf.simulate.base import AbstractReducer, AbstractAdmittanceReducer, AbstractScatteringReducer
from pmrf.simulate.result import SimulateResult

TopologyT = TypeVar('TopologyT', bound=Topology)

def reduce(
    topology: TopologyT,
    frequency: Frequency,
    solver: AbstractReducer,
    z0: ArrayLike = 50.0,
) -> SimulateResult:
    """
    Rerduce a topology down to its external network parameters.
    
    Args:
        topology: The Topology containing sub-models and connections.
        frequency: The frequency sweep over which to characterize the network.
        solver: An instance of a network characterizer.
        z0: The characteristic impedance for S-parameter evaluation.
        
    Returns:
        SimulateResult: A structured result containing the network matrices.
    """
    
    if isinstance(solver, AbstractScatteringReducer):
        rep = topology.to_ports()
        # Removed the redundant 'topology' argument
        batched_S, batched_z0 = topology.evaluate_scattering(frequency, z0=z0, layout=solver.layout)
        
        # in_axes for batched_z0 is None, as it has no Frequency dimension
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None, None))
        solution = vmapped_solver(batched_S, batched_z0, rep)
        
        return SimulateResult(
            solution=solution,
            z0=z0,
        )
        
    elif isinstance(solver, AbstractAdmittanceReducer):
        rep = topology.to_nodal()
        # Removed the redundant 'topology' argument
        batched_Y_elements = topology.evaluate_admittance(frequency)
        
        # batched_Y_elements is (F, N*N). The representation has no frequency axis.
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None))
        solution = vmapped_solver(batched_Y_elements, rep)
        
        return SimulateResult(
            solution=solution,
            z0=z0,
        )
        
    else:
        raise TypeError(f"Unrecognized solver type: {type(solver)}")