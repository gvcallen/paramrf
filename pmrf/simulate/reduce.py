"""pmrf/simulate/reduce.py"""

from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpy as np

from pmrf.frequency import Frequency
from pmrf.topology import Topology
from pmrf.simulate.base import AbstractReducer, AbstractAdmittanceReducer, AbstractScatteringReducer, PortRepresentation, NodalRepresentation
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
        rep = topology_to_ports(topology)
        batched_S, batched_z0 = topology.evaluate_scattering(frequency, z0=z0, layout='block_diagonal')
        
        if not jnp.isscalar(z0):
            raise Exception("Reduce currently only accepts scalar characteristic impedances")
        
        # in_axes for batched_z0 is None, as it has no Frequency dimension
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None, None))
        solution = vmapped_solver(batched_S, batched_z0, rep)
        
        return SimulateResult(
            solution=solution,
            z0=z0,
        )
        
    elif isinstance(solver, AbstractAdmittanceReducer):
        rep = topology_to_nodal(topology)
        batched_Y_elements = topology.evaluate_admittance(frequency, layout='flattened')
        
        # batched_Y_elements is (F, N*N). The representation has no frequency axis.
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None))
        solution = vmapped_solver(batched_Y_elements, rep)
        
        return SimulateResult(
            solution=solution,
            z0=z0,
        )
        
    else:
        raise TypeError(f"Unrecognized solver type: {type(solver)}")
    
    
def topology_to_ports(topology: Topology) -> PortRepresentation:
    """Generates the static S-Parameter topological representation."""
    num_ports = sum(m.nports for m in topology.models)
    port_to_net_map = topology.connected_components()
    
    # Assumes circuit.port_idxs contains flat global port indices
    ext_idx = np.array(topology.port_indices or [], dtype=int)
    int_idx = np.setdiff1d(np.arange(num_ports), ext_idx)
    
    return PortRepresentation(
        num_ports=num_ports,
        ext_idx=ext_idx,
        int_idx=int_idx,
        port_to_net_map=port_to_net_map
    )    

def topology_to_nodal(topology: Topology) -> NodalRepresentation:
    """Generates the static Y-Parameter topological representation using a dummy ground node."""
    port_to_net_map = topology.connected_components()
    unique_nets = np.unique(port_to_net_map)
    
    # Identify which nets represent ground
    ground_nets = set()
    for p in (topology.ground_indices or []):
        ground_nets.add(port_to_net_map[p])
        
    # Remap active nets to (0 ... V-1) and ground nets to a dummy node (V)
    num_active = 0
    remap = {}
    for net in unique_nets:
        if net not in ground_nets:
            remap[net] = num_active
            num_active += 1
            
    for net in ground_nets:
        remap[net] = num_active # Dummy ground node at the end
        
    final_port_nodes = np.array([remap[net] for net in port_to_net_map], dtype=int)
    
    # Build r_idx and c_idx by unrolling the local port matrices
    r_idx, c_idx = [], []
    offset = 0
    for m in topology.models:
        n = m.nports
        nodes = final_port_nodes[offset:offset+n]
        
        # Vectorized meshgrid equivalent
        r_idx.extend(np.repeat(nodes, n))
        c_idx.extend(np.tile(nodes, n))
        
        offset += n
        
    # Identify external and internal active nets
    ext_nets = set()
    for p in (topology.port_indices or []):
        net = final_port_nodes[p]
        if net != num_active: # Exclude the dummy ground node
            ext_nets.add(net)
            
    ext_idx = np.array(sorted(list(ext_nets)), dtype=int)
    all_active = set(range(num_active))
    int_idx = np.array(sorted(list(all_active - ext_nets)), dtype=int)
    
    return NodalRepresentation(
        num_nodes=num_active + 1, # +1 creates the space for the dummy ground node
        r_idx=np.array(r_idx, dtype=int),
        c_idx=np.array(c_idx, dtype=int),
        ext_idx=ext_idx,
        int_idx=int_idx
    )    
