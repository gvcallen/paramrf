from typing import TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
import numpy as np

from pmrf.frequency import Frequency
from pmrf.simulate.topology import Topology
from pmrf.simulate.base import (
    AbstractReducer, 
    AbstractAdmittanceReducer, 
    AbstractMNAReducer,
    AbstractScatteringReducer, 
    PortRepresentation, 
    NodalRepresentation,
    MNARepresentation
)
from pmrf.simulate.result import SimulateResult

TopologyT = TypeVar('TopologyT', bound=Topology)

def reduce(
    topology: TopologyT,
    frequency: Frequency,
    solver: AbstractReducer,
    z0: ArrayLike = 50.0,
) -> SimulateResult:
    """
    Reduces a topology down to its external network parameters.

    Parameters
    ----------
    topology : TopologyT
        The Topology containing sub-components and connections.
    frequency : Frequency
        The frequency sweep over which to characterize the network.
    solver : AbstractReducer
        An instance of a network reduction algorithm (e.g., Hallbjorner or Kron).
    z0 : ArrayLike, optional
        The characteristic impedance for S-parameter evaluation, by default 50.0.

    Returns
    -------
    SimulateResult
        A structured result containing the fully reduced network matrices.

    Raises
    ------
    ValueError
        If a non-scalar characteristic impedance (`z0`) is passed to a scattering reducer.
    TypeError
        If the provided solver does not inherit from `AbstractScatteringReducer` 
        or `AbstractAdmittanceReducer`.
    """
    
    if isinstance(solver, AbstractScatteringReducer):
        rep = topology_to_ports(topology)
        batched_S, batched_z0 = topology.evaluate_scattering(frequency, z0=z0, layout='block_diagonal')
        
        if not jnp.isscalar(z0):
            raise ValueError("Reduce currently only accepts scalar characteristic impedances")
        
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None, None))
        solution = vmapped_solver(batched_S, batched_z0, rep)
        return SimulateResult(
            solution=solution,
            z0=z0,
        )
    elif isinstance(solver, AbstractAdmittanceReducer):
        rep = topology_to_nodal(topology)
        batched_Y_elements = topology.evaluate_admittance(frequency, layout='flattened')
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, None))
        solution = vmapped_solver(batched_Y_elements, rep)
        
        return SimulateResult(
            solution=solution,
            z0=z0,
        )
    elif isinstance(solver, AbstractMNAReducer):
        rep = topology_to_modified_nodal(topology)
        batched_Y, batched_B, batched_C, batched_D = topology.evaluate_mna(frequency)
        
        vmapped_solver = jax.vmap(solver.run, in_axes=(0, 0, 0, 0, None))
        solution = vmapped_solver(batched_Y, batched_B, batched_C, batched_D, rep)
        
        return SimulateResult(
            solution=solution,
            z0=z0,
        )
    else:
        raise TypeError(f"Unrecognized solver type: {type(solver)}")
    
    
def topology_to_ports(topology: Topology) -> PortRepresentation:
    """
    Generates the static S-Parameter topological representation.

    Parameters
    ----------
    topology : Topology
        The parsed circuit topology containing connectivity information.

    Returns
    -------
    PortRepresentation
        The structural map defining external vs. internal ports and their nets.
    """
    num_ports = sum(c.nports for c in topology.components)
    port_to_net_map = topology.connected_components()
    ext_idx = np.array(topology.port_indices or [], dtype=int)
    int_idx = np.setdiff1d(np.arange(num_ports), ext_idx)
    
    return PortRepresentation(
        ext_idx=ext_idx,
        int_idx=int_idx,
        port_to_net_map=port_to_net_map
    )    


def topology_to_nodal(topology: Topology) -> NodalRepresentation:
    """
    Generates the static Y-Parameter topological representation using a dummy ground node.

    Parameters
    ----------
    topology : Topology
        The parsed circuit topology containing connectivity and ground reference information.

    Returns
    -------
    NodalRepresentation
        The structural map providing flattened indices for global COO scatter-add assembly,
        including partitions for external and internal nodal nets.
    """
    port_to_net_map = topology.connected_components()
    unique_nets = np.unique(port_to_net_map)
    
    ground_nets = set()
    for p in (topology.ground_indices or []):
        ground_nets.add(port_to_net_map[p])
        
    num_active = 0
    remap = {}
    for net in unique_nets:
        if net not in ground_nets:
            remap[net] = num_active
            num_active += 1
            
    for net in ground_nets:
        remap[net] = num_active
        
    final_port_nodes = np.array([remap[net] for net in port_to_net_map], dtype=int)
    
    r_idx, c_idx = [], []
    offset = 0
    for c in topology.components:
        n = c.nports
        nodes = final_port_nodes[offset:offset+n]
        
        r_idx.extend(np.repeat(nodes, n))
        c_idx.extend(np.tile(nodes, n))
        
        offset += n
        
    ext_nets = set()
    for p in (topology.port_indices or []):
        net = final_port_nodes[p]
        if net != num_active:
            ext_nets.add(net)
            
    ext_idx = np.array(sorted(list(ext_nets)), dtype=int)
    all_active = set(range(num_active))
    int_idx = np.array(sorted(list(all_active - ext_nets)), dtype=int)
    
    return NodalRepresentation(
        r_idx=np.array(r_idx, dtype=int),
        c_idx=np.array(c_idx, dtype=int),
        ext_idx=ext_idx,
        int_idx=int_idx
    )

def topology_to_modified_nodal(topology: Topology) -> MNARepresentation:
    """
    Generates the static MNA topological representation.
    """
    port_to_net_map = topology.connected_components()
    unique_nets = np.unique(port_to_net_map)
    
    ground_nets = set()
    for p in (topology.ground_indices or []):
        ground_nets.add(port_to_net_map[p])
        
    num_active = 0
    remap = {}
    for net in unique_nets:
        if net not in ground_nets:
            remap[net] = num_active
            num_active += 1
            
    for net in ground_nets:
        remap[net] = num_active
        
    final_port_nodes = np.array([remap[net] for net in port_to_net_map], dtype=int)
    
    y_r_idx, y_c_idx = [], []
    b_r_idx, b_c_idx = [], []
    c_r_idx, c_c_idx = [], []
    d_r_idx, d_c_idx = [], []
    
    offset = 0
    aux_offset = 0
    
    freq_dummy = Frequency(1, 2, 2)
    
    for c in topology.components:
        n = c.nports
        nodes = final_port_nodes[offset:offset+n]
        
        stamp_shape = jax.eval_shape(lambda comp=c: comp.mna(freq_dummy))
        k = stamp_shape.D.shape[1]
        
        aux_nodes = np.arange(aux_offset, aux_offset + k)
        
        y_r_idx.extend(np.repeat(nodes, n))
        y_c_idx.extend(np.tile(nodes, n))
        
        if k > 0:
            b_r_idx.extend(np.repeat(nodes, k))
            b_c_idx.extend(np.tile(aux_nodes, n))
            
            c_r_idx.extend(np.repeat(aux_nodes, n))
            c_c_idx.extend(np.tile(nodes, k))
            
            d_r_idx.extend(np.repeat(aux_nodes, k))
            d_c_idx.extend(np.tile(aux_nodes, k))
            
        offset += n
        aux_offset += k
        
    ext_nets = set()
    for p in (topology.port_indices or []):
        net = final_port_nodes[p]
        if net != num_active:
            ext_nets.add(net)
            
    ext_idx = np.array(sorted(list(ext_nets)), dtype=int)
    all_active = set(range(num_active))
    int_idx = np.array(sorted(list(all_active - ext_nets)), dtype=int)
    
    return MNARepresentation(
        y_r_idx=np.array(y_r_idx, dtype=int),
        y_c_idx=np.array(y_c_idx, dtype=int),
        b_r_idx=np.array(b_r_idx, dtype=int),
        b_c_idx=np.array(b_c_idx, dtype=int),
        c_r_idx=np.array(c_r_idx, dtype=int),
        c_c_idx=np.array(c_c_idx, dtype=int),
        d_r_idx=np.array(d_r_idx, dtype=int),
        d_c_idx=np.array(d_c_idx, dtype=int),
        ext_idx=ext_idx,
        int_idx=int_idx,
        aux_idx=np.arange(aux_offset, dtype=int)
    )