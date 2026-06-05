"""pmrf/simulate/solvers/scattering.py"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import lineax as lx
from typing import Tuple

from pmrf.models.composite.interconnected.circuit.base import (
    AbstractScatteringCircuitSolver,
    PortRepresentation,
    ScatteringResult,
)

from pmrf.rf.conversions import s2s


class GlobalScatteringCircuitSolver(AbstractScatteringCircuitSolver):
    """
    Global S-parameter reduction solver.
    
    Assembles the full system into a single matrix and solves for the external 
    ports simultaneously.
    
    Best suited for arbitrary circuits.
    """
    #: Numerical regularization to prevent singular matrix division during lossless resonance.
    eps: float = eqx.field(default=1e-12, static=True)
    
    #: The lineax solver to use for the global matrix inversion. Defaults to AutoLinearSolver.
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=None), static=True
    )

    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        s_bd, z0, ext_idx_np, int_idx_np, net_map_np = inject_virtual_probes(
            s_block_diagonal, z0_ports, 
            np.array(topology.ext_idx), 
            np.array(topology.int_idx), 
            np.array(topology.port_to_net_map)
        )

        S_trav = s2s(s_bd, z0, 'traveling', 'power')
        X = build_connection_matrix(net_map_np, z0, S_trav.dtype)
        
        T = jnp.eye(net_map_np.shape[0], dtype=S_trav.dtype) - S_trav @ X
        
        S_ext_trav = reduce_to_external(T, X, jnp.array(ext_idx_np), self.eps, self.linear_solver)
        
        z0_ext = z0[ext_idx_np]
        return ScatteringResult(s=s2s(S_ext_trav, z0_ext, 'power', 'traveling'), z0=z0_ext)


class HierarchicalScatteringCircuitSolver(AbstractScatteringCircuitSolver):
    """
    Hierarchical S-parameter reduction solver.
    
    Operates sequentially by applying a generalized block Schur complement 
    (sub-network growth) to each internal net block-by-block.
    Unrolls the circuit topology into the algorithm, which may
    result in increased compile times.

    Best suited for circuits that contain a mix of chain-like
    topology and arbitrary interconnections.
    """
    eps: float = eqx.field(default=1e-12, static=True)
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=None), static=True
    )

    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        s_bd, z0, ext_idx_np, _, net_map_np = inject_virtual_probes(
            s_block_diagonal, z0_ports, 
            np.array(topology.ext_idx), 
            np.array(topology.int_idx), 
            np.array(topology.port_to_net_map)
        )
        
        all_net_ids = np.unique(net_map_np)
        ext_net_ids = np.unique(net_map_np[ext_idx_np])
        pure_int_net_ids = np.setdiff1d(all_net_ids, ext_net_ids)

        S = s2s(s_bd, z0, 'traveling', 'power')

        # Sequentially fold purely internal nets
        for net_id in pure_int_net_ids:
            idx_list = np.where(net_map_np == net_id)[0]
            if len(idx_list) < 2:
                continue
                
            idx = jnp.array(idx_list)
            X_local = build_local_connection_matrix(z0[idx], S.dtype)
            
            S_II = S[jnp.ix_(idx, idx)]
            S_All_I = S[:, idx]
            S_I_All = S[idx, :]
            
            M = jnp.eye(len(idx_list), dtype=S.dtype) - X_local @ S_II
            M_reg = M + self.eps * jnp.eye(len(idx_list), dtype=S.dtype)
            operator_M = lx.MatrixLinearOperator(M_reg)
            
            rhs = X_local @ S_I_All
            W = jax.vmap(
                lambda b: lx.linear_solve(operator_M, b, self.linear_solver).value,
                in_axes=1, out_axes=1
            )(rhs)
            
            # Sub-network growth
            S = S + S_All_I @ W

        # Resolve mixed nets (touching external ports)
        mask_filter = np.isin(net_map_np, ext_net_ids)
        X = build_connection_matrix(net_map_np, z0, S.dtype, mask_filter=mask_filter)
        
        T = jnp.eye(net_map_np.shape[0], dtype=S.dtype) - S @ X
        
        S_ext_trav = reduce_to_external(T, X, jnp.array(ext_idx_np), self.eps, self.linear_solver)
        
        z0_ext = z0[ext_idx_np]
        return ScatteringResult(s=s2s(S_ext_trav, z0_ext, 'power', 'traveling'), z0=z0_ext)


class SequentialScatteringCircuitSolver(AbstractScatteringCircuitSolver):
    """
    Sequential S-parameter reduction solver (Matrix Contraction).
    
    Physically contracts the S-matrix at each step by dropping eliminated 
    internal ports. Operates sequentially on purely internal nets, then uses a 
    final block Schur complement on the contracted matrix to resolve 
    external boundaries. Unrolls the circuit topology into the algorithm,
    which may result in increased compile times.

    Best suited for long chain-like/separated networks,
    though consider using an explicit Cascade if possible.
    
    """
    eps: float = eqx.field(default=1e-12, static=True)
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=None), static=True
    )

    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        s_bd, z0, ext_idx_np, _, net_map_np = inject_virtual_probes(
            s_block_diagonal, z0_ports, 
            np.array(topology.ext_idx), 
            np.array(topology.int_idx), 
            np.array(topology.port_to_net_map)
        )
        
        S = s2s(s_bd, z0, 'traveling', 'power')
        active_ports = list(range(S.shape[-1]))
        
        all_net_ids = np.unique(net_map_np)
        ext_net_ids = np.unique(net_map_np[ext_idx_np])
        pure_int_net_ids = np.setdiff1d(all_net_ids, ext_net_ids)
        
        # Sequentially contract purely internal nets
        for net_id in pure_int_net_ids:
            net_ports = np.where(net_map_np == net_id)[0]
            if len(net_ports) < 2:
                continue
                
            local_idx = [active_ports.index(p) for p in net_ports]
            idx_jnp = jnp.array(local_idx)
            
            X_local = build_local_connection_matrix(z0[net_ports], S.dtype)
            
            S_II = S[jnp.ix_(idx_jnp, idx_jnp)]
            S_All_I = S[:, idx_jnp]
            S_I_All = S[idx_jnp, :]
            
            M = jnp.eye(len(net_ports), dtype=S.dtype) - X_local @ S_II
            M_reg = M + self.eps * jnp.eye(len(net_ports), dtype=S.dtype)
            operator_M = lx.MatrixLinearOperator(M_reg)
            
            rhs = X_local @ S_I_All
            W = jax.vmap(
                lambda b: lx.linear_solve(operator_M, b, self.linear_solver).value,
                in_axes=1, out_axes=1
            )(rhs)
            
            S = S + S_All_I @ W
            
            # Drop eliminated internal ports and shrink tracking structures
            keep_local_idx = [i for i in range(len(active_ports)) if i not in local_idx]
            keep_jnp = jnp.array(keep_local_idx)
            
            S = S[jnp.ix_(keep_jnp, keep_jnp)]
            active_ports = [active_ports[i] for i in keep_local_idx]

        # Resolve mixed nets on the remaining contracted matrix
        active_global_idx = np.array(active_ports)
        rem_net_map = net_map_np[active_global_idx]
        rem_z0 = z0[active_global_idx]
        
        mask_filter = np.isin(rem_net_map, ext_net_ids)
        X = build_connection_matrix(rem_net_map, rem_z0, S.dtype, mask_filter=mask_filter)
        
        T = jnp.eye(len(active_ports), dtype=S.dtype) - S @ X
        
        # Map global ext_idx to the local indices of the currently contracted matrix
        local_ext_idx = jnp.array([active_ports.index(p) for p in ext_idx_np])
        
        S_ext_trav = reduce_to_external(T, X, local_ext_idx, self.eps, self.linear_solver)
        
        z0_ext = z0[ext_idx_np]
        return ScatteringResult(s=s2s(S_ext_trav, z0_ext, 'power', 'traveling'), z0=z0_ext)
    

def inject_virtual_probes(
    s_block_diagonal: jax.Array, 
    z0_ports: jax.Array, 
    ext_idx: np.ndarray, 
    int_idx: np.ndarray, 
    net_map: np.ndarray
) -> Tuple[jax.Array, jax.Array, np.ndarray, np.ndarray, np.ndarray]:
    """Handles dangling external ports by injecting virtual matched probes."""
    net_counts = np.bincount(net_map)
    ext_net_counts = net_counts[net_map[ext_idx]]
    
    dangling_mask = ext_net_counts == 1
    num_dangling = np.sum(dangling_mask)
    
    if num_dangling > 0:
        N_orig = s_block_diagonal.shape[-1]
        pad_width = ((0, num_dangling), (0, num_dangling))
        s_block_diagonal = jnp.pad(s_block_diagonal, pad_width, mode='constant')
        
        dangling_ext_indices = ext_idx[dangling_mask]
        z0_new = z0_ports[dangling_ext_indices]
        z0_ports = jnp.concatenate([z0_ports, z0_new])
        
        new_port_indices = np.arange(N_orig, N_orig + num_dangling)
        net_map = np.concatenate([net_map, net_map[dangling_ext_indices]])
        int_idx = np.concatenate([int_idx, dangling_ext_indices])
        ext_idx[dangling_mask] = new_port_indices
        
    return s_block_diagonal, z0_ports, ext_idx, int_idx, net_map


def build_connection_matrix(
    net_map: np.ndarray, 
    z0_ports: jax.Array, 
    dtype: jnp.dtype, 
    mask_filter: np.ndarray = None
) -> jax.Array:
    """Builds the generalized wave connection matrix (X) for the provided nets."""
    N = net_map.shape[0]
    mask = jnp.array(net_map[:, None] == net_map[None, :], dtype=dtype)
    
    if mask_filter is not None:
        mask = mask * mask_filter[:, None]
        
    y0 = 1.0 / z0_ports
    y0_sum = jnp.dot(mask, y0)
    y0_sum = jnp.where(y0_sum == 0, 1.0, y0_sum) # Prevent division by zero
    
    y0_sqrt_prod = jnp.sqrt(y0[:, None] * y0[None, :])
    X = mask * ((2.0 * y0_sqrt_prod) / y0_sum[:, None] - jnp.eye(N, dtype=dtype))
    return X


def build_local_connection_matrix(z0_local: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Builds the connection matrix (X) for an isolated subset of ports."""
    y0_local = 1.0 / z0_local
    y0_sum = jnp.sum(y0_local)
    y0_sqrt_prod = jnp.sqrt(y0_local[:, None] * y0_local[None, :])
    return (2.0 * y0_sqrt_prod) / y0_sum - jnp.eye(z0_local.shape[0], dtype=dtype)


def reduce_to_external(
    T: jax.Array, 
    X: jax.Array, 
    ext_idx: jax.Array, 
    eps: float, 
    linear_solver: lx.AbstractLinearSolver
) -> jax.Array:
    """
    Unified global reduction. 
    Bypasses block Schur partitions by solving the external excitation globally.
    """
    N = T.shape[0]
    N_ext = len(ext_idx)
    
    # Form the excitation matrix (Identity for ext ports, zeros for int ports)
    E = jnp.zeros((N, N_ext), dtype=T.dtype)
    E = E.at[ext_idx, :].set(jnp.eye(N_ext, dtype=T.dtype))
    
    T_reg = T + eps * jnp.eye(N, dtype=T.dtype)
    operator_T = lx.MatrixLinearOperator(T_reg)
    
    # Single unified linear solve
    Y = jax.vmap(
        lambda b: lx.linear_solve(operator_T, b, linear_solver).value,
        in_axes=1, out_axes=1
    )(E)
    
    # S_ext is precisely the external rows of X multiplied by the global solution
    X_ext = X[ext_idx, :]
    return X_ext @ Y