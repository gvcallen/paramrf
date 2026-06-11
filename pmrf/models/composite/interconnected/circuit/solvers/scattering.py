"""pmrf/simulate/solvers/scattering.py"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import lineax as lx
from typing import Tuple, Optional

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
        z0_ports: Optional[jax.Array], 
        z0_ext: Optional[jax.Array], 
        topology: PortRepresentation
    ) -> ScatteringResult:
        N_act = s_block_diagonal.shape[-1]
        N_ext = len(topology.ext_net_ids)
        
        # Statically inject the VNA probes (S=0)
        pad_width = ((0, N_ext), (0, N_ext))
        s_bd = jnp.pad(s_block_diagonal, pad_width, mode='constant')
        
        # Wire the active topology and the VNA probes together
        # The active ports use port_to_net_map. The VNA probes connect to ext_net_ids.
        net_map = jnp.concatenate([topology.port_to_net_map, topology.ext_net_ids])
        
        # The external indices are simply the newly added probes at the end of the array
        ext_idx = jnp.arange(N_act, N_act + N_ext)
        
        z0 = jnp.concatenate([z0_ports, z0_ext])
        S_trav = s2s(s_bd, z0, 'traveling', 'power')

        # Standard Hallbjörner generalized wave reduction
        X = build_connection_matrix(net_map, z0, S_trav.dtype)
        T = jnp.eye(net_map.shape[0], dtype=S_trav.dtype) - S_trav @ X
        
        S_ext_trav = reduce_to_external(T, X, ext_idx, self.eps, self.linear_solver)
        
        if z0_ports is None:
            return ScatteringResult(s=S_ext_trav, z0=z0_ext)
        else:
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
        z0_ports: Optional[jax.Array], 
        z0_ext: Optional[jax.Array],
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        N_act = s_block_diagonal.shape[-1]
        N_ext = len(topology.ext_net_ids)
        
        # Statically inject the VNA probes (S=0)
        pad_width = ((0, N_ext), (0, N_ext))
        s_bd = jnp.pad(s_block_diagonal, pad_width, mode='constant')
        
        # Wire the active topology and the VNA probes together
        net_map_np = np.concatenate([topology.port_to_net_map, topology.ext_net_ids])
        ext_idx_np = np.arange(N_act, N_act + N_ext)
        
        # Map the nets
        all_net_ids = np.unique(net_map_np)
        ext_net_ids = np.unique(topology.ext_net_ids)
        pure_int_net_ids = np.setdiff1d(all_net_ids, ext_net_ids)

        z0 = jnp.concatenate([z0_ports, z0_ext])
        S = s2s(s_bd, z0, 'traveling', 'power')

        # Sequentially fold purely internal nets
        for net_id in pure_int_net_ids:
            idx_list = np.where(net_map_np == net_id)[0]
            if len(idx_list) < 2:
                continue
                
            idx = jnp.array(idx_list)
            z0_local = z0[idx] if z0 is not None else None
            X_local = build_local_connection_matrix(len(idx_list), z0_local, S.dtype)
            
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
        
        if z0_ports is None:
            return ScatteringResult(s=S_ext_trav, z0=z0_ext)
        else:
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
        z0_ports: Optional[jax.Array], 
        z0_ext: Optional[jax.Array],
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        N_act = s_block_diagonal.shape[-1]
        N_ext = len(topology.ext_net_ids)
        
        # Statically inject the VNA probes (S=0)
        pad_width = ((0, N_ext), (0, N_ext))
        s_bd = jnp.pad(s_block_diagonal, pad_width, mode='constant')
        
        # Wire the active topology and the VNA probes together
        net_map_np = np.concatenate([topology.port_to_net_map, topology.ext_net_ids])
        ext_idx_np = np.arange(N_act, N_act + N_ext)
        
        z0 = jnp.concatenate([z0_ports, z0_ext])
        S = s2s(s_bd, z0, 'traveling', 'power')

        active_ports = list(range(S.shape[-1]))
        
        all_net_ids = np.unique(net_map_np)
        ext_net_ids = np.unique(topology.ext_net_ids)
        pure_int_net_ids = np.setdiff1d(all_net_ids, ext_net_ids)
        
        # Sequentially contract purely internal nets
        for net_id in pure_int_net_ids:
            net_ports = np.where(net_map_np == net_id)[0]
            if len(net_ports) < 2:
                continue
                
            local_idx = [active_ports.index(p) for p in net_ports]
            idx_jnp = jnp.array(local_idx)
            
            z0_local = z0[net_ports] if z0 is not None else None
            X_local = build_local_connection_matrix(len(net_ports), z0_local, S.dtype)
            
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
        rem_z0 = z0[active_global_idx] if z0 is not None else None
        
        mask_filter = np.isin(rem_net_map, ext_net_ids)
        X = build_connection_matrix(rem_net_map, rem_z0, S.dtype, mask_filter=mask_filter)
        
        T = jnp.eye(len(active_ports), dtype=S.dtype) - S @ X
        
        # Map global ext_idx to the local indices of the currently contracted matrix
        local_ext_idx = jnp.array([active_ports.index(p) for p in ext_idx_np])
        
        S_ext_trav = reduce_to_external(T, X, local_ext_idx, self.eps, self.linear_solver)
        
        if z0_ports is None:
            return ScatteringResult(s=S_ext_trav, z0=z0_ext)
        else:
            return ScatteringResult(s=s2s(S_ext_trav, z0_ext, 'power', 'traveling'), z0=z0_ext)


def build_connection_matrix(
    net_map: np.ndarray, 
    z0: Optional[jax.Array], 
    dtype: jnp.dtype, 
    mask_filter: np.ndarray = None
) -> jax.Array:
    """
    Builds the generalized wave connection matrix (X) for the provided nets.
    If z0 is None, it uses the optimized topological formulation:
    $$ X_{ij} = \\frac{2}{n} - \\delta_{ij} $$
    """
    N = net_map.shape[0]
    mask = jnp.array(net_map[:, None] == net_map[None, :], dtype=dtype)
    
    if mask_filter is not None:
        mask = mask * mask_filter[:, None]
        
    if z0 is None:
        # Uniform impedance topology-only generation
        n_ports = jnp.sum(mask, axis=1)
        n_ports = jnp.where(n_ports == 0, 1.0, n_ports) # Prevent division by zero
        X = mask * (2.0 / n_ports[:, None] - jnp.eye(N, dtype=dtype))
    else:
        # Generalized impedance generation
        y0 = 1.0 / z0
        y0_sum = jnp.dot(mask, y0)
        y0_sum = jnp.where(y0_sum == 0, 1.0, y0_sum) # Prevent division by zero
        
        y0_sqrt_prod = jnp.sqrt(y0[:, None] * y0[None, :])
        X = mask * ((2.0 * y0_sqrt_prod) / y0_sum[:, None] - jnp.eye(N, dtype=dtype))
        
    return X


def build_local_connection_matrix(
    n_ports: int, 
    z0_local: Optional[jax.Array], 
    dtype: jnp.dtype
) -> jax.Array:
    """Builds the connection matrix (X) for an isolated subset of ports."""
    if z0_local is None:
        return (2.0 / n_ports) * jnp.ones((n_ports, n_ports), dtype=dtype) - jnp.eye(n_ports, dtype=dtype)
    else:
        y0_local = 1.0 / z0_local
        y0_sum = jnp.sum(y0_local)
        y0_sqrt_prod = jnp.sqrt(y0_local[:, None] * y0_local[None, :])
        return (2.0 * y0_sqrt_prod) / y0_sum - jnp.eye(n_ports, dtype=dtype)


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