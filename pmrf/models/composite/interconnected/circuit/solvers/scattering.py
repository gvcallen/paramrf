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
        
        # Wire the active topology and the external nodes together
        net_map = jnp.concatenate([topology.port_to_net_map, topology.ext_net_ids])
        
        if z0_ports is not None:
            z0 = jnp.concatenate([z0_ports, z0_ext])
            C_i = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        else:
            z0 = None
            C_i = s_block_diagonal

        # Standard Hallbjörner generalized wave reduction (Eq. 8)
        X = build_connection_matrix(net_map, z0, C_i.dtype)
        
        S_ext_trav = reduce_to_external(C_i, X, N_act, self.eps, self.linear_solver)
        
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
        
        net_map_np = np.concatenate([topology.port_to_net_map, topology.ext_net_ids])
        
        # Map the nets
        all_net_ids = np.unique(net_map_np)
        ext_net_ids = np.unique(topology.ext_net_ids)
        pure_int_net_ids = np.setdiff1d(all_net_ids, ext_net_ids)

        if z0_ports is not None:
            z0 = jnp.concatenate([z0_ports, z0_ext])
            S = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        else:
            z0 = None
            S = s_block_diagonal

        # Sequentially fold purely internal nets
        for net_id in pure_int_net_ids:
            idx_list = np.where(topology.port_to_net_map == net_id)[0]
            if len(idx_list) < 2:
                continue
                
            idx = jnp.array(idx_list)
            z0_local = z0_ports[idx] if z0_ports is not None else None
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
        
        S_ext_trav = reduce_to_external(S, X, N_act, self.eps, self.linear_solver)
        
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
        
        if z0_ports is not None:
            S = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        else:
            S = s_block_diagonal

        active_ports = list(range(S.shape[-1]))
        
        net_map_act = topology.port_to_net_map
        all_net_ids = np.unique(net_map_act)
        ext_net_ids = np.unique(topology.ext_net_ids)
        pure_int_net_ids = np.setdiff1d(all_net_ids, ext_net_ids)
        
        # Sequentially contract purely internal nets
        for net_id in pure_int_net_ids:
            net_ports = np.where(net_map_act == net_id)[0]
            if len(net_ports) < 2:
                continue
                
            local_idx = [active_ports.index(p) for p in net_ports]
            idx_jnp = jnp.array(local_idx)
            
            z0_local = z0_ports[net_ports] if z0_ports is not None else None
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

        # Final Schur complement for the contracted matrix
        active_global_idx = np.array(active_ports)
        rem_net_map = net_map_act[active_global_idx]
        rem_z0 = z0_ports[active_global_idx] if z0_ports is not None else None
        
        final_net_map = np.concatenate([rem_net_map, topology.ext_net_ids])
        
        if rem_z0 is not None:
            final_z0 = jnp.concatenate([rem_z0, z0_ext])
        else:
            final_z0 = None
            
        # No mask filter needed, all remaining nodes in the contracted matrix touch an external port
        X = build_connection_matrix(final_net_map, final_z0, S.dtype)
        
        N_rem = len(active_ports)
        S_ext_trav = reduce_to_external(S, X, N_rem, self.eps, self.linear_solver)
        
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
    Builds the generalized wave connection matrix (X) for the provided nets (Eq. 8).
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
    C_i: jax.Array, 
    X: jax.Array, 
    N_act: int, 
    eps: float, 
    linear_solver: lx.AbstractLinearSolver
) -> jax.Array:
    """
    Unified global reduction using the block Schur complement (Eq. 10).
    Bypasses padding probes by partitioning the system natively into 
    internal and external boundaries.
    
    S_reduced = X_ee + X_ei * C_i * (I - X_ii * C_i)^-1 * X_ie
    """
    N_ext = X.shape[0] - N_act
    
    int_idx = jnp.arange(N_act)
    ext_idx = jnp.arange(N_act, N_act + N_ext)
    
    # Partition X into ee, ei, ie, ii blocks (Eq. 9)
    X_ee = X[jnp.ix_(ext_idx, ext_idx)]
    X_ei = X[jnp.ix_(ext_idx, int_idx)]
    X_ie = X[jnp.ix_(int_idx, ext_idx)]
    X_ii = X[jnp.ix_(int_idx, int_idx)]
    
    # Formulate linear system matrix A = (I - X_ii @ C_i)
    I_ii = jnp.eye(N_act, dtype=C_i.dtype)
    A = I_ii - X_ii @ C_i
    
    # Apply Tikhonov regularization for resonant singularities
    A_reg = A + eps * I_ii
    operator_A = lx.MatrixLinearOperator(A_reg)
    
    # Solve for Y = (I - X_ii @ C_i)^-1 @ X_ie
    # This replaces explicit matrix inversion with a batched linear solve
    Y = jax.vmap(
        lambda b: lx.linear_solve(operator_A, b, linear_solver).value,
        in_axes=1, out_axes=1
    )(X_ie)
    
    # Compute reduced external S-parameters (Eq. 10)
    return X_ee + X_ei @ C_i @ Y