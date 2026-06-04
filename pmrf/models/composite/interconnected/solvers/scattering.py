"""pmrf/simulate/solvers/scattering.py"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import lineax as lx

from pmrf.models.composite.interconnected.base import (
    AbstractScatteringCircuitSolver,
    PortRepresentation,
    ScatteringResult,
)

from pmrf.rf.conversions import s2s
from pmrf.math import nudge_diag


class GlobalScatteringCircuitSolver(AbstractScatteringCircuitSolver):
    """
    Global S-parameter reduction solver.
    
    Assembles the full system into a single matrix and solves for the external 
    ports simultaneously. Best suited for arbitrary meshes and, when paired with 
    a sparse solver, massive networks (e.g., >10,000 nodes).
    """
    #: Numerical regularization to prevent singular matrix division during lossless resonance.
    eps: float = eqx.field(default=1e-12, static=True)
    
    #: The lineax solver to use for the global matrix inversion. Defaults to AutoLinearSolver.
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=False), static=True
    )

    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        # --- Virtual probe injection ---
        ext_idx_np = np.array(topology.ext_idx)
        net_map_np = np.array(topology.port_to_net_map)
        
        net_counts = np.bincount(net_map_np)
        ext_net_counts = net_counts[net_map_np[ext_idx_np]]
        
        dangling_mask = ext_net_counts == 1
        num_dangling = np.sum(dangling_mask)
        
        if num_dangling > 0:
            N_orig = s_block_diagonal.shape[-1]
            pad_width = ((0, num_dangling), (0, num_dangling))
            s_block_diagonal = jnp.pad(s_block_diagonal, pad_width, mode='constant')
            
            dangling_ext_indices = ext_idx_np[dangling_mask]
            z0_new = z0_ports[dangling_ext_indices]
            z0_ports = jnp.concatenate([z0_ports, z0_new])
            
            new_port_indices = np.arange(N_orig, N_orig + num_dangling)
            net_map_np = np.concatenate([net_map_np, net_map_np[dangling_ext_indices]])
            int_idx_np = np.concatenate([topology.int_idx, dangling_ext_indices])
            ext_idx_np[dangling_mask] = new_port_indices
        else:
            int_idx_np = np.array(topology.int_idx)
            
        N = net_map_np.shape[0]
        S_trav = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        
        # --- Sparse vs Dense Assembly of the Connection Matrix (X) ---
        mask = jnp.array(net_map_np[:, None] == net_map_np[None, :], dtype=s_block_diagonal.dtype)
        y0 = 1.0 / z0_ports
        y0_sum = jnp.dot(mask, y0) 
        y0_sqrt_prod = jnp.sqrt(y0[:, None] * y0[None, :])
        X = mask * ((2.0 * y0_sqrt_prod) / y0_sum[:, None] - jnp.eye(N, dtype=s_block_diagonal.dtype))
        
        T = jnp.eye(N, dtype=s_block_diagonal.dtype) - S_trav @ X
        
        T_A = T[jnp.ix_(ext_idx_np, ext_idx_np)]
        T_B = T[jnp.ix_(ext_idx_np, int_idx_np)]
        T_C = T[jnp.ix_(int_idx_np, ext_idx_np)]
        T_D = T[jnp.ix_(int_idx_np, int_idx_np)]
        
        X_A = X[jnp.ix_(ext_idx_np, ext_idx_np)]
        X_B = X[jnp.ix_(ext_idx_np, int_idx_np)]
        
        # --- Schur Complement via lineax ---
        T_D_reg = T_D + self.eps * jnp.eye(T_D.shape[0], dtype=T_D.dtype)
        operator_D = lx.MatrixLinearOperator(T_D_reg)
        
        tmp_mat = jax.vmap(
            lambda b: lx.linear_solve(operator_D, b, self.linear_solver).value,
            in_axes=1, out_axes=1
        )(T_C)
        
        numerator = X_A - X_B @ tmp_mat
        denominator = T_A - T_B @ tmp_mat
        
        den_reg = denominator + self.eps * jnp.eye(denominator.shape[0], dtype=denominator.dtype)
        operator_den = lx.MatrixLinearOperator(den_reg.T)
        
        S_ext_trav = jax.vmap(
            lambda b: lx.linear_solve(operator_den, b, self.linear_solver).value,
            in_axes=1, out_axes=1
        )(numerator.T).T
        
        z0_ext = z0_ports[ext_idx_np]
        S_ext_power = s2s(S_ext_trav, z0_ext, 'power', 'traveling')
        
        return ScatteringResult(s=S_ext_power, z0=z0_ext)


class HierarchicalScatteringCircuitSolver(AbstractScatteringCircuitSolver):
    """
    Hierarchical S-parameter reduction solver.
    
    Operates sequentially by applying a generalized block Schur complement 
    (sub-network growth) to each internal net.
    
    Unlike the Global solver which requires a single massive matrix inversion, 
    this solver sequentially reduces networks block-by-block. This avoids huge 
    sparse matrices and scales significantly better for highly cascaded or 
    chain-like network topologies.
    """
    #: Numerical regularization to prevent singular matrix division.
    eps: float = eqx.field(default=1e-12, static=True)
    
    #: The lineax solver to use for the local block inversions.
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=False), static=True
    )

    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        ext_idx_np = np.array(topology.ext_idx)
        int_idx_np = np.array(topology.int_idx)
        net_map_np = np.array(topology.port_to_net_map)
        
        # Convert initial disconnected blocks to traveling waves
        S = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        
        # Find all unique internal nets (nets that contain internal ports to be eliminated)
        int_net_ids = np.unique(net_map_np[int_idx_np])

        # Sequentially fold each internal net into the global S-matrix
        # Because the topology arrays are evaluated as static NumPy arrays, 
        # JAX will cleanly unroll this loop during tracing/JIT.
        for net_id in int_net_ids:
            
            # Find all ports participating in this specific net
            idx_list = np.where(net_map_np == net_id)[0]
            
            # Skip if there's nothing to connect (e.g., a dangling unconnected port)
            if len(idx_list) < 2:
                continue
                
            idx = jnp.array(idx_list)
            
            # Calculate the ideal connection matrix (X_local) for this net.
            # This natively handles differing Z0 impedances (mismatches) and 
            # generalizes scikit-rf's 2-port connection to N-port nodes.
            y0_local = 1.0 / z0_ports[idx]
            y0_sum = jnp.sum(y0_local)
            y0_sqrt_prod = jnp.sqrt(y0_local[:, None] * y0_local[None, :])
            X_local = (2.0 * y0_sqrt_prod) / y0_sum - jnp.eye(len(idx_list), dtype=S.dtype)
            
            # Extract the sub-blocks of the current global S-matrix
            S_II = S[jnp.ix_(idx, idx)]
            S_All_I = S[:, idx]
            S_I_All = S[idx, :]
            
            # Formulate the local Schur complement denominator (I - X * S_II)
            M = jnp.eye(len(idx_list), dtype=S.dtype) - X_local @ S_II
            M_reg = M + self.eps * jnp.eye(len(idx_list), dtype=S.dtype)
            operator_M = lx.MatrixLinearOperator(M_reg)
            
            # Solve the local system. 
            # We vmap over the columns of (X_local @ S_I_All) to solve for all ports.
            rhs = X_local @ S_I_All
            W = jax.vmap(
                lambda b: lx.linear_solve(operator_M, b, self.linear_solver).value,
                in_axes=1, out_axes=1
            )(rhs)
            
            # Apply the sub-network growth update to the entire S-matrix
            S = S + S_All_I @ W

        # Once all internal nets have been folded sequentially, the external 
        # ports hold the fully reduced network parameters.
        S_ext_trav = S[jnp.ix_(ext_idx_np, ext_idx_np)]
        z0_ext = z0_ports[ext_idx_np]
        
        # Convert back to standard power waves
        S_ext_power = s2s(S_ext_trav, z0_ext, 'power', 'traveling')
        
        return ScatteringResult(s=S_ext_power, z0=z0_ext)


class SequentialScatteringCircuitSolver(AbstractScatteringCircuitSolver):
    """
    Sequential S-parameter reduction solver (Matrix Contraction).
    
    Physically contracts the S-matrix at each step by dropping eliminated 
    internal ports. This perfectly mirrors the classic Redheffer Star Product 
    and subnetwork growth algorithms. 
    """
    eps: float = eqx.field(default=1e-12, static=True)
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=False), static=True
    )

    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        ext_idx_np = np.array(topology.ext_idx)
        int_idx_np = np.array(topology.int_idx)
        net_map_np = np.array(topology.port_to_net_map)
        
        # --- Initialization ---
        # Note: No virtual probe padding needed for Sequential/Hierarchical solvers!
        S = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        
        active_ports = list(range(S.shape[-1]))
        ext_idx_set = set(ext_idx_np.tolist())
        
        # Only iterate over nets that actually contain internal ports to be eliminated
        int_net_ids = np.unique(net_map_np[int_idx_np])
        
        # --- Sequential Network Contraction ---
        for net_id in int_net_ids:
            net_ports = np.where(net_map_np == net_id)[0]
            
            # Skip unconnected ports
            if len(net_ports) < 2:
                continue
                
            local_idx = [active_ports.index(p) for p in net_ports]
            idx_jnp = jnp.array(local_idx)
            
            # 1. Compute the local connection matrix X for the net
            y0_local = 1.0 / z0_ports[net_ports]
            y0_sum = jnp.sum(y0_local)

            y0_sqrt_prod = jnp.sqrt(y0_local[:, None] * y0_local[None, :])
            X_local = (2.0 * y0_sqrt_prod) / y0_sum - jnp.eye(len(net_ports), dtype=S.dtype)
            
            # 2. Extract sub-blocks from the CURRENT shrinking S-matrix
            S_II = S[jnp.ix_(idx_jnp, idx_jnp)]
            S_All_I = S[:, idx_jnp]
            S_I_All = S[idx_jnp, :]
            
            # 3. Solve the local wave interactions
            M = jnp.eye(len(net_ports), dtype=S.dtype) - X_local @ S_II
            M_reg = M + self.eps * jnp.eye(len(net_ports), dtype=S.dtype)
            
            operator_M = lx.MatrixLinearOperator(M_reg)
            rhs = X_local @ S_I_All
            W = jax.vmap(
                lambda b: lx.linear_solve(operator_M, b, self.linear_solver).value,
                in_axes=1, out_axes=1
            )(rhs)
            
            # 4. Apply the generalized Redheffer update to the full matrix
            S = S + S_All_I @ W
            
            # 5. SHRINK THE MATRIX: Drop the strictly internal ports of this net
            elim_ports = [p for p in net_ports if p not in ext_idx_set]
            if not elim_ports:
                continue
                
            elim_local_idx = [active_ports.index(p) for p in elim_ports]
            keep_local_idx = [i for i in range(len(active_ports)) if i not in elim_local_idx]
            
            keep_jnp = jnp.array(keep_local_idx)
            
            # Physically reallocate the JAX array to a smaller dimension
            S = S[jnp.ix_(keep_jnp, keep_jnp)]
            
            # Update the Python tracking list
            active_ports = [active_ports[i] for i in keep_local_idx]

        # --- Final Cleanup ---
        final_ordering = [active_ports.index(p) for p in ext_idx_np]
        final_idx_jnp = jnp.array(final_ordering)
        
        S_ext_trav = S[jnp.ix_(final_idx_jnp, final_idx_jnp)]
        z0_ext = z0_ports[ext_idx_np]
        
        S_ext_power = s2s(S_ext_trav, z0_ext, 'power', 'traveling')
        
        return ScatteringResult(s=S_ext_power, z0=z0_ext)