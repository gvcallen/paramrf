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
        
        # Connection Matrix (X)
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
        
        # Schur Complement
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
        # -------------------------------
        
        # Identify purely internal nets
        all_net_ids = np.unique(net_map_np)
        ext_net_ids = np.unique(net_map_np[ext_idx_np])
        pure_int_net_ids = np.setdiff1d(all_net_ids, ext_net_ids)

        S = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')

        # Sequentially fold purely internal nets
        for net_id in pure_int_net_ids:
            idx_list = np.where(net_map_np == net_id)[0]
            
            if len(idx_list) < 2:
                continue
                
            idx = jnp.array(idx_list)
            
            y0_local = 1.0 / z0_ports[idx]
            y0_sum = jnp.sum(y0_local)
            y0_sqrt_prod = jnp.sqrt(y0_local[:, None] * y0_local[None, :])
            X_local = (2.0 * y0_sqrt_prod) / y0_sum - jnp.eye(len(idx_list), dtype=S.dtype)
            
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
        N = net_map_np.shape[0]
        
        # Build the connection matrix ONLY for the remaining external-facing nets
        mask = jnp.array(net_map_np[:, None] == net_map_np[None, :], dtype=S.dtype)
        is_ext_net_port_np = np.isin(net_map_np, ext_net_ids)
        mask_ext = mask * is_ext_net_port_np[:, None]
        
        y0 = 1.0 / z0_ports
        y0_sum_ext = jnp.dot(mask_ext, y0)
        # Prevent division by zero for purely internal ports that are masked out
        y0_sum_ext = jnp.where(y0_sum_ext == 0, 1.0, y0_sum_ext) 
        
        y0_sqrt_prod = jnp.sqrt(y0[:, None] * y0[None, :])
        X = mask_ext * ((2.0 * y0_sqrt_prod) / y0_sum_ext[:, None] - jnp.eye(N, dtype=S.dtype))
        
        # Calculate final external boundaries looking into the nodes (X)
        T = jnp.eye(N, dtype=S.dtype) - S @ X
        
        T_A = T[jnp.ix_(ext_idx_np, ext_idx_np)]
        T_B = T[jnp.ix_(ext_idx_np, int_idx_np)]
        T_C = T[jnp.ix_(int_idx_np, ext_idx_np)]
        T_D = T[jnp.ix_(int_idx_np, int_idx_np)]
        
        X_A = X[jnp.ix_(ext_idx_np, ext_idx_np)]
        X_B = X[jnp.ix_(ext_idx_np, int_idx_np)]
        
        # Invert the highly reduced internal partition
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


class SequentialScatteringCircuitSolver(AbstractScatteringCircuitSolver):
    """
    Sequential S-parameter reduction solver (Matrix Contraction).
    
    Physically contracts the S-matrix at each step by dropping eliminated 
    internal ports. Operates sequentially on purely internal nets, then uses a 
    final block Schur complement on the highly contracted matrix to resolve 
    external boundaries.
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
        # -------------------------------
        
        S = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        
        # Track the global indices of the rows/cols currently in the matrix
        active_ports = list(range(S.shape[-1]))
        
        # Identify purely internal nets
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
            
            # Compute the local connection matrix X for the net
            y0_local = 1.0 / z0_ports[net_ports]
            y0_sum = jnp.sum(y0_local)
            y0_sqrt_prod = jnp.sqrt(y0_local[:, None] * y0_local[None, :])
            X_local = (2.0 * y0_sqrt_prod) / y0_sum - jnp.eye(len(net_ports), dtype=S.dtype)
            
            # Extract sub-blocks from the CURRENT shrinking S-matrix
            S_II = S[jnp.ix_(idx_jnp, idx_jnp)]
            S_All_I = S[:, idx_jnp]
            S_I_All = S[idx_jnp, :]
            
            # Solve the local wave interactions
            M = jnp.eye(len(net_ports), dtype=S.dtype) - X_local @ S_II
            M_reg = M + self.eps * jnp.eye(len(net_ports), dtype=S.dtype)
            
            operator_M = lx.MatrixLinearOperator(M_reg)
            rhs = X_local @ S_I_All
            W = jax.vmap(
                lambda b: lx.linear_solve(operator_M, b, self.linear_solver).value,
                in_axes=1, out_axes=1
            )(rhs)
            
            # Apply the generalized Redheffer update to the full matrix
            S = S + S_All_I @ W
            
            # Since these are strictly pure internal nets, we drop all of them
            keep_local_idx = [i for i in range(len(active_ports)) if i not in local_idx]
            keep_jnp = jnp.array(keep_local_idx)
            
            # Physically reallocate the JAX array to a smaller dimension
            S = S[jnp.ix_(keep_jnp, keep_jnp)]
            
            # Update the Python tracking list
            active_ports = [active_ports[i] for i in keep_local_idx]

        # Resolve mixed nets (Global Schur on remaining contracted matrix)
        active_global_idx = np.array(active_ports)
        N_rem = len(active_ports)
        
        rem_net_map = net_map_np[active_global_idx]
        rem_z0 = z0_ports[active_global_idx]
        
        # Build connection matrix X for the remaining ports
        mask = jnp.array(rem_net_map[:, None] == rem_net_map[None, :], dtype=S.dtype)
        is_ext_net_port_np = np.isin(rem_net_map, ext_net_ids)
        mask_ext = mask * is_ext_net_port_np[:, None]
        
        y0 = 1.0 / rem_z0
        y0_sum_ext = jnp.dot(mask_ext, y0)
        y0_sum_ext = jnp.where(y0_sum_ext == 0, 1.0, y0_sum_ext) # Prevent zero division
        
        y0_sqrt_prod = jnp.sqrt(y0[:, None] * y0[None, :])
        X = mask_ext * ((2.0 * y0_sqrt_prod) / y0_sum_ext[:, None] - jnp.eye(N_rem, dtype=S.dtype))
        
        T = jnp.eye(N_rem, dtype=S.dtype) - S @ X
        
        # Map global ext_idx and remaining int_idx to the local indices of the contracted matrix
        local_ext_idx = jnp.array([active_ports.index(p) for p in ext_idx_np])
        remaining_int_idx = [p for p in active_ports if p not in ext_idx_np]
        local_int_idx = jnp.array([active_ports.index(p) for p in remaining_int_idx])
        
        if len(remaining_int_idx) > 0:
            T_A = T[jnp.ix_(local_ext_idx, local_ext_idx)]
            T_B = T[jnp.ix_(local_ext_idx, local_int_idx)]
            T_C = T[jnp.ix_(local_int_idx, local_ext_idx)]
            T_D = T[jnp.ix_(local_int_idx, local_int_idx)]
            
            X_A = X[jnp.ix_(local_ext_idx, local_ext_idx)]
            X_B = X[jnp.ix_(local_ext_idx, local_int_idx)]
            
            T_D_reg = T_D + self.eps * jnp.eye(T_D.shape[0], dtype=T_D.dtype)
            operator_D = lx.MatrixLinearOperator(T_D_reg)
            
            tmp_mat = jax.vmap(
                lambda b: lx.linear_solve(operator_D, b, self.linear_solver).value,
                in_axes=1, out_axes=1
            )(T_C)
            
            numerator = X_A - X_B @ tmp_mat
            denominator = T_A - T_B @ tmp_mat
        else:
            # Edge case: No internal ports left, fully contracted to external boundaries
            numerator = X[jnp.ix_(local_ext_idx, local_ext_idx)]
            denominator = T[jnp.ix_(local_ext_idx, local_ext_idx)]
            
        den_reg = denominator + self.eps * jnp.eye(denominator.shape[0], dtype=denominator.dtype)
        operator_den = lx.MatrixLinearOperator(den_reg.T)
        
        S_ext_trav = jax.vmap(
            lambda b: lx.linear_solve(operator_den, b, self.linear_solver).value,
            in_axes=1, out_axes=1
        )(numerator.T).T
        
        z0_ext = z0_ports[ext_idx_np]
        S_ext_power = s2s(S_ext_trav, z0_ext, 'power', 'traveling')
        
        return ScatteringResult(s=S_ext_power, z0=z0_ext)