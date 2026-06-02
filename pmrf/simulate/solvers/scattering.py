"""pmrf/simulate/solvers/scattering.py"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import lineax as lx
from jax.experimental import sparse

from pmrf.simulate.base import (
    AbstractScatteringCascader, 
    AbstractScatteringReducer, 
    PortRepresentation, 
    ScatteringResult, 
    AbstractScatteringTerminator
)
from pmrf.rf.conversions import s2s
from pmrf.math import nudge_diag


class GlobalScatteringReducer(AbstractScatteringReducer):
    """
    Global S-parameter reduction solver.
    
    Assembles the full system into a single matrix and solves for the external 
    ports simultaneously. Best suited for arbitrary meshes and, when paired with 
    a sparse solver, massive networks (e.g., >10,000 nodes).
    """
    #: Numerical regularization to prevent singular matrix division during lossless resonance.
    eps: float = eqx.field(default=1e-12, static=True)
    
    #: Uses BCOO sparse assembly for the connection matrix to save memory on massive meshes.
    use_sparse: bool = eqx.field(default=False, static=True)
    
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
        if self.use_sparse:
            # Trace-time index calculation
            unique_nets, net_inv = np.unique(net_map_np, return_inverse=True)
            r_idx_list, c_idx_list = [], []
            for net_id in unique_nets:
                ports = np.where(net_map_np == net_id)[0]
                r_grid, c_grid = np.meshgrid(ports, ports, indexing='ij')
                r_idx_list.append(r_grid.flatten())
                c_idx_list.append(c_grid.flatten())
                
            r_idx = jnp.array(np.concatenate(r_idx_list))
            c_idx = jnp.array(np.concatenate(c_idx_list))
            
            # JIT-time sparse assembly
            y0 = 1.0 / z0_ports
            y0_sum_per_net = jax.ops.segment_sum(y0, jnp.array(net_inv), num_segments=len(unique_nets))
            y0_sum_flat = y0_sum_per_net[jnp.array(net_inv)[r_idx]]
            
            val = (2.0 * jnp.sqrt(y0[r_idx] * y0[c_idx])) / y0_sum_flat
            val = jnp.where(r_idx == c_idx, val - 1.0, val)
            
            indices = jnp.stack([r_idx, c_idx], axis=-1)
            X = sparse.BCOO((val, indices), shape=(N, N)).todense()
        else:
            # Dense broadcasting
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
        
        # vmap required to solve 2D matrix systems (AX = B) natively in lineax
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


class HierarchicalScatteringReducer(AbstractScatteringReducer):
    """
    Hierarchical Tree (Block Diakoptics) solver for S-parameters.
    
    Eliminates internal nets in chunked multi-port clusters. Minimizes sequential 
    JAX unroll depth while keeping the maximum matrix inversion size bounded.
    """
    max_pairs_per_stage: int = eqx.field(default=4, static=True)
    eps: float = eqx.field(default=1e-12, static=True)

    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        S_curr, z0_curr, active_ext_idx, active_int_idx = self._inject_virtual_probes(
            s_block_diagonal, z0_ports, topology
        )
        
        pairs = self._extract_pairs(active_int_idx, topology.port_to_net_map)
        
        if pairs:
            k_idx = jnp.array([p[0] for p in pairs])
            l_idx = jnp.array([p[1] for p in pairs])
            z0_diff = jnp.abs(z0_curr[k_idx] - z0_curr[l_idx])
            mismatch_detected = jnp.any(z0_diff > 1e-6)
            
            S_curr = eqx.error_if(
                S_curr,
                mismatch_detected,
                "HierarchicalScatteringReducer requires connected ports to have identical reference impedances."
            )

        stages = [
            pairs[i : i + self.max_pairs_per_stage] 
            for i in range(0, len(pairs), self.max_pairs_per_stage)
        ]
        
        active_global_idx = list(range(S_curr.shape[0]))
        
        for stage_pairs in stages:
            elim_ports_global = [p for pair in stage_pairs for p in pair]
            
            elim_rel_idx = [active_global_idx.index(p) for p in elim_ports_global]
            keep_rel_idx = [i for i in range(len(active_global_idx)) if i not in elim_rel_idx]
            
            S_curr = self._multi_port_schur(
                S_curr, jnp.array(keep_rel_idx), jnp.array(elim_rel_idx)
            )
            active_global_idx = [active_global_idx[i] for i in keep_rel_idx]

        final_ext_rel_idx = [active_global_idx.index(p) for p in active_ext_idx]
        
        S_ext = S_curr[jnp.ix_(final_ext_rel_idx, final_ext_rel_idx)]
        z0_ext = z0_curr[active_ext_idx]
        
        return ScatteringResult(s=S_ext, z0=z0_ext)

    def _multi_port_schur(self, S: jax.Array, keep_idx: jax.Array, elim_idx: jax.Array) -> jax.Array:
        S_ee = S[jnp.ix_(keep_idx, keep_idx)]
        S_ei = S[jnp.ix_(keep_idx, elim_idx)]
        S_ie = S[jnp.ix_(elim_idx, keep_idx)]
        S_ii = S[jnp.ix_(elim_idx, elim_idx)]
        
        K = elim_idx.shape[0]
        
        row = jnp.arange(K)
        col = row + jnp.where(row % 2 == 0, 1, -1)
        Gamma = jnp.zeros((K, K), dtype=S.dtype).at[row, col].set(1.0)
        
        M = jnp.eye(K, dtype=S.dtype) - S_ii @ Gamma
        if self.eps > 0:
            M += self.eps * jnp.eye(K, dtype=S.dtype)
            
        X = jax.scipy.linalg.solve(M, S_ie, assume_a="gen")
        S_new = S_ee + S_ei @ Gamma @ X
        
        return S_new

    @staticmethod
    def _inject_virtual_probes(
        s_matrix: jax.Array, z0: jax.Array, topology: PortRepresentation
    ) -> Tuple[jax.Array, jax.Array, np.ndarray, np.ndarray]:
        ext_idx_np = np.array(topology.ext_idx)
        net_map_np = np.array(topology.port_to_net_map)
        
        net_counts = np.bincount(net_map_np)
        ext_net_counts = net_counts[net_map_np[ext_idx_np]]
        
        dangling_mask = ext_net_counts == 1
        num_dangling = np.sum(dangling_mask)
        
        if num_dangling == 0:
            return s_matrix, z0, ext_idx_np, np.array(topology.int_idx)
            
        N_orig = s_matrix.shape[-1]
        
        pad_width = ((0, num_dangling), (0, num_dangling))
        s_padded = jnp.pad(s_matrix, pad_width, mode='constant')
        
        dangling_ext_indices = ext_idx_np[dangling_mask]
        z0_padded = jnp.concatenate([z0, z0[dangling_ext_indices]])
        
        new_port_indices = np.arange(N_orig, N_orig + num_dangling)
        ext_idx_np[dangling_mask] = new_port_indices
        int_idx_np = np.concatenate([topology.int_idx, dangling_ext_indices])
        
        return s_padded, z0_padded, ext_idx_np, int_idx_np

    @staticmethod
    def _extract_pairs(int_idx: np.ndarray, port_to_net_map: np.ndarray) -> List[Tuple[int, int]]:
        int_ports = int_idx.tolist()
        net_map = np.array(port_to_net_map)
        
        pairs = []
        visited = set()
        
        for p in int_ports:
            if p in visited:
                continue
                
            net_id = net_map[p]
            connected_ports = [x for x in int_ports if net_map[x] == net_id]
            
            if len(connected_ports) != 2:
                raise ValueError(
                    f"HierarchicalScatteringReducer requires pairs of ports. "
                    f"Net {net_id} has {len(connected_ports)} ports connected."
                )
                
            pairs.append(tuple(connected_ports))
            visited.update(connected_ports)
            
        return pairs


class SequentialScatteringReducer(AbstractScatteringReducer):
    """
    Sequential net-elimination solver for S-parameters.
    
    Eliminates entire nets sequentially. Ideal for smaller arbitrary networks. 
    Natively supports multi-port (T-junction) nets and impedance steps via 
    internal conversion to traveling waves.
    """
    eps: float = eqx.field(default=1e-12, static=True)
    
    def run(
        self, 
        s_block_diagonal: jax.Array, 
        z0_ports: jax.Array, 
        topology: PortRepresentation, 
    ) -> ScatteringResult:
        
        S_trav = s2s(s_block_diagonal, z0_ports, 'traveling', 'power')
        nets_to_eliminate = self._extract_nets(topology)
        
        S_current = S_trav
        z0_current = z0_ports
        active_indices = list(range(topology.num_ports))
        
        for net_global_ports in nets_to_eliminate:
            net_curr_idx = [active_indices.index(p) for p in net_global_ports]
            S_current, z0_current = self._net_eliminate(S_current, z0_current, net_curr_idx)
            active_indices = [p for p in active_indices if p not in net_global_ports]

        S_ext_power = s2s(S_current, z0_current, 'power', 'traveling')
        
        return ScatteringResult(s=S_ext_power, z0=z0_current)

    @staticmethod
    def _extract_nets(topology: PortRepresentation) -> list[list[int]]:
        int_ports = topology.int_idx.tolist()
        net_map = np.array(topology.port_to_net_map)
        
        nets = {}
        for p in int_ports:
            net_id = net_map[p]
            if net_id not in nets:
                nets[net_id] = []
            nets[net_id].append(p)
            
        return list(nets.values())

    def _net_eliminate(self, S: jax.Array, z0: jax.Array, k_idx: list[int]) -> tuple[jax.Array, jax.Array]:
        K = jnp.array(k_idx)
        N = S.shape[0]
        
        E = jnp.array([i for i in range(N) if i not in k_idx])
        
        y0 = 1.0 / z0[K]
        y0_sum = jnp.sum(y0)
        y0_sqrt = jnp.sqrt(y0)
        
        Gamma = (2.0 / y0_sum) * jnp.outer(y0_sqrt, y0_sqrt) - jnp.eye(len(K))
        
        S_KK = S[jnp.ix_(K, K)]
        S_EE = S[jnp.ix_(E, E)]
        S_KE = S[jnp.ix_(K, E)]
        S_EK = S[jnp.ix_(E, K)]
        
        A = jnp.eye(len(K), dtype=S.dtype) - Gamma @ S_KK
        A_reg = A + self.eps * jnp.eye(len(K), dtype=S.dtype)
        
        M = jax.scipy.linalg.solve(A_reg, Gamma, assume_a="gen")
        
        S_reduced = S_EE + S_EK @ M @ S_KE
        z0_reduced = z0[E]
        
        return S_reduced, z0_reduced


class SequentialScatteringCascader(AbstractScatteringCascader):
    """
    Strict left-to-right 1D S-parameter cascader.
    
    Executes a memory-efficient `lax.scan` over sequential components.
    Provides unmatched speed for simple chains (e.g., sliced transmission lines).
    """
    eps: float = eqx.field(default=1e-12, static=True)
    
    def run(
        self, 
        s_stacked: jnp.ndarray,
        port_z0: jnp.ndarray,
    ) -> ScatteringResult:
        
        if s_stacked.shape[0] == 1:
            return ScatteringResult(s=s_stacked[0], z0=port_z0[0])

        def scan_fn(carry, x):
            S_acc, z0_acc = carry
            S_i, z0_i = x
            S_next, z0_next = self.cascade_two(S_acc, z0_acc, S_i, z0_i)
            return (S_next, z0_next), None

        (S_cas, z0_cas), _ = jax.lax.scan(
            scan_fn, 
            init=(s_stacked[0], port_z0[0]), 
            xs=(s_stacked[1:], port_z0[1:])
        )
        
        return ScatteringResult(s=S_cas, z0=z0_cas)
        
    def cascade_two(
        self,
        Smat_A: jnp.ndarray,
        z0_A: jnp.ndarray,
        Smat_B: jnp.ndarray,
        z0_B: jnp.ndarray,
    ):
        nports = Smat_A.shape[0]
        N = nports // 2

        z0_cas = jnp.concatenate((z0_A[:N], z0_B[N:]), axis=0)

        A11 = Smat_A[:N, :N]
        A12 = Smat_A[:N, N:]
        A21 = Smat_A[N:, :N]
        A22 = Smat_A[N:, N:]

        B11 = Smat_B[:N, :N]
        B12 = Smat_B[:N, N:]
        B21 = Smat_B[N:, :N]
        B22 = Smat_B[N:, N:]

        I = jnp.eye(N, dtype=Smat_A.dtype)

        M = nudge_diag(I - B11 @ A22, eps=self.eps)
        N_mat = nudge_diag(I - A22 @ B11, eps=self.eps)
        
        X = jnp.linalg.solve(M, I)
        Y = jnp.linalg.solve(N_mat, I)

        S11 = A11 + A12 @ X @ B11 @ A21
        S12 = A12 @ X @ B12
        S21 = B21 @ Y @ A21
        S22 = B22 + B21 @ Y @ A22 @ B12

        top = jnp.concatenate((S11, S12), axis=1)
        bottom = jnp.concatenate((S21, S22), axis=1)
        S_cas = jnp.concatenate((top, bottom), axis=0)

        return S_cas, z0_cas
    

class ScatteringTerminator(AbstractScatteringTerminator):
    """
    Exact boundary condition substitution for S-parameters.
    
    Mathematically collapses an active network by terminating a subset 
    of its ports with a known load S-matrix.
    """
    def run(
        self, 
        s_from: jnp.ndarray,
        z0_from: jnp.ndarray,
        s_into: jnp.ndarray,
        z0_into: jnp.ndarray,
    ) -> ScatteringResult:
        
        N = s_into.shape[0]
        
        S11 = s_from[:N, :N]
        S12 = s_from[:N, N:]
        S21 = s_from[N:, :N]
        S22 = s_from[N:, N:]
        
        z0_out = z0_from[N:]
        
        def apply_renorm(operand):
            S_L, z_old, z_new = operand
            
            g = (z_new - z_old) / (z_new + jnp.conj(z_old))
            G = jnp.diag(g)
            I = jnp.eye(N, dtype=S_L.dtype)
            
            I_minus_G = I - G
            
            X = jnp.linalg.solve(I_minus_G, S_L - G)          
            Z = jnp.linalg.solve(I - G @ S_L, I_minus_G)      
            
            return X @ Z

        def skip_renorm(operand):
            S_L, _, _ = operand
            return S_L

        needs_renorm = jnp.logical_not(jnp.allclose(z0_out, z0_into))
        
        S_L_matched = jax.lax.cond(
            needs_renorm,
            apply_renorm,
            skip_renorm,
            (s_into, z0_into, z0_out)
        )

        I = jnp.eye(N, dtype=s_from.dtype)
        diff = I - S22 @ S_L_matched
        X = jnp.linalg.solve(diff, S21)
        
        S_term = S11 + S12 @ S_L_matched @ X
        z0_term = z0_from[:N]
        
        return ScatteringResult(s=S_term, z0=z0_term)