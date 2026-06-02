"""pmrf/simulate/solvers/nodal.py"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from jax.experimental import sparse

from pmrf.simulate.base import (
    AbstractAdmittanceReducer, 
    AbstractMNAReducer, 
    NodalRepresentation, 
    MNARepresentation,
    AdmittanceResult
)

class GlobalNodalReducer(AbstractAdmittanceReducer):
    """
    Global Y-domain Nodal Admittance circuit solver.

    Assembles a complete Nodal Admittance Matrix (NAM) and eliminates 
    internal nodes simultaneously via a Schur complement. Highly efficient 
    for pure Y-domain networks, but requires the Modified Nodal approach (MNA) 
    if ideal components (e.g., ideal transformers) are present.
    """
    #: Numerical regularization (equivalent to adding GMIN to ground) to prevent singular matrices.
    eps: float = eqx.field(default=1e-12, static=True)
    
    #: Uses BCOO sparse assembly before converting to dense. Faster for massive networks.
    use_sparse: bool = eqx.field(default=False, static=True)
    
    #: The lineax solver to use for the global matrix inversion. Defaults to AutoLinearSolver.
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=False), static=True
    )

    def run(
        self, 
        y_flattened: jax.Array,
        topology: NodalRepresentation, 
    ) -> AdmittanceResult:
        
        N = topology.num_nodes
        
        # --- Assemble Global Matrix ---
        if self.use_sparse:
            # Trace-time filter to drop out-of-bounds ground indices
            valid = (topology.r_idx < N) & (topology.c_idx < N)
            
            # Static slice of dynamic array is natively supported by JAX
            indices = jnp.stack([topology.r_idx[valid], topology.c_idx[valid]], axis=-1)
            Y_global = sparse.BCOO((y_flattened[valid], indices), shape=(N, N)).todense()
        else:
            Y_global = jnp.zeros((N, N), dtype=y_flattened.dtype)
            Y_global = Y_global.at[topology.r_idx, topology.c_idx].add(y_flattened, mode='drop')
        
        # Apply standard GMIN regularization to the entire diagonal
        if self.eps > 0:
            Y_global += self.eps * jnp.eye(N, dtype=Y_global.dtype)
            
        # --- Sub-matrix Partitioning ---
        Y_ee = Y_global[jnp.ix_(topology.ext_idx, topology.ext_idx)]
        
        # --- Schur Complement Reduction via lineax ---
        if topology.int_idx.size > 0:
            Y_ei = Y_global[jnp.ix_(topology.ext_idx, topology.int_idx)]
            Y_ie = Y_global[jnp.ix_(topology.int_idx, topology.ext_idx)]
            Y_ii = Y_global[jnp.ix_(topology.int_idx, topology.int_idx)]
            
            operator_ii = lx.MatrixLinearOperator(Y_ii)
            
            # vmap over columns (axis=1) of Y_ie
            X = jax.vmap(
                lambda b: lx.linear_solve(operator_ii, b, self.linear_solver).value,
                in_axes=1, out_axes=1
            )(Y_ie)
            
            y_reduced = Y_ee - Y_ei @ X
        else:
            y_reduced = Y_ee
            
        return AdmittanceResult(y=y_reduced)


class GlobalMNAReducer(AbstractMNAReducer):
    """
    Global Modified Nodal Analysis (MNA) circuit solver.

    Generalizes standard Nodal Analysis to gracefully handle ideal components 
    by augmenting the Y-matrix with auxiliary variables (currents/voltages).
    Eliminates all internal nodes and auxiliary variables simultaneously to 
    yield the pure external Y-parameters of the reduced network.
    """
    #: Numerical regularization to prevent singular matrices on floating nodes/aux variables.
    eps: float = eqx.field(default=1e-12, static=True)
    
    #: Uses BCOO sparse assembly before converting to dense. Faster for massive networks.
    use_sparse: bool = eqx.field(default=False, static=True)
    
    #: The lineax solver to use for the global matrix inversion. Defaults to AutoLinearSolver.
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=False), static=True
    )

    def run(
        self, 
        y_flattened: jax.Array,
        b_flattened: jax.Array,
        c_flattened: jax.Array,
        d_flattened: jax.Array,
        topology: MNARepresentation, 
    ) -> AdmittanceResult:
        
        N = topology.num_nodes
        K = topology.num_aux

        # --- Assemble the four global sub-blocks ---
        if self.use_sparse:
            # Trace-time filters for ground nodes (N is ground for standard nodes)
            val_y = (topology.y_r_idx < N) & (topology.y_c_idx < N)
            val_b = (topology.b_r_idx < N)
            val_c = (topology.c_c_idx < N)
            # D block maps aux-to-aux, never connects to standard ground

            idx_y = jnp.stack([topology.y_r_idx[val_y], topology.y_c_idx[val_y]], axis=-1)
            Y_g = sparse.BCOO((y_flattened[val_y], idx_y), shape=(N, N)).todense()
            
            idx_b = jnp.stack([topology.b_r_idx[val_b], topology.b_c_idx[val_b]], axis=-1)
            B_g = sparse.BCOO((b_flattened[val_b], idx_b), shape=(N, K)).todense()
            
            idx_c = jnp.stack([topology.c_r_idx[val_c], topology.c_c_idx[val_c]], axis=-1)
            C_g = sparse.BCOO((c_flattened[val_c], idx_c), shape=(K, N)).todense()
            
            idx_d = jnp.stack([topology.d_r_idx, topology.d_c_idx], axis=-1)
            D_g = sparse.BCOO((d_flattened, idx_d), shape=(K, K)).todense()
        else:
            Y_g = jnp.zeros((N, N), dtype=y_flattened.dtype)
            Y_g = Y_g.at[topology.y_r_idx, topology.y_c_idx].add(y_flattened, mode='drop')
            
            B_g = jnp.zeros((N, K), dtype=b_flattened.dtype)
            B_g = B_g.at[topology.b_r_idx, topology.b_c_idx].add(b_flattened, mode='drop')
            
            C_g = jnp.zeros((K, N), dtype=c_flattened.dtype)
            C_g = C_g.at[topology.c_r_idx, topology.c_c_idx].add(c_flattened, mode='drop')
            
            D_g = jnp.zeros((K, K), dtype=d_flattened.dtype)
            D_g = D_g.at[topology.d_r_idx, topology.d_c_idx].add(d_flattened, mode='drop')

        # Snap the blocks together into the unified MNA matrix
        M_global = jnp.block([
            [Y_g, B_g],
            [C_g, D_g]
        ])
        
        # Apply standard GMIN regularization to the entire diagonal
        if self.eps > 0:
            M_global += self.eps * jnp.eye(N + K, dtype=M_global.dtype)
            
        # --- Identify all rows/cols to eliminate ---
        aux_idx = jnp.arange(N, N + K, dtype=topology.int_idx.dtype)
        full_int_idx = jnp.concatenate([topology.int_idx, aux_idx])
        
        # --- Sub-matrix Partitioning ---
        M_ee = M_global[jnp.ix_(topology.ext_idx, topology.ext_idx)]
        
        # --- Schur Complement Reduction via lineax ---
        if full_int_idx.size > 0:
            M_ei = M_global[jnp.ix_(topology.ext_idx, full_int_idx)]
            M_ie = M_global[jnp.ix_(full_int_idx, topology.ext_idx)]
            M_ii = M_global[jnp.ix_(full_int_idx, full_int_idx)]
            
            operator_ii = lx.MatrixLinearOperator(M_ii)
            
            # vmap over columns (axis=1) of M_ie
            X = jax.vmap(
                lambda b: lx.linear_solve(operator_ii, b, self.linear_solver).value,
                in_axes=1, out_axes=1
            )(M_ie)
            
            y_reduced = M_ee - M_ei @ X
        else:
            y_reduced = M_ee
            
        return AdmittanceResult(y=y_reduced)