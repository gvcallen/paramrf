import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.simulate.base import (
    AbstractModifiedAdmittanceReducer, 
    ModifiedNodalRepresentation,
    AdmittanceResult
)

class ModifiedKronReducer(AbstractModifiedAdmittanceReducer):
    """
    Modified Nodal Analysis (MNA) circuit solver using Kron reduction.

    This solver generalizes standard Nodal Analysis to gracefully handle 
    ideal components (like ideal transformers, lossless lines, and perfect 
    source converters) that do not possess a Y-matrix. 
    
    It constructs the full MNA system by assembling Y, B, C, and D block 
    matrices, and then uses a Schur complement to simultaneously eliminate 
    all internal nodes and all auxiliary variables, yielding the pure 
    external Y-parameters of the reduced network.
    """
    eps: float = eqx.field(default=1e-12, static=True)

    def run(
        self, 
        y_flattened: jax.Array,
        b_flattened: jax.Array,
        c_flattened: jax.Array,
        d_flattened: jax.Array,
        topology: ModifiedNodalRepresentation, 
    ) -> AdmittanceResult:
        
        N = topology.num_nodes
        K = topology.num_aux

        # Assemble the four global sub-blocks via scatter-add
        Y_g = jnp.zeros((N, N), dtype=y_flattened.dtype)
        Y_g = Y_g.at[topology.y_r_idx, topology.y_c_idx].add(y_flattened)
        
        B_g = jnp.zeros((N, K), dtype=b_flattened.dtype)
        B_g = B_g.at[topology.b_r_idx, topology.b_c_idx].add(b_flattened)
        
        C_g = jnp.zeros((K, N), dtype=c_flattened.dtype)
        C_g = C_g.at[topology.c_r_idx, topology.c_c_idx].add(c_flattened)
        
        D_g = jnp.zeros((K, K), dtype=d_flattened.dtype)
        D_g = D_g.at[topology.d_r_idx, topology.d_c_idx].add(d_flattened)

        # Snap the blocks together into the unified MNA matrix
        # If K=0 (pure nodal), JAX handles the zero-dimension arrays automatically
        # and just returns Y_g.
        M_global = jnp.block([
            [Y_g, B_g],
            [C_g, D_g]
        ])
        
        # Apply numerical regularization to the diagonal
        if self.eps > 0:
            M_global += self.eps * jnp.eye(N + K, dtype=M_global.dtype)
            
        # Identify all rows/cols to eliminate (Internal Nodes + ALL Aux Variables)
        # Aux variables are appended sequentially after the N standard nodes.
        aux_idx = jnp.arange(N, N + K, dtype=topology.int_idx.dtype)
        full_int_idx = jnp.concatenate([topology.int_idx, aux_idx])
        
        # Sub-matrix Partitioning
        M_ee = M_global[jnp.ix_(topology.ext_idx, topology.ext_idx)]
        
        # Schur Complement Reduction
        if full_int_idx.size > 0:
            M_ei = M_global[jnp.ix_(topology.ext_idx, full_int_idx)]
            M_ie = M_global[jnp.ix_(full_int_idx, topology.ext_idx)]
            M_ii = M_global[jnp.ix_(full_int_idx, full_int_idx)]
            
            X = jax.scipy.linalg.solve(M_ii, M_ie, assume_a="gen")
            y_reduced = M_ee - M_ei @ X
        else:
            y_reduced = M_ee
            
        return AdmittanceResult(y=y_reduced)