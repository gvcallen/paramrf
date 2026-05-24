"""pmrf/simulate/kron.py"""

import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.simulate.base import AbstractAdmittanceReducer, NodalRepresentation, AdmittanceResult

class Kron(AbstractAdmittanceReducer):
    """Reduces an admittance network by applying kron reduction to internal nodes."""
    
    eps: float = eqx.field(default=1e-12, static=True)

    def run(
        self, 
        y_matrices: jax.Array,
        topology: NodalRepresentation, 
    ) -> AdmittanceResult:
        
        # 1. Assemble Global Matrix via scatter-add
        Y_global = jnp.zeros(
            (topology.num_nodes, topology.num_nodes), 
            dtype=y_matrices.dtype
        )
        Y_global = Y_global.at[topology.r_idx, topology.c_idx].add(y_matrices)
        
        if self.eps > 0:
            Y_global += self.eps * jnp.eye(topology.num_nodes, dtype=Y_global.dtype)
            
        # 2. Sub-matrix Partitioning
        Y_ee = Y_global[jnp.ix_(topology.ext_idx, topology.ext_idx)]
        
        # 3. Schur Complement
        if topology.int_idx.size > 0:
            Y_ei = Y_global[jnp.ix_(topology.ext_idx, topology.int_idx)]
            Y_ie = Y_global[jnp.ix_(topology.int_idx, topology.ext_idx)]
            Y_ii = Y_global[jnp.ix_(topology.int_idx, topology.int_idx)]
            
            X = jax.scipy.linalg.solve(Y_ii, Y_ie, assume_a="gen")
            y_reduced = Y_ee - Y_ei @ X
        else:
            y_reduced = Y_ee
            
        return AdmittanceResult(y=y_reduced)
    