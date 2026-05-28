import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.simulate.base import AbstractAdmittanceReducer, NodalRepresentation, AdmittanceResult

class KronReducer(AbstractAdmittanceReducer):
    """
    Y-domain Nodal admittance circuit solver using Kron reduction.

    This method is very fast when working with pure, Y-domain components.

    However, since Y-matrices are not defined for many ideal components, it cannot be used
    in more complex circuits, where it may produce NaNs or large numerical instabilities.
    In this case, a Modified Nodal Admittance (MNA) reduction must be performed using a
    relevant solver.

    Therefore, when using this solver, it is important to compare the output of this method with S-parameter methods
    (such as :class:`pmrf.simulate.Hallbjorner) to ensure the output is stable for the range
    of parameters in your model.
    """

    eps: float = eqx.field(default=1e-12, static=True)

    def run(
        self, 
        y_flattened: jax.Array,
        topology: NodalRepresentation, 
    ) -> AdmittanceResult:
        
        # Assemble Global Matrix via scatter-add
        Y_global = jnp.zeros(
            (topology.num_nodes, topology.num_nodes), 
            dtype=y_flattened.dtype
        )
        Y_global = Y_global.at[topology.r_idx, topology.c_idx].add(y_flattened)
        
        if self.eps > 0:
            Y_global += self.eps * jnp.eye(topology.num_nodes, dtype=Y_global.dtype)
            
        # Sub-matrix Partitioning
        Y_ee = Y_global[jnp.ix_(topology.ext_idx, topology.ext_idx)]
        
        # Schur Complement
        if topology.int_idx.size > 0:
            Y_ei = Y_global[jnp.ix_(topology.ext_idx, topology.int_idx)]
            Y_ie = Y_global[jnp.ix_(topology.int_idx, topology.ext_idx)]
            Y_ii = Y_global[jnp.ix_(topology.int_idx, topology.int_idx)]
            
            X = jax.scipy.linalg.solve(Y_ii, Y_ie, assume_a="gen")
            y_reduced = Y_ee - Y_ei @ X
        else:
            y_reduced = Y_ee
            
        return AdmittanceResult(y=y_reduced)
    