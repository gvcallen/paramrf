"""pmrf/simulate/solvers/nodal.py"""

import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
import lineax as lx

from pmrf.models.composite.interconnected.circuit.base import (
    AbstractAdmittanceCircuitSolver,
    AbstractMNACircuitSolver,
    NodalRepresentation, 
    MNARepresentation,
    AdmittanceResult,
    ScatteringResult,
)
from pmrf.rf import MNAStamp, mna2s

class GlobalNodalCircuitSolver(AbstractAdmittanceCircuitSolver):
    """
    Global Y-domain Nodal Admittance circuit solver.

    Assembles a complete Nodal Admittance Matrix (NAM) and eliminates 
    internal nodes simultaneously via a Schur complement. Highly efficient 
    for pure Y-domain networks, but requires the Modified Nodal approach (MNA) 
    if ideal components (e.g., ideal transformers) are present.

    It returns Y, so it cannot represent a zero impedance: a component whose
    ``y()`` is not finite, or ports shorted together, give a non-finite or
    regularised result. Use :class:`GlobalMNACircuitSolver` for those (ADR-0007).
    """
    #: Numerical regularization (equivalent to adding GMIN to ground) to prevent singular matrices.
    eps: float = eqx.field(default=1e-12, static=True)
    
    #: The lineax solver to use for the global matrix inversion. Defaults to AutoLinearSolver.
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=None), static=True
    )

    def run(
        self, 
        y_flattened: jax.Array,
        topology: NodalRepresentation, 
    ) -> AdmittanceResult:
        
        N = topology.num_nodes
        
        Y_global = jnp.zeros((N, N), dtype=y_flattened.dtype)
        Y_global = Y_global.at[topology.r_idx, topology.c_idx].add(y_flattened, mode='drop')
        
        # Apply standard GMIN regularization to the entire diagonal
        if self.eps > 0:
            Y_global += self.eps * jnp.eye(N, dtype=Y_global.dtype)
            
        # Sub-matrix Partitioning
        Y_ee = Y_global[jnp.ix_(topology.ext_idx, topology.ext_idx)]
        
        # Schur Complement Reduction
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


class GlobalMNACircuitSolver(AbstractMNACircuitSolver):
    r"""
    Global Modified Nodal Analysis (MNA) circuit solver.

    Generalizes standard Nodal Analysis to gracefully handle ideal components 
    by augmenting the Y-matrix with auxiliary variables (currents/voltages).
    Loads each external port with the probe reference impedance and solves the
    whole system at once, returning S at that reference (ADR-0007). Ports shorted
    together have no finite Y, but the loaded system stays non-singular, so their
    S and its gradient are exact.

    **Mathematical Formulation**

    The global system is regularised physically, so it is never singular: a
    conductance $G_{min}$ = ``eps`` from every node to ground, and a series
    resistance of ``eps`` ohms in every auxiliary branch,

    $$\begin{bmatrix} Y + G_{min} I & B \\ C & D - \epsilon I \end{bmatrix}.$$

    Internal nodes join the auxiliary variables, and :func:`pmrf.rf.mna2s` gives S
    from the resulting stamp at the external nodes.

    References
    ----------
    C.-W. Ho, A. E. Ruehli and P. A. Brennan, "The modified nodal approach to network
    analysis," IEEE Trans. Circuits Syst., vol. 22, no. 6, pp. 504-509, 1975.
    """
    #: GMIN to ground on every node, and the series resistance (ohms) of every auxiliary branch.
    eps: float = eqx.field(default=1e-12, static=True)
    
    #: The lineax solver for the loaded MNA system. Defaults to LU: the system is never singular.
    linear_solver: lx.AbstractLinearSolver = eqx.field(
        default=lx.AutoLinearSolver(well_posed=True), static=True
    )

    def run(
        self, 
        y_flattened: jax.Array,
        b_flattened: jax.Array,
        c_flattened: jax.Array,
        d_flattened: jax.Array,
        z0: jax.Array,
        topology: MNARepresentation, 
    ) -> ScatteringResult:
        
        N = topology.num_nodes
        K = topology.num_aux

        Y_g = jnp.zeros((N, N), dtype=y_flattened.dtype)
        Y_g = Y_g.at[topology.y_r_idx, topology.y_c_idx].add(y_flattened, mode='drop')
        
        B_g = jnp.zeros((N, K), dtype=b_flattened.dtype)
        B_g = B_g.at[topology.b_r_idx, topology.b_c_idx].add(b_flattened, mode='drop')
        
        C_g = jnp.zeros((K, N), dtype=c_flattened.dtype)
        C_g = C_g.at[topology.c_r_idx, topology.c_c_idx].add(c_flattened, mode='drop')
        
        D_g = jnp.zeros((K, K), dtype=d_flattened.dtype)
        D_g = D_g.at[topology.d_r_idx, topology.d_c_idx].add(d_flattened, mode='drop')

        if self.eps > 0:
            Y_g += self.eps * jnp.eye(N, dtype=Y_g.dtype)
            D_g -= self.eps * jnp.eye(K, dtype=D_g.dtype)

        M_global = jnp.block([
            [Y_g, B_g],
            [C_g, D_g]
        ])

        # Internal nodes are eliminated alongside the auxiliary variables.
        aux_idx = np.arange(N, N + K, dtype=int)
        full_int_idx = np.concatenate([topology.int_idx, aux_idx]).astype(int)
        ext_idx = topology.ext_idx

        stamp = MNAStamp(
            Y=M_global[np.ix_(ext_idx, ext_idx)],
            B=M_global[np.ix_(ext_idx, full_int_idx)],
            C=M_global[np.ix_(full_int_idx, ext_idx)],
            D=M_global[np.ix_(full_int_idx, full_int_idx)],
        )
        s = mna2s(stamp, z0, linear_solver=self.linear_solver)
        return ScatteringResult(s=s, z0=z0)
