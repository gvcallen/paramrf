"""
Models that alter the nodal environment of wrapped models.

This includes adding/removing ground, introducing coupling, etc.
"""

import jax.numpy as jnp

import jax.numpy as jnp
from pmrf.models import Model, Frequency
from pmrf.rf.conversions import s2y, y2s, s2z, z2s

class GroundLifted(Model, transparent=True):
    """
    A wrapper that converts an N-port grounded model into a 2N-port ungrounded model.

    The inner component's signal paths map to the even ports (0, 2, 4, ..., 2N-2).
    The inner component's ground is lifted and connected to the odd ports (1, 3, 5, ..., 2N-1),
    forming an isolated common-return star node.
    """
    #: The inner N-port model to be wrapped.
    model: Model

    def s(self, freq: Frequency) -> jnp.ndarray:
        # 1. Tentatively evaluate the inner model to dynamically determine N.
        # Using a scalar 50.0 ensures it evaluates safely without shape mismatch.
        s_inner_test = self.model.with_fields(z0=50.0).s(freq)
        n = s_inner_test.shape[-1]

        # 2. Parse reference impedances dynamically for N signal and N return ports
        if jnp.isscalar(self.z0):
            inner_z0 = self.z0
            z_ret = jnp.full((freq.npoints, n), self.z0, dtype=jnp.complex128)
        else:
            # Check if z0 provided enough ports (2N). If not, apply fallback logic.
            if self.z0.shape[-1] >= 2 * n:
                inner_z0 = self.z0[..., 0:2*n:2]  # Even indices (Signals)
                z_ret = self.z0[..., 1:2*n:2]     # Odd indices (Returns)
            else:
                # Fallback: Treat z0[0] as the impedance for all signals, z0[1] for all returns
                inner_z0 = jnp.repeat(self.z0[..., 0:1], n, axis=-1)
                z_ret = jnp.repeat(
                    self.z0[..., 1:2] if self.z0.shape[-1] > 1 else self.z0[..., 0:1], 
                    n, axis=-1
                )

        # 3. Evaluate the exact Signal Path S-matrix (N x N)
        s_inner = self.model.with_fields(z0=inner_z0).s(freq)

        # 4. Compute the exact Return Path S-matrix (N x N parallel star node)
        y_ret = 1.0 / z_ret
        r_ret = z_ret.real
        y_tot = jnp.sum(y_ret, axis=-1, keepdims=True)  # Shape: (..., 1)

        # Calculate the S_ij numerator terms: sqrt(Re(Z)) * Y
        term = jnp.sqrt(r_ret) * y_ret  # Shape: (..., n)

        # Use einsum for the outer product term_i * term_j across the batch
        s_ret = 2.0 * jnp.einsum('...i,...j->...ij', term, term)
        
        # Divide by Y_tot (expanding dims to broadcast across the NxN matrices)
        s_ret = s_ret / y_tot[..., jnp.newaxis]

        # Apply the diagonal correction: S_ii = (above) - conj(Z_i)/Z_i
        diag_correction = jnp.conj(z_ret) / z_ret
        i = jnp.arange(n)
        s_ret = s_ret.at[..., i, i].add(-diag_correction)

        # 5. Assemble the interlaced 2N x 2N matrix
        s_out = jnp.zeros(s_inner.shape[:-2] + (2 * n, 2 * n), dtype=jnp.complex128)
        
        # JAX array slicing maps the Signal matrix to even ports, Return matrix to odd ports
        s_out = s_out.at[..., 0::2, 0::2].set(s_inner)
        s_out = s_out.at[..., 1::2, 1::2].set(s_ret)

        return s_out
    

class GroundExposed(Model, transparent=True):
    """
    A wrapper that converts an N-port grounded model into an (N+1)-port model
    by exposing the global ground as a single, accessible terminal.

    The original signal ports remain at indices 0 to N-1.
    The new exposed ground port is at index N.
    """
    #: The inner N-port model to be wrapped.
    model: Model

    def s(self, freq: Frequency) -> jnp.ndarray:
        # 1. Parse reference impedances
        if jnp.isscalar(self.z0):
            z0_inner = self.z0
            z0_new_port = self.z0
        else:
            z0_inner = self.z0[..., :-1]
            z0_new_port = self.z0[..., -1:]

        # 2. Get inner S-parameters and convert to Y-parameters
        s_inner = self.model.with_fields(z0=z0_inner).s(freq)
        y_inner = s2y(s_inner, z0=z0_inner)

        # 3. Apply the Indefinite Admittance Matrix (IAM) transformation
        # Sum across rows and columns
        col_sums = jnp.sum(y_inner, axis=-1, keepdims=True)
        row_sums = jnp.sum(y_inner, axis=-2, keepdims=True)
        total_sum = jnp.sum(y_inner, axis=(-2, -1), keepdims=True)

        # 4. Assemble the (N+1) x (N+1) Y-matrix
        # Top block: [y_inner, -col_sums]
        top_block = jnp.concatenate([y_inner, -col_sums], axis=-1)
        
        # Bottom block: [-row_sums, total_sum]
        bottom_block = jnp.concatenate([-row_sums, total_sum], axis=-1)

        # Combine top and bottom
        y_exposed = jnp.concatenate([top_block, bottom_block], axis=-2)

        # 5. Convert back to S-parameters
        return y2s(y_exposed, z0=self.z0)
    

class Shunt(Model, transparent=True):
    r"""
    Represents a 1-port network connected in parallel (shunt) across a 2-port line.

    This maps the reflection coefficient ($\Gamma$ or $S_{11}$) of a 1-port 
    model into a 2-port transmission matrix. 

    Attributes
    ----------
    model : Model
        The 1-port model to be connected in shunt.
    """
    model: Model
    
    def __post_init__(self):
        if self.model.nports != 1:
            raise ValueError(f"Shunt requires a 1-port model. Received a {self.model.nports}-port model.")

    def s(self, freq: Frequency) -> jnp.ndarray:
        # Get the 1-port S-parameters. Shape: (npoints, 1, 1)
        # Note: This assumes self.model.z0 == self.z0. If your library allows 
        # mixed reference impedances, you will need to renormalize s_1p first.
        s_1p = self.model.s(freq)
        
        # Extract the reflection coefficient array
        gamma = s_1p[:, 0, 0]
        
        # Calculate 2-port S-parameters directly from 1-port Gamma
        # This avoids divide-by-zero errors for ideal opens/shorts
        denom = gamma + 3.0
        s11 = (gamma - 1.0) / denom
        s21 = 2.0 * (1.0 + gamma) / denom
        
        # Construct the (npoints, 2, 2) S-parameter array
        S_shunt = jnp.array([
            [s11, s21],
            [s21, s11],
        ]).transpose(2, 0, 1)
        
        return S_shunt
    
    
class CoupledOnePorts(Model, transparent=True):
    """
    Wraps N 1-port models (e.g. inductors) and couples them via a given K-matrix.
    
    This creates an N-port model where the off-diagonal interactions are defined
    by the mutual admittance: Y_ij = k_ij * sqrt(Y_ii * Y_jj).
    """
    #: The sequence of 1-port models to couple.
    models: list[Model]
    #: The NxN coupling coefficient matrix.
    k_matrix: jnp.ndarray 

    def __post_init__(self):
        # Validate that all inputted models are 1-port
        for i, m in enumerate(self.models):
            if m.nports != 1:
                raise ValueError(f"CoupledOnePorts requires 1-port models. Model {i} has {m.nports} ports.")
        
        # Validate k_matrix dimensions
        n = len(self.models)
        if self.k_matrix.shape != (n, n):
            raise ValueError(f"k_matrix must be shape ({n}, {n}), got {self.k_matrix.shape}")

    def s(self, freq: Frequency) -> jnp.ndarray:
        n = len(self.models)
        
        # 1. Evaluate all 1-port models and convert S -> Y.
        y_diags = []
        for m in self.models:
            # Evaluate the 1-port model
            s_i = m.with_fields(z0=self.z0).s(freq)
            
            # Convert to Y-parameters (Shape will be ..., 1, 1)
            y_i = s2y(s_i, z0=self.z0)
            
            # Extract the scalar admittance value per frequency point
            y_diags.append(y_i[..., 0, 0]) 
            
        # Stack into a single array of shape: (..., n)
        y_diag = jnp.stack(y_diags, axis=-1)
        
        # 2. Calculate the mutual admittance terms using the outer product.
        # Y_ij = k_ij * sqrt(Y_ii * Y_jj)
        
        # Expand dims to create an NxN outer product matrix: (..., n, n)
        y_outer = y_diag[..., jnp.newaxis] * y_diag[..., jnp.newaxis, :]
        
        # Multiply by the k_matrix to apply the coupling coefficients
        y_coupled = self.k_matrix * jnp.sqrt(y_outer)
        
        # 3. Restore the exact original self-admittances on the diagonal.
        i = jnp.arange(n)
        y_coupled = y_coupled.at[..., i, i].set(y_diag)
        
        # 4. Convert the full NxN coupled Y-matrix directly back to S-parameters
        return y2s(y_coupled, z0=self.z0)
    
class CoupledTwoPorts(Model, transparent=True):
    """
    Wraps N 2-port models (e.g., Inductors) and couples them via a given K-matrix.
    
    Returns a 2N-port model where Model 1 occupies ports (0, 1), 
    Model 2 occupies ports (2, 3), and so on.
    """
    #: The sequence of 2-port series models to couple.
    models: list[Model]
    #: The NxN coupling coefficient matrix (k).
    k_matrix: jnp.ndarray 

    def __post_init__(self):
        # Validate that all inputted models are 2-port
        for i, m in enumerate(self.models):
            if m.nports != 2:
                raise ValueError(f"CoupledSeriesElements requires 2-port models. Model {i} has {m.nports} ports.")
        
        n = len(self.models)
        if self.k_matrix.shape != (n, n):
            raise ValueError(f"k_matrix must be shape ({n}, {n}), got {self.k_matrix.shape}")

    def s(self, freq: Frequency) -> jnp.ndarray:
        n = len(self.models)
        
        # 1. Extract the series branch impedance (Zs) for each 2-port model
        z_branch_list = []
        for m in self.models:
            # Evaluate using a safe fixed z0 to extract the Y-parameters cleanly
            s_i = m.with_fields(z0=50.0).s(freq)
            y_i = s2y(s_i, z0=50.0)
            
            # For a series element between port 0 and 1, Y_series = -Y_12
            y_series = -y_i[..., 0, 1]
            
            # Convert to series impedance
            z_series = 1.0 / y_series
            z_branch_list.append(z_series)
            
        # Stack into an array of shape: (..., n)
        z_branch = jnp.stack(z_branch_list, axis=-1)
        
        # 2. Apply Coupling Matrix to create the NxN Z_branch matrix
        # Z_ij = k_ij * sqrt(Z_i * Z_j)
        z_outer = z_branch[..., jnp.newaxis] * z_branch[..., jnp.newaxis, :]
        z_b_matrix = self.k_matrix * jnp.sqrt(z_outer)
        
        # Restore exact diagonals
        i = jnp.arange(n)
        z_b_matrix = z_b_matrix.at[..., i, i].set(z_branch)
        
        # 3. Invert the NxN branch impedance matrix to get branch admittance
        y_b_matrix = jnp.linalg.inv(z_b_matrix)
        
        # 4. Construct the 2N x N Incidence Matrix (A)
        # Port ordering: 0=In1, 1=Out1, 2=In2, 3=Out2...
        A = jnp.zeros((2 * n, n), dtype=jnp.float64)
        A = A.at[0::2, :].set(jnp.eye(n))  # Even ports are Inputs (+1)
        A = A.at[1::2, :].set(-jnp.eye(n)) # Odd ports are Outputs (-1)
        
        # 5. Compute the 2Nx2N nodal Y-matrix: Y_nodal = A * Y_b * A^T
        # Using einsum efficiently broadcasts the matrix multiplication over the frequency axis
        y_nodal = jnp.einsum('pi,...ij,qj->...pq', A, y_b_matrix, A)
        
        # 6. Convert back to S-parameters using the user's actual requested Z0
        return y2s(y_nodal, z0=self.z0)