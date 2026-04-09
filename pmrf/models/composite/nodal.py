"""
Models that alter the nodal environment of a model.
"""

import jax.numpy as jnp

import jax.numpy as jnp
from pmrf.core import Model, Frequency
from pmrf.rf.conversions import s2y, y2s

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