"""
Models that transform the ports or layout of another model.
"""
import jax.numpy as jnp
from parax import field

import jax.numpy as jnp
from pmrf.core import Model, Frequency
from pmrf.rf.conversions import s2y, y2s

class Renumbered(Model, transparent=True):
    """
    A container that re-numbers the ports of a given `Model`.

    This is useful for creating complex network topologies by explicitly
    re-mapping the port indices of a sub-network.
    
    Attributes
    ----------
    model : Model
        The underlying model to renumber.
    from_ports : tuple[int]
        The original port indices that map to `to_ports`.
    to_ports : tuple[int]
        The new port indices. Can be `None`, in which case `from_ports`
        must contain exactly two ports to be swapped.
    """
    model: Model
    from_ports: tuple[int]
    to_ports: tuple[int] = None

    def __post_init__(self):
        model = self.model
        if self.to_ports is None:
            if len(self.from_ports) != 2:
                raise Exception("from_ports must have length==2 if to_ports is None")
            self.to_ports = (self.from_ports[1], self.from_ports[0])
        
        if model.primary_property == 'a' and len(self.from_ports) != 2 and len(self.to_ports) != 2:
            raise ValueError("(from_ports, to_ports) must be either (0, 1) or (1, 0) for 'a' primary networks")        
        
        if len(self.from_ports) != len(self.to_ports):
            raise ValueError("from_ports and to_ports must have the same length for Renumbered")

    def renumber(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the port renumbering to a parameter matrix.

        Parameters
        ----------
        p : jnp.ndarray
            The parameter matrix to renumber (e.g., S-parameters).

        Returns
        -------
        jnp.ndarray
            The renumbered parameter matrix.
        """
        p_new = p.copy()
        p_new = p_new.at[:, self.to_ports, :].set(p[:, self.from_ports, :])
        p_new = p_new.at[:, :, self.to_ports].set(p_new[:, :, self.from_ports])
        return p_new
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        return self.renumber(self.model.a(freq))

    def s(self, freq: Frequency) -> jnp.ndarray:
        return self.renumber(self.model.s(freq))

    def y(self, freq: Frequency) -> jnp.ndarray:
        return self.renumber(self.model.y(freq))

    def z(self, freq: Frequency) -> jnp.ndarray:
        return self.renumber(self.model.z(freq))
    
    
class Flipped(Renumbered):
    """
    A model container that flips the ports of a multi-port network.

    For a 2-port network, this is equivalent to swapping port 1 and port 2.
    For a 4-port network, ports (1,2) are swapped with (3,4), and so on.
    This is a convenient specialization of the `Renumbered` model.
    """
    to_ports: tuple[int] = field(init=False)
    from_ports: tuple[int] = field(init=False)

    def __post_init__(self):
        if self.model.nports % 2 != 0:
            raise ValueError("You can only flip multiple-of-two-port Networks")
        
        n = int(self.model.nports / 2)
        self.to_ports = tuple(range(0, 2 * n))
        self.from_ports = tuple(range(n, 2 * n)) + tuple(range(0, n))

        super().__post_init__()

        self.name = 'flipped'
        
        
class Stacked(Model, transparent=True):
    """
    A container that stacks multiple models in a block-diagonal fashion.

    This combines several `Model` objects into a single, larger model where
    the individual S-parameter matrices are placed along the diagonal of the
    combined S-parameter matrix. This represents a set of unconnected
    networks treated as a single component.

    Attributes
    ----------
    models : tuple[Model, ...]
        The models to stack.
    """
    models: tuple[Model, ...]
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        num_ports = sum(model.nports for model in self.models)

        s = jnp.zeros((freq.npoints, num_ports, num_ports), dtype=jnp.complex128)
        i = 0
        for submodel in self.models:
            s_sub = submodel.s(freq)
            n_sub = submodel.nports
            
            s = s.at[:,i:i+n_sub,i:i+n_sub].set(s_sub)
            
            i += n_sub
        return s
    

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