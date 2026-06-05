"""
Composite models that physically connect ports of other models in series.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import InitVar
from typing import Literal

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils import field
from pmrf.math import nudge_diag
from pmrf.types import ArrayLike
from pmrf.rf import a2s, s2a

EVAL_Z0 = 50.0

class Cascade(Model):
    """
    Represents a cascade, or series connection, of two or more `Model` objects.

    This container connects multiple models end-to-end. The output port of
    one model is connected to the input port of the next.

    All models must have 2N-many ports. Ports N to 2*N-1 of the first model
    are connected to ports 0 to N-1 of the second, and so on.

    Any nested `Cascade` instances are automatically flattened to maintain
    a simple, linear chain of models.

    Parameters
    ----------
    cascade : tuple[Model, ...]
        The sequence of models in the cascade.
    method : {'s', 'a'}, default='s'
        The underlying mathematical domain to use for the cascade reduction.
    flatten: bool, default=True
        (experimental) Flattens the cascade into one large cascade if they contain sub-cascades.

    Examples
    --------
    Cascading models is most easily done using the `**` operator, which is
    an alias for creating a `Cascade` model.

    >>> import pmrf as prf
    >>> from pmrf.models import Resistor, Capacitor, Inductor

    # Create individual component models
    >>> res = Resistor(50)
    >>> cap = Capacitor(1e-12)
    >>> ind = Inductor(1e-9)

    # Cascade them together in a series R-L-C configuration
    # This is equivalent to Cascade(models=(res, ind, cap))
    >>> rlc_series = res ** ind ** cap

    # Define a frequency axis
    >>> freq = prf.Frequency(start=1, stop=10, npoints=101, unit='ghz')

    # Calculate the S-parameters of the cascaded network
    >>> s_params = rlc_series.s(freq)
    """
    #: The models.
    cascade: tuple[Model, ...]
    
    #: (experimental) Flatten the connections if they contain any sub-circuits
    flatten: bool = field(default=True, static=True, kw_only=True)
    
    #: The cascade reduction algorithm method.
    method: Literal['s', 'a'] = field(default='s', kw_only=True, static=True)
    
    #: Epsilon for matrix singularity nudging in scattering cascade.
    eps: float = field(default=1e-12, static=True, kw_only=True)

    def __post_init__(self):
        for model in self.cascade:
            if model.nports % 2 != 0:
                raise ValueError('All networks must be 2N-ports for Cascade')
            
    def expand(self):
        from pmrf.models import Circuit
        
        built = Circuit.from_chain(self.cascade)
        port_map = [(built, p) for p in range(self.nports)]
        return port_map, []
            
    @property
    def number_of_ports(self):
        return self.cascade[0].number_of_ports
    
    def flattened(self) -> 'Cascade':
        merged = []
        for model in self.cascade:
            # Only extend if the user has not given it a name or metadata
            if isinstance(model, Cascade) and model.name is None and model.metadata is None:
                merged.extend(model.cascade)
            else:
                merged.append(model)

        return Cascade(merged, method=self.method, eps=self.eps, flatten=False)

    # --- DATA EVALUATION ---

    def _evaluate_scattering(self, freq: Frequency, z0: ArrayLike) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluates S-parameters and stacks them into shape (Nf, N_models, P, P)."""
        S_blocks = jnp.stack([c.s(freq, z0=z0) for c in self.cascade], axis=1)
        
        Nf = S_blocks.shape[0]
        n_components = len(self.cascade)
        c_ports = self.cascade[0].nports
        
        # Broadcast port impedances to match shape
        z0_blocks = jnp.broadcast_to(jnp.asarray(z0, dtype=S_blocks.dtype), (Nf, n_components, c_ports))
        return S_blocks, z0_blocks

    def _evaluate_abcd(self, freq: Frequency) -> jnp.ndarray:
        """Evaluates ABCD-parameters and stacks them into shape (Nf, N_models, P, P)."""
        return jnp.stack([c.a(freq) for c in self.cascade], axis=1)

    # --- CASCADE ALGORITHMS (Single Frequency Point) ---

    def _cascade_two_s(self, Smat_A: jnp.ndarray, z0_A: jnp.ndarray, Smat_B: jnp.ndarray, z0_B: jnp.ndarray):
        """Mathematical routine to combine two S-parameter matrices."""
        nports = Smat_A.shape[0]
        N = nports // 2
        
        # Verify no un-renormalized impedance step exists between the stages
        mismatch_detected = jnp.any(jnp.abs(z0_A[N:] - z0_B[:N]) > 1e-6)
        Smat_A = eqx.error_if(
            Smat_A, 
            mismatch_detected, 
            "Scattering cascade requires matching reference impedances between connected ports. "
            "Renormalize stages or use a Circuit solver for arbitrary impedance steps."
        )

        z0_cas = jnp.concatenate((z0_A[:N], z0_B[N:]), axis=0)

        A11, A12 = Smat_A[:N, :N], Smat_A[:N, N:]
        A21, A22 = Smat_A[N:, :N], Smat_A[N:, N:]

        B11, B12 = Smat_B[:N, :N], Smat_B[:N, N:]
        B21, B22 = Smat_B[N:, :N], Smat_B[N:, N:]

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

    def _cascade_scattering(self, s_stacked: jnp.ndarray, port_z0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Scans the `_cascade_two_s` routine across sequential S-parameter components."""
        if s_stacked.shape[0] == 1:
            return s_stacked[0], port_z0[0]

        def scan_fn(carry, x):
            S_acc, z0_acc = carry
            S_i, z0_i = x
            S_next, z0_next = self._cascade_two_s(S_acc, z0_acc, S_i, z0_i)
            return (S_next, z0_next), None

        (S_cas, z0_cas), _ = jax.lax.scan(
            scan_fn, 
            init=(s_stacked[0], port_z0[0]), 
            xs=(s_stacked[1:], port_z0[1:])
        )
        return S_cas, z0_cas

    def _cascade_abcd(self, a_stacked: jnp.ndarray) -> jnp.ndarray:
        """Scans matrix multiplication across sequential ABCD components."""
        if a_stacked.shape[0] == 1:
            return a_stacked[0]

        def scan_fn(carry, x):
            return carry @ x, None

        a_cas, _ = jax.lax.scan(
            scan_fn, 
            init=a_stacked[0], 
            xs=a_stacked[1:]
        )
        return a_cas

    # --- SIMULATION & CONVERSION ---

    def _solve(self, freq: Frequency, z0: ArrayLike = EVAL_Z0) -> tuple[jnp.ndarray, jnp.ndarray, str]:
        """Dispatches data prep and solving across the active vmapped mathematical method."""
        if self.flatten:
            flat = self
        else:
            flat = self.flattened()

        if flat.method == 's':
            s_blocks, z0_blocks = flat._evaluate_scattering(freq, z0)
            run_vmap = jax.vmap(flat._cascade_scattering, in_axes=(0, 0))
            s_cas, z0_cas = run_vmap(s_blocks, z0_blocks)
            return s_cas, z0_cas, 's'
            
        elif flat.method == 'a':
            a_blocks = flat._evaluate_abcd(freq)
            run_vmap = jax.vmap(flat._cascade_abcd, in_axes=(0,))
            a_cas = run_vmap(a_blocks)
            return a_cas, None, 'a'
            
        else:
            raise ValueError(f"Unknown cascade method: {self.method}")

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        mat, mat_z0, domain = self._solve(freq, z0)
        if domain == 's':
            return mat
        elif domain == 'a':
            return a2s(mat, z0=z0)

    def a(self, freq: Frequency) -> jnp.ndarray:
        mat, mat_z0, domain = self._solve(freq)
        if domain == 's':
            return s2a(mat, z0=mat_z0)
        elif domain == 'a':
            return mat