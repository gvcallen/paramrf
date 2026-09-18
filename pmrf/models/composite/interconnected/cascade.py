"""
Composite models that physically connect ports of other models in series.
"""
import jax
import jax.numpy as jnp
from dataclasses import InitVar
from functools import partial
from typing import Literal

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils import field
from pmrf.types import ArrayLike
from pmrf.rf import a2s, s2a, a2mna, s2mna, MNAStamp
from pmrf.rf import cascade_scattering, cascade_abcd


HUB_Z0 = 50.0 + 0.0j

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
    
    #: Relative cutoff for singular values in scattering cascade elimination.
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

    # --- SIMULATION & CONVERSION ---

    def _solve(self, freq: Frequency, z0: ArrayLike = EVAL_Z0) -> tuple[jnp.ndarray, jnp.ndarray, str]:
        """Dispatches data prep and solving across the active vmapped mathematical method."""
        if self.flatten:
            flat = self
        else:
            flat = self.flattened()

        if flat.method == 's':
            s_blocks, z0_blocks = flat._evaluate_scattering(freq, z0)
            reduce_s = partial(cascade_scattering, eps=flat.eps)
            run_vmap = jax.vmap(reduce_s, in_axes=(0, 0))
            s_cas, z0_cas = run_vmap(s_blocks, z0_blocks)
            return s_cas, z0_cas, 's'
            
        elif flat.method == 'a':
            a_blocks = flat._evaluate_abcd(freq)
            run_vmap = jax.vmap(cascade_abcd, in_axes=(0,))
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

    def mna(self, freq: Frequency) -> MNAStamp:
        """
        Overrides the generic `Model.mna()` fallback, which prefers `.a()`
        over `.s()` whenever both are overridden.
        """
        if self.nports <= 2:
            return a2mna(self.a(freq))
        return s2mna(self.s(freq, z0=HUB_Z0), z0=HUB_Z0)
