"""
Models that transform the ports or layout of another model.
"""
import jax.numpy as jnp

import jax.numpy as jnp
from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils import field, ArrayLike

class Renumbered(Model):
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
        
        if model.primary_domain == 'a' and len(self.from_ports) != 2 and len(self.to_ports) != 2:
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

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return self.renumber(self.model.s(freq, z0=z0))

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