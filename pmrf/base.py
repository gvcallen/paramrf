"""General base and core classes"""

from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx

from pmrf.frequency import Frequency

class MNAStamp(eqx.Module):
    """Represents a batched MNA stamp for a component."""
    
    #: Nodal admittance matrix. Shape: (nf, n, n)
    Y: jnp.ndarray  
    
    #: Maps auxiliary variables to node currents. Shape: (nf, n, k)
    B: jnp.ndarray  
    
    #: Maps node voltages to auxiliary constraints. Shape: (nf, k, n)
    C: jnp.ndarray  
    
    #: Auxiliary variable relationships. Shape: (nf, k, k)
    D: jnp.ndarray  
    
    @property
    def num_ports(self) -> int:
        return self.Y.shape[1]
        
    @property
    def num_aux(self) -> int:
        return self.D.shape[1]
    

class AbstractComponent(eqx.Module):
    """
    A lower-level interface for any N-port microwave component.

    This is used to provide a strict separation between the user-facing
    models in :mod:`pmrf.models` and lower-level algorithms in e.g.
    :mod:`pmrf.simulate`.
    """
    #: The number of ports the component has.
    nports: eqx.AbstractVar[int]

    @abstractmethod
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        """Evaluates the Scattering parameters."""
        raise NotImplementedError

    @abstractmethod
    def y(self, freq: Frequency) -> jnp.ndarray:
        """Evaluates the Admittance parameters."""
        raise NotImplementedError

    @abstractmethod
    def z(self, freq: Frequency) -> jnp.ndarray:
        """Evaluates the Impedance parameters."""
        raise NotImplementedError
    
    @abstractmethod
    def a(self, freq: Frequency) -> jnp.ndarray:
        """Evaluates the Impedance parameters."""
        raise NotImplementedError

    @abstractmethod
    def mna(self, freq: Frequency) -> MNAStamp:
        """Evaluates the Modified Nodal Analysis matrices."""
        raise NotImplementedError