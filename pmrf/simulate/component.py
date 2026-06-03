"""Component base class for simulation"""

from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx

from pmrf.frequency import Frequency


class MNAStamp(eqx.Module):
    """Represents an MNA stamp for a component."""
    
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

    @property
    @abstractmethod
    def nports(self) -> int:
        """The number of ports the component has."""
        raise NotImplementedError

    @abstractmethod
    def s(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        """Evaluates the Scattering parameters."""
        raise NotImplementedError

    @abstractmethod
    def y(self, frequency: Frequency) -> jnp.ndarray:
        """Evaluates the Admittance parameters."""
        raise NotImplementedError

    @abstractmethod
    def z(self, frequency: Frequency) -> jnp.ndarray:
        """Evaluates the Impedance parameters."""
        raise NotImplementedError
    
    @abstractmethod
    def a(self, frequency: Frequency) -> jnp.ndarray:
        """Evaluates the ABCD parameters."""
        raise NotImplementedError

    @abstractmethod
    def mna(self, frequency: Frequency) -> MNAStamp:
        """Evaluates the Modified Nodal Analysis matrices."""
        raise NotImplementedError