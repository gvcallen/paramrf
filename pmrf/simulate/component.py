"""Component base class for simulation"""

from abc import abstractmethod

import jax.numpy as jnp
from jaxtyping import ArrayLike
import equinox as eqx


class MNAStamp(eqx.Module):
    """Represents an MNA stamp for a component."""
    
    #: Nodal admittance matrix. Shape: (n, n)
    Y: jnp.ndarray  
    
    #: Maps auxiliary variables to node currents. Shape: (n, k)
    B: jnp.ndarray  
    
    #: Maps node voltages to auxiliary constraints. Shape: (k, n)
    C: jnp.ndarray  
    
    #: Auxiliary variable relationships. Shape: (k, k)
    D: jnp.ndarray  
    
    @property
    def num_ports(self) -> int:
        return self.Y.shape[0]
        
    @property
    def num_aux(self) -> int:
        return self.D.shape[0]
    

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
    def s_omega(self, w: ArrayLike, z0: ArrayLike = 50.0) -> jnp.ndarray:
        """Evaluates the Scattering parameters for a single angular frequency."""
        raise NotImplementedError

    @abstractmethod
    def y_omega(self, w: ArrayLike) -> jnp.ndarray:
        """Evaluates the Admittance parameters for a single angular frequency."""
        raise NotImplementedError

    @abstractmethod
    def z_omega(self, w: ArrayLike) -> jnp.ndarray:
        """Evaluates the Impedance parameters for a single angular frequency."""
        raise NotImplementedError
    
    @abstractmethod
    def a_omega(self, w: ArrayLike) -> jnp.ndarray:
        """Evaluates the ABCD parameters for a single angular frequency."""
        raise NotImplementedError

    @abstractmethod
    def mna_omega(self, w: ArrayLike) -> MNAStamp:
        """Evaluates the Modified Nodal Analysis matrices for a single angular frequency."""
        raise NotImplementedError