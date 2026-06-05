import jax.numpy as jnp
import equinox as eqx

class MNAStamp(eqx.Module):
    """Represents an MNA stamp."""
    
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