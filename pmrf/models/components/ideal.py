"""
Ideal models, such as ports, grounds and transformers.
"""
import jax.numpy as jnp
from pmrf.core import Frequency
from pmrf.core import Model
from pmrf.models.components.lumped import MATCH, SHORT

class Port(Model):
    """
    Represents a circuit port.

    This class serves as a placeholder or marker for external connections in a circuit definition.
    Calling an instance returns a matched load model.
    """
    def __call__(self) -> Model:
        """
        Return the equivalent model for a terminated port.

        Returns
        -------
        Model
            A matched load (`MATCH`).
        """
        return MATCH
    
class Ground(Model):
    """
    Represents a ground connection.

    This class serves as a placeholder for a ground node in a circuit definition.
    Calling an instance returns a short circuit model.
    """
    def __call__(self) -> Model:
        return SHORT

class Transformer(Model):
    """
    An ideal, lossless, frequency-independent 4-port 1:1 transformer.

    The S-parameters are constant across all frequencies.
    """
    def s(self, freq: Frequency) -> jnp.ndarray:
        s = 0.5 * jnp.ones((freq.npoints, 4, 4), dtype='complex')
        s = s.at[:, 0, 3].set(-0.5)
        s = s.at[:, 1, 2].set(-0.5)
        s = s.at[:, 2, 1].set(-0.5)
        s = s.at[:, 3, 0].set(-0.5)

        return s
    
class SourceConverter(Model):
    """
    An ideal 3-port source converter.

    This model represents a specific ideal component with a fixed, frequency-independent
    3x3 scattering matrix.
    """
    def s(self, freq: Frequency) -> jnp.ndarray:
        s_one = jnp.array([
            [ 1,  2, -2],
            [ 2,  1,  2],
            [-2,  2,  1]
        ], dtype='complex')
        s_one /= 3.0

        s = jnp.tile(s_one, (freq.npoints, 1, 1))        
        return s