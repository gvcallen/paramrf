import jax.numpy as jnp
from pmrf.frequency import Frequency
from pmrf.models.model import Model
from pmrf.models.lumped import MATCH, SHORT

class Port(Model):
    def __call__(self) -> Model:
        return MATCH

class Ground(Model):
    def __call__(self) -> Model:
        return SHORT

class Transformer(Model):
    """
    **Overview**

    An ideal, lossless, frequency-independent 4-port 1:1 transformer.
    """
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Returns the fixed S-parameter matrix for the transformer.

        Args:
            freq (Frequency): The frequency axis for the calculation.

        Returns:
            np.ndarray: The 4x4 S-parameter matrix, constant across frequency.
        """
        s = 0.5 * jnp.ones((freq.npoints, 4, 4), dtype='complex')
        s = s.at[:, 0, 3].set(-0.5)
        s = s.at[:, 1, 2].set(-0.5)
        s = s.at[:, 2, 1].set(-0.5)
        s = s.at[:, 3, 0].set(-0.5)

        return s
    
class SourceConverter(Model):
    def s(self, freq: Frequency) -> jnp.ndarray:
        s_one = jnp.array([
            [ 1,  2, -2],
            [ 2,  1,  2],
            [-2,  2,  1]
        ], dtype='complex')
        s_one /= 3.0

        s = jnp.tile(s_one, (freq.npoints, 1, 1))        
        return s