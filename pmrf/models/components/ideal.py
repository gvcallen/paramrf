"""
Ideal models, such as ports, grounds and transformers.
"""
import jax.numpy as jnp
from jaxtyping import ArrayLike

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.models.components.lumped import Match, Short
from pmrf.utils import field, freeze
from pmrf.parameters import Param, param

class Port(Model):
    """
    Represents a circuit port.

    This class serves as a placeholder or marker for external connections in a circuit definition.
    Calling an instance returns a matched load model.
    """
    def build(self) -> Model:
        return Match()
    
class Ground(Model):
    """
    Represents a ground connection.

    This class serves as a placeholder for a ground node in a circuit definition.
    Calling an instance returns a short circuit model.
    """
    def build(self) -> Model:
        return Short()

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
    

class Isolator(Model):
    """
    (experimental) An ideal 2-port isolator. 
    
    Allows perfect transmission from Port 1 to Port 2, and infinite 
    isolation (zero transmission) from Port 2 to Port 1. Both ports are 
    perfectly matched.
    """
    def s(self, freq: Frequency) -> jnp.ndarray:
        ones = jnp.ones(freq.npoints, dtype=complex)
        zeros = jnp.zeros(freq.npoints, dtype=complex)

        return jnp.array([
            [zeros, zeros],
            [ones, zeros],
        ]).transpose(2, 0, 1)


class Splitter(Model):
    """
    (experimental) An ideal, lossless n-way power splitter.

    The port impedances are matched when driving one port and terminating 
    the others in matched loads. Power is split equally among the remaining ports.

    Parameters
    ----------
    nports : int
        The number of ports. Defaults to 3 (a 2-way split).
    """
    nports: int = field(default=3, static=True)

    def s(self, freq: Frequency) -> jnp.ndarray:
        n = self.nports
        
        # S-parameter equations for an ideal N-port equal splitter
        s_ii = (2.0 - n) / n
        s_ij = 2.0 / n
        
        # Build the static NxN matrix
        S_mat = jnp.full((n, n), s_ij, dtype=complex)
        S_mat = S_mat.at[jnp.diag_indices(n)].set(s_ii)
        
        # Broadcast across the frequency axis
        return jnp.broadcast_to(S_mat, (freq.npoints, n, n))


class Tee(Model):
    """
    (experimental) An ideal, lossless 3-port Tee junction.

    This is a convenience wrapper around a 3-port Splitter.
    """
    def build(self) -> Model:
        return Splitter(nports=3)
    

class VariableAttenuator(Model):
    """
    An variable, matched, 2-port attenuator.

    Parameters
    ----------
    s21 : Param
        The linear voltage transmission coefficient. 
        For example, for a 3 dB attenuator, s21 = 10**(-3/20) ≈ 0.707.
    """
    s21: Param = param()

    def s(self, freq: Frequency) -> jnp.ndarray:
        s21 = self.s21 * jnp.ones(freq.npoints, dtype=complex)
        zeros = jnp.zeros(freq.npoints, dtype=complex)

        return jnp.array([
            [zeros, s21],
            [s21, zeros],
        ]).transpose(2, 0, 1)
        

class VariableDirectionalCoupler(Model):
    """
    An ideal 4-port tunable directional coupler.

    Port routing:
    - Port 1: Input
    - Port 2: Through
    - Port 3: Coupled
    - Port 4: Isolated

    Parameters
    ----------
    coupling : Param
        The linear voltage coupling factor. 
        For example, for a 20 dB coupler, coupling = 10**(-20/20) = 0.1.
    """
    coupling: Param = param()

    def s(self, freq: Frequency) -> jnp.ndarray:
        c = self.coupling
        t = jnp.sqrt(1.0 - c**2)
        
        ones = jnp.ones(freq.npoints, dtype=float)
        zeros = jnp.zeros(freq.npoints, dtype=complex)
        
        C = 1j * c * ones
        T = t * ones + 0j

        return jnp.array([
            [zeros, T, C, zeros],
            [T, zeros, zeros, C],
            [C, zeros, zeros, T],
            [zeros, C, T, zeros]
        ]).transpose(2, 0, 1)


class Attenuator(Model):
    """
    An ideal, matched, constant, 2-port attenuator.

    Parameters
    ----------
    s21 : ArrayLike
        The linear voltage transmission coefficient. 
        For example, for a 3 dB attenuator, s21 = 10**(-3/20) ≈ 0.707.
    """
    s21: ArrayLike = field(converter=freeze)

    def build(self) -> Model:
        return VariableAttenuator(self.s21)


class DirectionalCoupler(Model):
    """
    An ideal 4-port directional coupler.

    Port routing:
    - Port 1: Input
    - Port 2: Through
    - Port 3: Coupled
    - Port 4: Isolated

    Parameters
    ----------
    coupling : Param
        The linear voltage coupling factor. 
        For example, for a 20 dB coupler, coupling = 10**(-20/20) = 0.1.
    """
    coupling: ArrayLike = field(converter=freeze)

    def build(self) -> Model:
        return VariableDirectionalCoupler(self.coupling)


# class LosslessMismatch(Model):
#     """
#     An ideal, lossless, symmetric 2-port mismatch defined by its return loss.

#     This component automatically calculates the phase and magnitude of S21 
#     required to ensure the network remains unitary (perfectly lossless). 
#     It is extremely useful for inserting synthetic mismatch reflections 
#     into a cascade for Monte Carlo or tolerance analysis.

#     Parameters
#     ----------
#     s11 : Param
#         The complex reflection coefficient.
#     """
#     s11: Param = param()

#     def s(self, freq: Frequency) -> jnp.ndarray:
#         s11_val = self.s11
#         s21_mag = jnp.sqrt(1.0 - jnp.abs(s11_val)**2)
        
#         # Calculate the phase required for a lossless reciprocal 2-port
#         s11_angle = jnp.angle(s11_val)
#         s21_phase = s11_angle + jnp.where(s11_angle <= 0, jnp.pi / 2, -jnp.pi / 2)
        
#         s21_val = s21_mag * jnp.exp(1j * s21_phase)
        
#         ones = jnp.ones(freq.npoints, dtype=complex)
#         S11 = s11_val * ones
#         S21 = s21_val * ones

#         return jnp.array([
#             [S11, S21],
#             [S21, S11],
#         ]).transpose(2, 0, 1)