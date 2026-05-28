"""
Ideal models, such as ports, grounds and transformers.

These components are all non-tunable by default. To specify free parameters, use a constructor from :mod:`pmrf.parameters`.
"""
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils import field
from pmrf.rf import renormalize_s
from pmrf.types import ArrayLike
from pmrf.parameters import Param, param


class Load(Model):
    """
    A class for ideal N-port loads defined by their reflection coefficient.
    
    The reflection coefficient is defined into a reference impedance `z0`.
    """
    #: The reflection coefficient
    gamma: Param = param()
    
    #: The reference impedance at which this gamma is defined
    z0: Param = param(default=50.0)
    
    #: Number of ports
    nports: int = eqx.field(default=1, static=True)
    
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        s_matrix = jnp.asarray(self.gamma).reshape(-1, 1, 1) * \
                   jnp.eye(self.nports, dtype=jnp.complex128).reshape((-1, self.nports, self.nports)).\
                   repeat(freq.npoints, 0)
        
        return renormalize_s(s_matrix, self.z0, z0)


class Short(Model):
    """
    A standard ideal short circuit load (gamma = -1.0).
    
    This short does not have a reference impedance, since an ideal short circuit
    has the same S-parameters irrespective of the reference impedance.
    
    Parameters
    ----------
    nports : int
        The number of ports the short presents. Default is 1.
    """
    #: Number of ports
    nports: int = field(default=1, static=True)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        s_val = jnp.array(-1.0, dtype=jnp.complex128)
        return jnp.eye(self.nports, dtype=jnp.complex128) * s_val * jnp.ones((freq.npoints, 1, 1))    


class Open(Model):
    """
    A standard ideal open circuit load (gamma = 1.0).
    
    This open does not have a reference impedance, since an ideal short circuit
    has the same S-parameters irrespective of the reference impedance.
    
    Parameters
    ----------
    nports : int
        The number of ports the open presents. Default is 1.
    """
    #: Number of ports
    nports: int = field(default=1, static=True)
    
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        s_val = jnp.array(1.0, dtype=jnp.complex128)
        return jnp.eye(self.nports, dtype=jnp.complex128) * s_val * jnp.ones((freq.npoints, 1, 1))    


class Match(Model):
    """
    A standard ideal matched circuit load (gamma = 0.0).
    
    This load does not have a reference impedance, and dynamically
    matches the impedance at which the S-parameters are evaluated at.
    
    Parameters
    ----------
    nports : int
        The number of ports the match presents. Default is 1.
    """
    #: Number of ports
    nports: int = field(default=1, static=True)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return jnp.zeros(
            (freq.npoints, self.nports, self.nports), 
            dtype=jnp.complex128
        )


class Port(Model):
    """
    Represents a circuit port with a specific characteristic impedance.
    
    This class serves as a placeholder or marker for external connections in a :class:`pmrf.models.Circuit` definition.
    Calling `build` returns a matched load model.
    """
    #: Port characteristic impedance
    z0: Param = param(default=50.0)

    def build(self) -> Model:
        return Load(gamma=0.0, z0=self.z0)
    

class Ground(Model):
    """
    Represents a ground connection.

    This class serves as a placeholder for a ground node in a :class:`pmrf.models.Circuit` definition.
    Calling `build` returns a short circuit model.
    
    The ground does not have a reference impedance, since an ideal ground circuit
    has the same S-parameters irrespective of the reference impedance.
    """
    def build(self) -> Model:
        return Short()


class Transformer(Model):
    """
    (experimental) An ideal, lossless, frequency-independent 4-port 1:N transformer.

    The S-parameters are constant across all frequencies
    and are independent of the characteristic impedance.
    
    Parameters
    ----------
    N : float
        The turns ratio (1:N) from primary to secondary. Defaults to 1.0.
    """
    #: The turns ratio 1:N
    N: float = field(default=1.0, static=True, kw_only=True)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        N = self.N
        D = 1.0 + N**2
        
        # The intrinsic 4-port scattering matrix for an ideal 1:N transformer
        s_mat = jnp.array([
            [1.0,   N**2,  N,    -N],
            [N**2,  1.0,  -N,     N],
            [N,    -N,     N**2,  1.0],
            [-N,    N,     1.0,   N**2]
        ], dtype=jnp.complex128) / D

        # Broadcast the 4x4 matrix across the frequency grid
        return jnp.broadcast_to(s_mat, (freq.npoints, 4, 4))
    

class SourceConverter(Model):
    """
    (experimental) An ideal 3-port source converter.

    This model represents a specific ideal component with a fixed, frequency-independent
    3x3 scattering matrix that is independent of the characteristic impedance.
    """
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
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
    
    Allows perfect transmission from Port 1 to Port 2, and attenuates 
    reverse transmission from Port 2 to Port 1 by `isolation` dB. Both 
    ports are perfectly matched at the designed characteristic impedance.
    
    Parameters
    ----------
    isolation : Param, default=np.inf
        The reverse isolation in dB. 
        Defaults to infinity (perfect isolation).
    z0 : ArrayLike, default=50.0
        The intrinsic characteristic impedance for which the isolator was designed.
    """
    #: The reverse isolation in dB.
    isolation: Param = param(default=np.inf)
    
    #: The intrinsic characteristic impedance of the physical device.
    z0: np.ndarray = field(default=50.0, converter=np.asarray, kw_only=True)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        s12_lin = 10.0 ** (-self.isolation / 20.0)
        
        s21 = jnp.ones(freq.npoints, dtype=complex)
        s12 = s12_lin * jnp.ones(freq.npoints, dtype=complex)
        zeros = jnp.zeros(freq.npoints, dtype=complex)

        s_mat = jnp.array([
            [zeros, s12],
            [s21, zeros],
        ]).transpose(2, 0, 1)
        return renormalize_s(s_mat, self.z0, z0, 'power', 'power')


class Splitter(Model):
    """
    (experimental) An ideal n-way parallel node (lossless junction).

    This model represents a purely topological connection where all n ports 
    are tied to a single common voltage node. Because it is lossless and 
    reciprocal, it cannot be simultaneously matched at all ports (e.g., a 
    3-port Tee will have S11 = -1/3).
    
    This model does not have a reference impedance and dynamically
    matches the evaluation characteristic impedance.

    Parameters
    ----------
    nports : int
        The number of ports. Defaults to 3 (a 2-way split).
    """
    nports: int = field(default=3, static=True)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
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
    
    This model does not have a reference impedance and dynamically
    matches the evaluation characteristic impedance.
    """
    def build(self) -> Model:
        return Splitter(nports=3)
    

class Attenuator(Model):
    """
    (experimental) A matched, 2-port physical attenuator.

    Parameters
    ----------
    loss : Param
        The attenuation in dB (a positive value indicates loss).
    z0 : ArrayLike, default=50.0
        The intrinsic characteristic impedance for which the physical attenuator 
        was designed.
    """
    #: The attenuation in dB.
    loss: Param = param()
    
    #: The intrinsic characteristic impedance of the physical device.
    z0: np.ndarray = field(default=50.0, converter=np.asarray, kw_only=True)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        s21_lin = 10.0 ** (-self.loss / 20.0)
        
        s21 = s21_lin * jnp.ones(freq.npoints, dtype=complex)
        zeros = jnp.zeros(freq.npoints, dtype=complex)

        s_mat = jnp.array([
            [zeros, s21],
            [s21, zeros],
        ]).transpose(2, 0, 1)
        
        return renormalize_s(s_mat, self.z0, z0, 'power', 'power')
        
        
class DirectionalCoupler(Model):
    """
    (experimental) An ideal 4-port tunable directional coupler.

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
    z0 : ArrayLike, default=50.0
        The intrinsic characteristic impedance for which the coupler was designed.
    """
    #: The linear voltage coupling factor.
    coupling: Param = param()
    
    #: The intrinsic characteristic impedance of the physical device.
    z0: np.ndarray = field(default=50.0, converter=np.asarray, kw_only=True)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        c = self.coupling
        t = jnp.sqrt(1.0 - c**2)
        
        ones = jnp.ones(freq.npoints, dtype=float)
        zeros = jnp.zeros(freq.npoints, dtype=complex)
        
        C = 1j * c * ones
        T = t * ones + 0j

        s_mat = jnp.array([
            [zeros, T, C, zeros],
            [T, zeros, zeros, C],
            [C, zeros, zeros, T],
            [zeros, C, T, zeros]
        ]).transpose(2, 0, 1)
        
        return renormalize_s(s_mat, self.z0, z0, 'power', 'power')