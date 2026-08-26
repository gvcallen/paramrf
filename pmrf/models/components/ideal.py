"""
Ideal models, such as ports, grounds and transformers.

These components are all non-tunable by default. To specify free parameters, use a constructor from :mod:`pmrf.parameters`.
"""
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.utils import error_if, field
from pmrf.utils.rf import fix_z0_shape
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
    
    This class serves as a placeholder or marker for external connections
    in a :class:`pmrf.models.Circuit` definition, though can also be used
    as a simple tagged load in other types of models.
    """
    #: Port characteristic impedance
    z0: Param = param(default=50.0)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return Load(gamma=0.0, z0=self.z0).s(freq, z0=z0)
    

class Ground(Model):
    """
    Represents a ground connection.

    This class serves as a placeholder for a ground node in a :class:`pmrf.models.Circuit` definition.
    
    The ground does not have a reference impedance, since an ideal ground circuit
    has the same S-parameters irrespective of the reference impedance.
    """
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return Short().s(freq, z0=z0)


def _constraint_s(currents: ArrayLike, npoints: int) -> jnp.ndarray:
    r"""
    The S-parameters of an ideal, lossless constraint network.

    An ideal transformer has no dynamics of its own: it only constrains the terminal
    voltages and currents. Since such a network neither stores nor dissipates energy,
    every allowed voltage vector is orthogonal to every allowed current vector, and the
    two subspaces are orthogonal complements. Writing the allowed currents as the range
    of :math:`X`, the scattering matrix is then the reflection

    .. math::

        S = I - 2 X (X^T X)^{-1} X^T.

    The result is real, symmetric and orthogonal, i.e. reciprocal and lossless, and it is
    independent of both frequency and the reference impedance.

    Parameters
    ----------
    currents : ArrayLike
        An ``(n, r)`` matrix whose columns span the allowed terminal currents of the
        n-terminal network.
    npoints : int
        The number of frequency points to broadcast the result across.

    Returns
    -------
    jnp.ndarray
        S-parameter matrix with shape ``(npoints, n, n)``.
    """
    X = jnp.asarray(currents)
    n = X.shape[0]

    projector = X @ jnp.linalg.solve(X.T @ X, X.T)
    s_mat = (jnp.eye(n) - 2.0 * projector).astype(jnp.complex128)

    return jnp.broadcast_to(s_mat, (npoints, n, n))


class Transformer(Model):
    """
    (experimental) An ideal, lossless, frequency-independent 4-port 1:N transformer.

    The primary winding sits across ports 1 and 2, and the secondary winding across
    ports 3 and 4, so that :math:`V_3 - V_4 = N (V_1 - V_2)` and
    :math:`I_3 = -I_1 / N`. Both windings are isolated: neither terminal pair carries
    a net current, and their common-mode voltages are unconstrained.

    The S-parameters are constant across all frequencies
    and are independent of the characteristic impedance.
    
    Parameters
    ----------
    N : Param
        The turns ratio (1:N) from primary to secondary. Defaults to 1.0.
    """
    #: The turns ratio 1:N
    N: Param = param(default=1.0)

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


class CentreTappedTransformer(Model):
    """
    (experimental) An ideal, lossless, frequency-independent 5-port 1:N transformer
    with a tapped primary winding.

    This is a :class:`Transformer` with an additional terminal brought out from an
    intermediate point of the primary winding. The primary sits across ports 1 and 2,
    the secondary across ports 3 and 4, and the tap is port 5.

    The tap divides the primary into two series windings, with a fraction `tap` of the
    primary turns between port 1 and the tap, and the remainder between the tap and
    port 2. A `tap` of 0.5 therefore gives a true centre tap, and asymmetric taps are
    expressed by moving `tap` away from 0.5.

    The S-parameters are constant across all frequencies
    and are independent of the characteristic impedance.

    Parameters
    ----------
    N : Param
        The turns ratio (1:N) from the full primary to the secondary. Defaults to 1.0.
    tap : Param
        The fraction of the primary turns between port 1 and the tap. Defaults to 0.5.
    """
    #: The turns ratio 1:N
    N: Param = param(default=1.0)

    #: The fraction of the primary winding between port 1 and the tap
    tap: Param = param(default=0.5)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        N = self.N
        tap = self.tap

        # Currents of the two upper and lower primary windings, referred to the secondary
        upper = tap / N
        lower = (1.0 - tap) / N

        one = jnp.ones_like(upper)
        zero = jnp.zeros_like(upper)

        # Each column drives one primary section against the secondary at zero net MMF
        currents = jnp.stack([
            jnp.stack([one,  zero, -upper, upper, -one]),
            jnp.stack([zero, -one, -lower, lower,  one]),
        ], axis=-1)

        return _constraint_s(currents, freq.npoints)


class Autotransformer(Model):
    """
    (experimental) An ideal, lossless, frequency-independent 3-port 1:N autotransformer.

    A single tapped winding runs from port 1 to port 3, with the tap brought out at
    port 2, so that :math:`V_1 - V_3 = N (V_2 - V_3)`. Port 1 therefore sees the full
    winding and port 2 the tapped section, making `N` the step-up ratio from the tap to
    the full winding. Equivalently, this is a 1:(N-1) :class:`Transformer` with its
    secondary connected in series with its primary.

    Being a single winding, this component provides no isolation: the three terminal
    currents sum to zero.

    The S-parameters are constant across all frequencies
    and are independent of the characteristic impedance.

    Parameters
    ----------
    N : Param
        The turns ratio (1:N) from the tapped section to the full winding.
        Defaults to 1.0, for which the tap coincides with port 1.
    """
    #: The turns ratio 1:N
    N: Param = param(default=1.0)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        N = jnp.asarray(self.N)
        one = jnp.ones_like(N)

        # The series-connected sections carry a common current at zero net MMF
        currents = jnp.stack([-one, N, one - N])[:, jnp.newaxis]

        return _constraint_s(currents, freq.npoints)


class Balun(Model):
    """
    (experimental) An ideal, lossless, frequency-independent 3-port 1:N balun.

    This is a :class:`Transformer` whose primary is referenced to ground, converting the
    single-ended port 1 into the balanced pair of ports 2 and 3, such that
    :math:`V_2 - V_3 = N V_1`.

    The S-parameters are constant across all frequencies
    and are independent of the characteristic impedance.

    Parameters
    ----------
    N : Param
        The turns ratio (1:N) from the single-ended side to the balanced side.
        Defaults to 1.0, for which this is equivalent to a :class:`SourceConverter`.
    """
    #: The turns ratio 1:N
    N: Param = param(default=1.0)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        N = jnp.asarray(self.N)
        one = jnp.ones_like(N)

        # The single-ended current returns through ground, so the terminal currents
        # need not sum to zero
        currents = jnp.stack([N, -one, one])[:, jnp.newaxis]

        return _constraint_s(currents, freq.npoints)
    

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
    

def _require_equal_z0(x, z0: ArrayLike, nports: int, npoints: int):
    """
    Thread `x` through a runtime check that all `nports` reference impedances are equal.

    Parameters
    ----------
    x : Any
        The value to pass through, typically the S-parameter matrix being guarded.
    z0 : ArrayLike
        The reference impedance, in any shape accepted by :func:`pmrf.utils.rf.fix_z0_shape`.
    nports : int
        The number of ports.
    npoints : int
        The number of frequency points.

    Returns
    -------
    Any
        The unmodified input `x`.

    Raises
    ------
    equinox.EquinoxRuntimeError
        At runtime, if the ports do not all share the same reference impedance.
    """
    z0_arr = fix_z0_shape(z0, npoints, nports)
    unequal = jnp.any(z0_arr != z0_arr[..., :1])

    return error_if(
        x,
        unequal,
        f"All {nports} reference impedances must be equal, got z0 = {{}}",
        z0_arr,
    )


class MixedModeConverter(Model):
    r"""
    (experimental) An ideal, lossless, frequency-independent 4-port mixed-mode converter.

    This component converts a pair of equal-impedance physical (single-ended) ports into a
    differential-mode and a common-mode port, using the AWR voltage/current convention

    .. math::

        V_d = V_p - V_n, \qquad V_c = \frac{V_p + V_n}{2},

    with the corresponding modal currents, taken as flowing *into* the converter alongside
    the physical currents,

    .. math::

        I_d = \frac{I_n - I_p}{2}, \qquad I_c = -(I_p + I_n).

    Port ordering:

    - Port 1: differential mode of ports 3 and 4
    - Port 2: common mode of ports 3 and 4
    - Port 3: physical (single-ended) positive port `p`
    - Port 4: physical (single-ended) negative port `n`

    With every port referenced to the same impedance :math:`Z_0`, this gives the signed
    scattering matrix

    .. math::

        S = \frac{1}{3} \begin{bmatrix}
             1 &  0 &  2 & -2 \\
             0 & -1 &  2 &  2 \\
             2 &  2 &  0 &  1 \\
            -2 &  2 &  1 &  0
        \end{bmatrix},

    which is real, symmetric (reciprocal) and orthogonal (lossless).

    Unlike a power-normalized converter, the modal ports here carry the *physical* modal
    impedances: the differential port presents :math:`Z_d = 2 Z_0` and the common port
    :math:`Z_c = Z_0 / 2`, which for 50 Ohm physical ports are 100 Ohm and 25 Ohm. This is
    visible in the diagonal, where :math:`S_{11} = 1/3` and :math:`S_{22} = -1/3` are the
    reflections of :math:`2 Z_0` and :math:`Z_0 / 2` in :math:`Z_0`. Modal loads may
    therefore be connected to the modal ports directly, at their physical impedances and
    with no external factor-of-two scaling.

    The S-parameters are constant across all frequencies, and independent of the value of
    the reference impedance, provided every port shares the same one. This is validated on
    evaluation.

    Raises
    ------
    equinox.EquinoxRuntimeError
        At runtime, if the four reference impedances passed to :meth:`s` are not all equal.
    """
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        # Ordered (differential, common, positive, negative)
        s_mat = jnp.array([
            [ 1.0,  0.0,  2.0, -2.0],
            [ 0.0, -1.0,  2.0,  2.0],
            [ 2.0,  2.0,  0.0,  1.0],
            [-2.0,  2.0,  1.0,  0.0],
        ], dtype=jnp.complex128) / 3.0

        s_mat = jnp.broadcast_to(s_mat, (freq.npoints, 4, 4))

        return _require_equal_z0(s_mat, z0, 4, freq.npoints)


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
    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        return Splitter(nports=3).s(freq, z0=z0)
    

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
    

class Amplifier(Model):
    """
    (experimental) An ideal, matched, unilateral 2-port amplifier.
    
    Unlike an attenuator, an ideal amplifier provides unilateral forward 
    gain (S21) with perfect reverse isolation (S12 = 0).

    Parameters
    ----------
    gain : Param
        The forward gain in dB (a positive value indicates gain).
    z0 : ArrayLike, default=50.0
        The intrinsic characteristic impedance for which the amplifier 
        was designed.
    """
    #: The forward gain in dB.
    gain: Param = param()
    
    #: The intrinsic characteristic impedance of the physical device.
    z0: np.ndarray = field(default=50.0, converter=np.asarray, kw_only=True)

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        s21_lin = 10.0 ** (self.gain / 20.0)
        
        s21 = s21_lin * jnp.ones(freq.npoints, dtype=complex)
        zeros = jnp.zeros(freq.npoints, dtype=complex)

        # Note the structural difference from the reciprocal attenuator
        s_mat = jnp.array([
            [zeros, zeros],
            [s21,   zeros],
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
