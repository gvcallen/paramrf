"""
Uniform transmission lines (RLGC, coaxial, microstrip)
"""
from abc import abstractmethod

import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.models.base import Model
from pmrf.rf import renormalize_s
from pmrf.constraints import Positive
from pmrf.types import ArrayLike
from pmrf.parameters import Param, param


class RLGCResult(eqx.Module):
    R: jnp.ndarray
    L: jnp.ndarray
    G: jnp.ndarray
    C: jnp.ndarray


class TransmissionLine(Model):
    """
    Abstract base interface for transmission lines.

    Used purely as a marker. Has no specific implementation requirements,
    hence no Abstract prefix.
    """
    pass


class AbstractUniformLine(TransmissionLine):
    r"""
    Abstract base class for all uniform transmission line models.

    Provides the fundamental equations to construct S-parameters 
    based on frequency-dependent characteristic impedance ($Z_c$) 
    and total complex electrical length ($\gamma L$). Derived classes 
    must implement the `zc_and_gammaL` method.

    **Mathematical Formulation**

    For a single-ended 2-port transmission line, the traveling wave S-parameters with respect to $Z_c$ are:
    $$S_{11} = S_{22} = 0$$
    $$S_{21} = S_{12} = e^{-\gamma L}$$

    This model computes these S-parameters and then re-normalized them into $Z_c$ and the power-wave definition
    using :meth:`pmrf.rf.renormalize_s`.
    """

    @abstractmethod
    def zc_and_gammaL(self, frequency: Frequency) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Calculates characteristic impedance ($Z_c$) and complex electrical length ($\gamma L$).

        Parameters
        ----------
        frequency : Frequency
            The frequency axis.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Array of characteristic impedance ($Z_c$) and complex electrical length ($\gamma L$).
        """
        raise NotImplementedError
    
    def zc(self, frequency: Frequency) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Calculates the characteristic impedance ($Z_c$).
        
        This just calls :meth:`pmrf.models.AbstractUniformLine.zc_and_gammaL`.

        Parameters
        ----------
        frequency : Frequency
            The frequency axis.

        Returns
        -------
        jnp.ndarray
            The characteristic impedance ($Z_c$)
        """
        return self.zc_and_gammaL(frequency)[0]

    def gammaL(self, frequency: Frequency) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""
        Calculates the complex electrical length ($\gamma L$).
        
        This just calls :meth:`pmrf.models.AbstractUniformLine.zc_and_gammaL`.

        Parameters
        ----------
        frequency : Frequency
            The frequency axis.

        Returns
        -------
        jnp.ndarray
            The complex electrical length ($\gamma L$)
        """
        return self.zc_and_gammaL(frequency)[1]

    def s(self, frequency: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        zc, gL = self.zc_and_gammaL(frequency)
        
        a = jnp.zeros(frequency.npoints, dtype=complex)
        s21 = jnp.exp(-1*gL)

        s = jnp.array([
            [a, s21],
            [s21, a],
        ]).transpose(2, 0, 1)

        # Renormalize into the requested characteristic impedance and power waves
        # (the above formulation is in terms of traveling waves).
        return renormalize_s(s, zc, z0, 'traveling', 'power')

    def y(self, frequency: Frequency) -> jnp.ndarray:
        zc, gL = self.zc_and_gammaL(frequency)
        
        # Set a safe noisefloor based on machine epsilon
        float_dtype = jnp.real(gL).dtype
        eps = jnp.finfo(float_dtype).eps
        MIN_GL = (eps * 10) + 0j
        
        # Threshold GL
        gL_safe = jnp.where(jnp.abs(gL) < eps, MIN_GL, gL)
        
        y11 = 1.0 / (zc * jnp.tanh(gL_safe))
        y12 = -1.0 / (zc * jnp.sinh(gL_safe))
        
        y = jnp.array([
            [y11, y12],
            [y12, y11],
        ]).transpose(2, 0, 1)
        
        return y


class AbstractRLGCLine(AbstractUniformLine):
    r"""
    Abstract base class for a transmission line defined by its per-unit-length
    RLGC (Resistance, Inductance, Conductance, Capacitance) parameters.

    Derived classes must implement `rlgc` to define how these parameters
    behave over frequency.

    **Mathematical Formulation**

    The characteristic impedance ($Z_c$) and complex propagation constant ($\gamma$) are derived as:
    $$Z_c = \sqrt{\frac{R + j\omega L}{G + j\omega C}}$$
    $$\gamma = \sqrt{(R + j\omega L)(G + j\omega C)}$$

    The total complex electrical length is $\gamma L$.

    Parameters
    ----------
    length : Param
        Physical length of the line in meters.
    """
    #: Physical length of the line
    length: Param = param(constraint=Positive())

    @abstractmethod
    def rlgc(self, freq: Frequency) -> RLGCResult:
        r"""
        Calculates the frequency-dependent RLGC parameters.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.

        Returns
        -------
        RLGCResult
            The R, L, G, and C parameter vectors.
        """
        raise NotImplementedError("'rlgc' must be implemented in the derived class")       

    def zc_and_gammaL(self, frequency: Frequency) -> tuple[jnp.ndarray, jnp.ndarray]:
        w = frequency.w
        
        rlgc = self.rlgc(frequency)

        R, L, G, C = rlgc.R, rlgc.L, rlgc.G, rlgc.C
        
        zc = jnp.sqrt((R + 1j*w*L) / (G + 1j*w*C))
        gamma = jnp.sqrt((R + 1j*w*L) * (G + 1j*w*C))
        gammaL = gamma*self.length
        
        return zc, gammaL  