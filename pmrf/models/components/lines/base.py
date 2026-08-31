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


class ImmittanceResult(eqx.Module):
    r"""
    Per-unit-length series impedance and shunt admittance of a line.

    Immittance is the internal currency between a formulation and a line.
    $R$, $L$, $G$ and $C$ are derived views on it rather than the primary form,
    because $L = \Im(Z)/\omega$ and $C = \Im(Y)/\omega$ both carry a removable
    $0/0$ at DC that $(Z, Y)$ does not, and because
    :meth:`AbstractImmittanceLine.zc_and_gammaL` then reduces to $\sqrt{Z/Y}$ and
    $\sqrt{ZY}$ with no division by $\omega$ anywhere.

    Parameters
    ----------
    Z : jnp.ndarray
        Series impedance per unit length, $Z = R + j\omega L$, in ohm/m.
    Y : jnp.ndarray
        Shunt admittance per unit length, $Y = G + j\omega C$, in S/m.
    w : jnp.ndarray
        Angular frequency axis, kept so that `L` and `C` can be recovered.
    """
    #: Series impedance per unit length in ohm/m
    Z: jnp.ndarray

    #: Shunt admittance per unit length in S/m
    Y: jnp.ndarray

    #: Angular frequency axis in rad/s
    w: jnp.ndarray

    @classmethod
    def from_rlgc(cls, R, L, G, C, w) -> "ImmittanceResult":
        r"""Build an immittance from per-unit-length $R$, $L$, $G$ and $C$."""
        w = jnp.asarray(w)
        return cls(Z=R + 1j * w * L, Y=G + 1j * w * C, w=w)

    @classmethod
    def from_zc_gamma(cls, zc, gamma, w) -> "ImmittanceResult":
        r"""Invert characteristic impedance and propagation constant exactly.

        The per-unit-length quantities are the exact bijection
        $$Z = \gamma Z_c \qquad Y = \frac{\gamma}{Z_c}.$$

        A passive line must have non-negative series resistance and shunt
        conductance. Violating that condition indicates that an empirical
        formulation has been evaluated outside its physical regime.
        """
        Z = gamma * zc
        Y = gamma / zc
        # Exact lossless products can leave round-off-sized negative real parts.
        # Reject physical negativity while allowing that numerical noise floor.
        z_tol = 64 * jnp.finfo(jnp.real(Z).dtype).eps * jnp.maximum(jnp.abs(Z), 1)
        y_tol = 64 * jnp.finfo(jnp.real(Y).dtype).eps * jnp.maximum(jnp.abs(Y), 1)
        Z = eqx.error_if(
            Z, jnp.any(jnp.real(Z) < -z_tol), "inversion produced Re(Z) < 0"
        )
        Y = eqx.error_if(
            Y, jnp.any(jnp.real(Y) < -y_tol), "inversion produced Re(Y) < 0"
        )
        return cls(Z=Z, Y=Y, w=jnp.asarray(w))

    @property
    def R(self) -> jnp.ndarray:
        """Series resistance per unit length in ohm/m."""
        return jnp.real(self.Z)

    @property
    def G(self) -> jnp.ndarray:
        """Shunt conductance per unit length in S/m."""
        return jnp.real(self.Y)

    @property
    def L(self) -> jnp.ndarray:
        r"""Series inductance per unit length in H/m, $\Im(Z)/\omega$."""
        return self._per_w(jnp.imag(self.Z))

    @property
    def C(self) -> jnp.ndarray:
        r"""Shunt capacitance per unit length in F/m, $\Im(Y)/\omega$."""
        return self._per_w(jnp.imag(self.Y))

    def _per_w(self, x: jnp.ndarray) -> jnp.ndarray:
        # The DC limit is finite but reached as 0/0. Use the double-`where`
        # pattern so both the value and its gradient stay well defined, then
        # carry the lowest non-zero frequency into any DC entry, which is exact
        # whenever the reactance is linear in omega there.
        w = self.w
        safe_w = jnp.where(w > 0, w, 1.0)
        ratio = jnp.where(w > 0, x / safe_w, 0.0)
        first = jnp.take(ratio, jnp.argmax(w > 0))
        return jnp.where(w > 0, ratio, first)


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


class AbstractImmittanceLine(AbstractUniformLine):
    r"""
    Abstract base class for a transmission line defined by its per-unit-length
    series impedance and shunt admittance.

    Derived classes must implement `immittance` to define how those behave over
    frequency. $R$, $L$, $G$ and $C$ remain available as derived views on
    :class:`ImmittanceResult`.

    **Mathematical Formulation**

    The characteristic impedance ($Z_c$) and complex propagation constant ($\gamma$) are derived as:
    $$Z_c = \sqrt{\frac{Z}{Y}}$$
    $$\gamma = \sqrt{ZY}$$

    where $Z = R + j\omega L$ and $Y = G + j\omega C$. The total complex
    electrical length is $\gamma L$.

    Parameters
    ----------
    length : Param
        Physical length of the line in meters.
    """
    #: Physical length of the line
    length: Param = param(constraint=Positive())

    @abstractmethod
    def immittance(self, freq: Frequency) -> ImmittanceResult:
        r"""
        Calculates the frequency-dependent per-unit-length immittance.

        Parameters
        ----------
        freq : Frequency
            The frequency axis.

        Returns
        -------
        ImmittanceResult
            The series impedance and shunt admittance vectors.
        """
        raise NotImplementedError("'immittance' must be implemented in the derived class")

    def zc_and_gammaL(self, frequency: Frequency) -> tuple[jnp.ndarray, jnp.ndarray]:
        immittance = self.immittance(frequency)
        Z, Y = immittance.Z, immittance.Y

        w = immittance.w

        # Both square roots are singular at DC on a line with no static loss,
        # where Z = Y = 0: sqrt(Z/Y) is a raw 0/0 and sqrt(ZY) sits on the
        # branch point, whose derivative diverges. Guard both with the
        # double-`where` pattern so the gradient stays finite too.
        product = Z * Y
        singular_gamma = product == 0
        safe_product = jnp.where(singular_gamma, 1.0, product)
        gamma = jnp.where(singular_gamma, 0.0, jnp.sqrt(safe_product))

        singular_zc = (w <= 0) & singular_gamma
        safe_Y = jnp.where(singular_zc, 1.0, Y)
        ratio = jnp.where(singular_zc, 1.0, Z / safe_Y)
        zc = jnp.sqrt(ratio)
        zc = jnp.where(singular_zc, jnp.take(zc, jnp.argmax(w > 0)), zc)

        return zc, gamma * self.length
