"""
Bijectors not present in distreqx.
"""

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array
from typing import Callable

import jax.numpy as jnp
import distreqx.bijectors as bij


class Identity(
    bij.AbstractFowardInverseBijector,
    bij.AbstractInvLogDetJacBijector,
    bij.AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Identity bijector: y = x."""

    _is_constant_jacobian: bool = True
    _is_constant_log_det: bool = True

    def forward_and_log_det(self, x: PyTree):
        log_det = jnp.zeros_like(x)
        return x, log_det

    def inverse_and_log_det(self, y: PyTree):
        log_det = jnp.zeros_like(y)
        return y, log_det

    def same_as(self, other) -> bool:
        return isinstance(other, Identity)


class Inverse(bij.AbstractFwdLogDetJacBijector, bij.AbstractInvLogDetJacBijector, strict=True):
    """Inverted version of a given bijector."""

    bijector: bij.AbstractBijector
    # We set default values so Equinox doesn't demand them as init arguments
    _is_constant_jacobian: bool = False 
    _is_constant_log_det: bool = False

    def __post_init__(self):
        # self.bijector is already set by Equinox's auto-generated __init__
        is_constant_jacobian = self.bijector._is_constant_jacobian
        is_constant_log_det = getattr(self.bijector, '_is_constant_log_det', is_constant_jacobian)
        
        if is_constant_jacobian and not is_constant_log_det:
            raise ValueError(
                "The Jacobian is said to be constant, but its "
                "determinant is said not to be, which is impossible."
            )
            
        # Object.__setattr__ is sometimes needed in __post_init__ if frozen=True, 
        # but standard assignment usually works in Equinox modules.
        self._is_constant_jacobian = is_constant_jacobian
        self._is_constant_log_det = is_constant_log_det

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        return self.bijector.inverse(x)

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        return self.bijector.forward(y)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.bijector.inverse_and_log_det(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.bijector.forward_and_log_det(y)

    def same_as(self, other: bij.AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Inverse:
            return self.bijector.same_as(other.bijector)
        else:
            return self.bijector.same_as(other)
        
        
class Lambda(
    bij.AbstractFowardInverseBijector,
    bij.AbstractInvLogDetJacBijector,
    bij.AbstractFwdLogDetJacBijector,
    strict=True,
):
    """
    A bijector defined by arbitrary callable functions.
    
    This is useful for creating inline bijectors without needing to define 
    a custom class.
    """
    
    fn_forward: Callable[[PyTree], PyTree]
    fn_inverse: Callable[[PyTree], PyTree]
    fn_forward_log_det: Callable[[PyTree], PyTree]
    fn_inverse_log_det: Callable[[PyTree], PyTree]
    
    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = f(x) and log|det J(f)(x)| using the provided callables."""
        return self.fn_forward(x), self.fn_forward_log_det(x)

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)| using the provided callables."""
        return self.fn_inverse(y), self.fn_inverse_log_det(y)

    def same_as(self, other) -> bool:
        """
        Returns True if the other is a Lambda bijector with the exact same callables.
        
        Note: Python cannot reliably determine if two different lambda expressions 
        are mathematically equivalent, so this strictly checks for object identity 
        of the functions.
        """
        return (
            type(other) is Lambda and 
            self.fn_forward is other.fn_forward and 
            self.fn_inverse is other.fn_inverse and
            self.fn_forward_log_det is other.fn_forward_log_det and
            self.fn_inverse_log_det is other.fn_inverse_log_det
        )


class Exp(
    bij.AbstractFowardInverseBijector,
    bij.AbstractInvLogDetJacBijector,
    bij.AbstractFwdLogDetJacBijector,
    strict=True,
):
    """Exponential bijector: y = exp(x)."""
    
    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def forward_and_log_det(self, x: PyTree):
        y = jnp.exp(x)
        return y, x

    def inverse_and_log_det(self, y: PyTree):
        x = jnp.log(y)
        return x, -jnp.log(y)

    def same_as(self, other) -> bool:
        return isinstance(other, Exp)
    
    
class Transpose(bij.AbstractBijector):
    """
    Safely swaps the last two axes.
    Maps (..., nfreq, n_features) <-> (..., n_features, nfreq)
    """
    def forward_and_log_det(self, x: jnp.ndarray):
        y = jnp.swapaxes(x, -1, -2)
        return y, jnp.zeros_like(x[..., 0, 0])

    def inverse_and_log_det(self, y: jnp.ndarray):
        x = jnp.swapaxes(y, -1, -2)
        return x, jnp.zeros_like(y[..., 0, 0])    
    
    
class RealToComplex(bij.AbstractBijector):
    """Maps R^2 [real, imag] to a Complex array."""
    def forward_and_log_det(self, x: jnp.ndarray):
        y = jax.lax.complex(x[..., 0], x[..., 1])
        return y, jnp.zeros_like(x[..., 0])

    def inverse_and_log_det(self, y: jnp.ndarray):
        x = jnp.stack([jnp.real(y), jnp.imag(y)], axis=-1)
        return x, jnp.zeros_like(jnp.real(y))


class Rotate(bij.AbstractBijector):
    """Rotates an R^2 coordinate space by a given angle."""
    angle: jnp.ndarray

    def forward_and_log_det(self, x: jnp.ndarray):
        c, s = jnp.cos(self.angle), jnp.sin(self.angle)
        x0, x1 = x[..., 0], x[..., 1]
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c
        return jnp.stack([y0, y1], axis=-1), jnp.zeros_like(x0)

    def inverse_and_log_det(self, y: jnp.ndarray):
        c, s = jnp.cos(self.angle), jnp.sin(self.angle)
        y0, y1 = y[..., 0], y[..., 1]
        # Inverse rotation (transpose)
        x0 = y0 * c + y1 * s
        x1 = -y0 * s + y1 * c
        return jnp.stack([x0, x1], axis=-1), jnp.zeros_like(y0)


class LogPolarToComplex(bij.AbstractBijector):
    """
    Maps R^2 [log_magnitude, phase] to a Complex array.
    Mathematically: y = exp(log_mag) * exp(i * phase)
    """
    def forward_and_log_det(self, x: jnp.ndarray):
        log_mag, phase = x[..., 0], x[..., 1]
        mag = jnp.exp(log_mag)
        y_real = mag * jnp.cos(phase)
        y_imag = mag * jnp.sin(phase)
        y = jax.lax.complex(y_real, y_imag)
        
        # The Jacobian determinant of this transformation is exactly 2 * log_mag
        return y, 2.0 * log_mag

    def inverse_and_log_det(self, y: jnp.ndarray):
        log_mag = jnp.log(jnp.abs(y) + 1e-12)
        phase = jnp.angle(y)
        x = jnp.stack([log_mag, phase], axis=-1)
        return x, -2.0 * log_mag