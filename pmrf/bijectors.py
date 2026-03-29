"""
Additional bijectors not present in distreqx.
"""
import jax.numpy as jnp
from jaxtyping import PyTree, Array
from typing import Callable

import jax.numpy as jnp
import distreqx.bijectors as bij
import distreqx.distributions as dist


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

class Invert(bij.AbstractFwdLogDetJacBijector, bij.AbstractInvLogDetJacBijector, strict=True):
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
        if type(other) is Invert:
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

class ProbabilityIntegralTransform(
    bij.AbstractFowardInverseBijector,
    bij.AbstractInvLogDetJacBijector,
    bij.AbstractFwdLogDetJacBijector,
    strict=True,
):
    """
    A bijector that maps a distribution to the unit hypercube [0, 1] 
    using its Cumulative Distribution Function (CDF).
    
    Requires the distribution to have an Inverse Cumulative Distribution Function (ICDF).
    """
    distribution: dist.AbstractDistribution

    # The Jacobian is dependent on the input value (the PDF is not constant)
    _is_constant_jacobian: bool = False
    _is_constant_log_det: bool = False

    def forward_and_log_det(self, x: PyTree) -> tuple[PyTree, PyTree]:
        """Computes y = CDF(x) and log|det J(f)(x)| = log(PDF(x))."""
        y = self.distribution.cdf(x)
        log_det = self.distribution.log_prob(x)
        return y, log_det

    def inverse_and_log_det(self, y: PyTree) -> tuple[PyTree, PyTree]:
        """Computes x = iCDF(y) and log|det J(f^{-1})(y)| = -log(PDF(x))."""
        # Clip to prevent evaluating icdf at exact 0 or 1, which causes NaNs/Infs
        eps = jnp.finfo(y.dtype).eps if hasattr(y, 'dtype') else 1e-7
        y_clipped = jnp.clip(y, a_min=eps, a_max=1.0 - eps)
        
        x = self.distribution.icdf(y_clipped)
        log_det = -self.distribution.log_prob(x)
        return x, log_det

    def same_as(self, other) -> bool:
        """Returns True if the other is a CDFBijector with the same distribution."""
        # Using `is` for type checking ensures exact class match (avoids subclass false positives)
        return type(other) is ProbabilityIntegralTransform and self.distribution.same_as(other.distribution)