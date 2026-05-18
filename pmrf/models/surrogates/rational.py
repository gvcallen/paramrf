"""
Rational approximation surrogate models.
"""

import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models.adapters.base import AbstractSingleProperty
from pmrf.parameters import Param, param

class PolynomialRatio(AbstractSingleProperty):
    """
    (experimental) A general N-port model defined by a ratio of polynomials H(s) = A(s) / B(s) 
    where s = j*w.

    Parameters
    ----------
    a : Param
        Numerator coefficients. The first dimension is the polynomial degree.
        Shape can be (degree,), (degree, N, N), etc. 
        Coefficients are in increasing order of degree (index 0 is constant term).
    b : Param
        Denominator coefficients. The first dimension is the polynomial degree.
        Shape can be (degree,), (degree, N, N), etc.
        Coefficients are in increasing order of degree.
    """
    #: Numerator coefficients
    a: Param = None
    
    #: Denominator coefficients
    b: Param = None

    def primary_matrix(self, freq: Frequency) -> jnp.ndarray:
        s_cpx = 1j * freq.w
        
        a = jnp.atleast_1d(self.a)
        b = jnp.atleast_1d(self.b)
        
        deg_a = a.shape[0]
        deg_b = b.shape[0]
        
        S_pow_a = s_cpx[:, None] ** jnp.arange(deg_a)
        S_pow_b = s_cpx[:, None] ** jnp.arange(deg_b)
        A = jnp.tensordot(S_pow_a, a, axes=(1, 0))
        B = jnp.tensordot(S_pow_b, b, axes=(1, 0))
        
        H = A / B
        
        if jnp.ndim(H) == 1:
            H = H.reshape(-1, 1, 1)
            
        return H
    
    
class PoleResidue(AbstractSingleProperty):
    """
    (experimental) A general N-port model defined by a pole-residue expansion:
    H(s) = D + sum( R_i / (s - p_i) )

    Parameters
    ----------
    poles : Param
        The poles of the system. Shape: (num_poles,)
    residues : Param
        The residues of the system. 
        Shape: (num_poles,) for scalars, or (num_poles, N, N) for port matrices.
    d : Param
        Optional direct feedthrough matrix (constant term D). 
        Shape: () or (N, N).
    """
    #: The poles of the system
    poles: Param = None
    
    #: The residues of the system
    residues: Param = None
    
    #: Optional direct feedthrough matrix
    d: Param = None

    def primary_matrix(self, freq: Frequency) -> jnp.ndarray:
        s_cpx = 1j * freq.w
        
        p = jnp.atleast_1d(self.poles)
        r = jnp.atleast_1d(self.residues)
        
        d_val = jnp.asarray(self.d) if self.d is not None else 0.0
        denom = s_cpx[:, None] - p[None, :] 
        inv_denom = 1.0 / denom
        H = jnp.tensordot(inv_denom, r, axes=(1, 0)) + d_val
        
        if jnp.ndim(H) == 1:
            H = H.reshape(-1, 1, 1)
            
        return H
    
    
class StateSpace(AbstractSingleProperty):
    """
    (experimental) A general N-port model defined by continuous-time state-space matrices:
    H(s) = C * (sI - A)^-1 * B + D

    Parameters
    ----------
    a : Param
        State matrix (A). Shape: (state_dim, state_dim)
    b : Param
        Input matrix (B). Shape: (state_dim, N)
    c : Param
        Output matrix (C). Shape: (N, state_dim)
    d : Param
        Feedthrough matrix (D). Shape: (N, N)
    """
    #: State matrix (A)
    a: Param = None
    
    #: Input matrix (B)
    b: Param = None
    
    #: Output matrix (C)
    c: Param = None
    
    #: Feedthrough matrix (D)
    d: Param = None

    def primary_matrix(self, freq: Frequency) -> jnp.ndarray:
        s_cpx = 1j * freq.w
        
        A = jnp.atleast_2d(self.a)
        B = jnp.atleast_2d(self.b)
        C = jnp.atleast_2d(self.c)
        
        D = jnp.atleast_2d(self.d) if self.d is not None else 0.0
        
        state_dim = A.shape[0]
        F = s_cpx.shape[0]
        
        I = jnp.eye(state_dim, dtype=s_cpx.dtype)
        sI = s_cpx[:, None, None] * I
        sI_minus_A = sI - A[None, :, :]
        B_broadcast = jnp.broadcast_to(B[None, :, :], (F, B.shape[0], B.shape[1]))
        
        X = jnp.linalg.solve(sI_minus_A, B_broadcast)
        H = jnp.einsum('ij,fjk->fik', C, X) + D
        
        if jnp.ndim(H) == 1:
            H = H.reshape(-1, 1, 1)
            
        return H
    
    
class BarycentricRational(AbstractSingleProperty):
    """
    (experimental) A highly numerically stable rational model defined by a barycentric expansion:
    H(s) = [ sum( w_i * f_i / (s - s_i) ) ] / [ sum( w_i / (s - s_i) ) ]

    Parameters
    ----------
    support_points : Param
        The complex support points (s_i). Shape: (num_points,)
    weights : Param
        The barycentric weights (w_i). Shape: (num_points,)
    values : Param
        The function values at the support points (f_i). 
        Shape: (num_points,) for scalars, or (num_points, N, N) for port matrices.
    """
    #: Complex support points
    support_points: Param = None
    
    #: Barycentric weights
    weights: Param = None
    
    #: Function values at support points
    values: Param = None

    def primary_matrix(self, freq: Frequency) -> jnp.ndarray:
        s_cpx = 1j * freq.w
        
        s_i = jnp.atleast_1d(self.support_points)
        w_i = jnp.atleast_1d(self.weights)
        f_i = jnp.atleast_1d(self.values)
        
        denom = s_cpx[:, None] - s_i[None, :]
        
        eps = jnp.finfo(s_cpx.dtype).eps
        safe_denom = jnp.where(jnp.abs(denom) < eps, eps, denom)
        
        basis = w_i[None, :] / safe_denom
        B = jnp.sum(basis, axis=1)
        A = jnp.tensordot(basis, f_i, axes=(1, 0))
        
        if jnp.ndim(A) > 1:
            B_expanded = B.reshape(-1, *([1] * (jnp.ndim(A) - 1)))
            H = A / B_expanded
        else:
            H = A / B
            
        if jnp.ndim(H) == 1:
            H = H.reshape(-1, 1, 1)
            
        return H