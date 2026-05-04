"""
Rational approximation surrogate models.
"""

import jax.numpy as jnp
from parax import Param

from pmrf.core import Frequency
from pmrf.models.adapters.base import SingleProperty

class PolynomialRatio(SingleProperty):
    """
    A general N-port model defined by a ratio of polynomials H(s) = A(s) / B(s) 
    where s = j*w.
    """
    #: Numerator coefficients. The first dimension is the polynomial degree.
    #: Shape can be (degree,), (degree, N, N), etc. 
    #: Coefficients are in increasing order of degree (index 0 is constant term).
    a: Param = None
    
    #: Denominator coefficients. The first dimension is the polynomial degree.
    #: Shape can be (degree,), (degree, N, N), etc.
    #: Coefficients are in increasing order of degree.
    b: Param = None

    def output(self, freq: Frequency) -> jnp.ndarray:
        s_cpx = 1j * freq.w
        
        # Ensure parameters are at least 1D so we can extract the degree dimension
        a = jnp.atleast_1d(self.a)
        b = jnp.atleast_1d(self.b)
        
        deg_a = a.shape[0]
        deg_b = b.shape[0]
        
        # Compute powers of s: shape becomes (F, degree)
        S_pow_a = s_cpx[:, None] ** jnp.arange(deg_a)
        S_pow_b = s_cpx[:, None] ** jnp.arange(deg_b)
        
        # tensordot multiplies the degree dimension (axis 1 of S_pow, axis 0 of coeffs).
        # If `a` is shape (degree_a, N, N), A becomes (F, N, N).
        # If `a` is shape (degree_a,), A becomes (F,).
        A = jnp.tensordot(S_pow_a, a, axes=(1, 0))
        B = jnp.tensordot(S_pow_b, b, axes=(1, 0))
        
        # Element-wise division. This broadcasts naturally if B is a shared 
        # scalar polynomial (F,) and A is a matrix polynomial (F, N, N).
        H = A / B
        
        # The framework expects port matrices of shape (F, N, N). 
        # If 1D coefficients were provided, H will be (F,), so we expand to (F, 1, 1).
        if jnp.ndim(H) == 1:
            H = H.reshape(-1, 1, 1)
            
        return H
    
    
class PoleResidue(SingleProperty):
    """
    A general N-port model defined by a pole-residue expansion:
    H(s) = D + sum( R_i / (s - p_i) )
    """
    #: The poles of the system. Shape: (num_poles,)
    poles: Param = None
    
    #: The residues of the system. 
    #: Shape: (num_poles,) for scalars, or (num_poles, N, N) for port matrices.
    residues: Param = None
    
    #: Optional direct feedthrough matrix (constant term D). 
    #: Shape: () or (N, N).
    d: Param = None

    def output(self, freq: Frequency) -> jnp.ndarray:
        s_cpx = 1j * freq.w
        
        # Ensure arrays are at least 1D
        p = jnp.atleast_1d(self.poles)
        r = jnp.atleast_1d(self.residues)
        
        # If no D matrix is provided, default to 0
        d_val = jnp.asarray(self.d) if self.d is not None else 0.0
        
        # 1. Calculate the denominator terms: (s - p_i)
        # s_cpx is (F,), p is (num_poles,)
        # Broadcasting gives a shape of (F, num_poles)
        denom = s_cpx[:, None] - p[None, :] 
        
        # 2. Invert the denominator
        inv_denom = 1.0 / denom
        
        # 3. Multiply by residues and sum over the poles
        # tensordot sums over axis 1 of inv_denom (num_poles) 
        # and axis 0 of residues (num_poles).
        # If r is (num_poles, N, N), H becomes (F, N, N)
        H = jnp.tensordot(inv_denom, r, axes=(1, 0)) + d_val
        
        # Expand scalar responses to (F, 1, 1) to match framework expectations
        if jnp.ndim(H) == 1:
            H = H.reshape(-1, 1, 1)
            
        return H
    
    
class StateSpace(SingleProperty):
    """
    A general N-port model defined by continuous-time state-space matrices:
    H(s) = C * (sI - A)^-1 * B + D
    """
    #: State matrix (A). Shape: (state_dim, state_dim)
    a: Param = None
    
    #: Input matrix (B). Shape: (state_dim, N)
    b: Param = None
    
    #: Output matrix (C). Shape: (N, state_dim)
    c: Param = None
    
    #: Feedthrough matrix (D). Shape: (N, N)
    d: Param = None

    def output(self, freq: Frequency) -> jnp.ndarray:
        s_cpx = 1j * freq.w
        
        # Ensure matrices are at least 2D
        A = jnp.atleast_2d(self.a)
        B = jnp.atleast_2d(self.b)
        C = jnp.atleast_2d(self.c)
        
        # Default D to zero if not provided
        D = jnp.atleast_2d(self.d) if self.d is not None else 0.0
        
        state_dim = A.shape[0]
        F = s_cpx.shape[0]
        
        # 1. Construct the (sI) matrix for all frequencies
        # Shape becomes (F, state_dim, state_dim)
        I = jnp.eye(state_dim, dtype=s_cpx.dtype)
        sI = s_cpx[:, None, None] * I
        
        # 2. Compute (sI - A)
        # Broadcasting A to match the frequency batch dimension
        sI_minus_A = sI - A[None, :, :]
        
        # 3. Solve the linear system (sI - A) * X = B
        # B needs to be broadcast across the frequency dimension: (F, state_dim, N)
        B_broadcast = jnp.broadcast_to(B[None, :, :], (F, B.shape[0], B.shape[1]))
        
        # jnp.linalg.solve is numerically superior to taking the inverse directly
        # X shape: (F, state_dim, N)
        X = jnp.linalg.solve(sI_minus_A, B_broadcast)
        
        # 4. Compute C * X + D
        # C is (N, state_dim), X is (F, state_dim, N). 
        # Einsum efficiently handles the batched matrix multiplication.
        H = jnp.einsum('ij,fjk->fik', C, X) + D
        
        # The framework expects port matrices of shape (F, N, N).
        # Expand scalar responses to (F, 1, 1).
        if jnp.ndim(H) == 1:
            H = H.reshape(-1, 1, 1)
            
        return H
    
    
class BarycentricRational(SingleProperty):
    """
    A highly numerically stable rational model defined by a barycentric expansion:
    H(s) = [ sum( w_i * f_i / (s - s_i) ) ] / [ sum( w_i / (s - s_i) ) ]
    """
    #: The complex support points (s_i). Shape: (num_points,)
    support_points: Param = None
    
    #: The barycentric weights (w_i). Shape: (num_points,)
    weights: Param = None
    
    #: The function values at the support points (f_i). 
    #: Shape: (num_points,) for scalars, or (num_points, N, N) for port matrices.
    values: Param = None

    def output(self, freq: Frequency) -> jnp.ndarray:
        s_cpx = 1j * freq.w
        
        s_i = jnp.atleast_1d(self.support_points)
        w_i = jnp.atleast_1d(self.weights)
        f_i = jnp.atleast_1d(self.values)
        
        # 1. Calculate the distance to support points: (s - s_i)
        # Shape: (F, num_points)
        denom = s_cpx[:, None] - s_i[None, :]
        
        # Prevent division by zero if evaluating exactly on a support point
        eps = jnp.finfo(s_cpx.dtype).eps
        safe_denom = jnp.where(jnp.abs(denom) < eps, eps, denom)
        
        # 2. Compute the basis functions: w_i / (s - s_i)
        # Shape: (F, num_points)
        basis = w_i[None, :] / safe_denom
        
        # 3. Compute the global denominator: sum( basis )
        # Shape: (F,)
        B = jnp.sum(basis, axis=1)
        
        # 4. Compute the global numerator: sum( basis * f_i )
        # Tensordot sums over the num_points dimension.
        # If f_i is (num_points, N, N), A becomes (F, N, N)
        A = jnp.tensordot(basis, f_i, axes=(1, 0))
        
        # 5. Divide Numerator by Denominator
        # We need to reshape B to broadcast properly against A if A is a matrix
        if jnp.ndim(A) > 1:
            # Reshape B from (F,) to (F, 1, 1) or whatever the ndim requires
            B_expanded = B.reshape(-1, *([1] * (jnp.ndim(A) - 1)))
            H = A / B_expanded
        else:
            H = A / B
            
        # Match framework expectations
        if jnp.ndim(H) == 1:
            H = H.reshape(-1, 1, 1)
            
        return H