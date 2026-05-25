"""
Models that alter the nodal environment of wrapped models.

This includes adding/removing ground, introducing coupling, etc.
"""
from typing import Any

import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.rf import renormalize_s
from pmrf.types import ArrayLike


class GroundLifted(Model):
    r"""
    A wrapper that converts an N-port grounded model into a 2N-port ungrounded model.

    The inner component's signal paths map to the even ports (0, 2, 4, ..., 2N-2).
    The inner component's ground is lifted and connected to the odd ports (1, 3, 5, ..., 2N-1),
    forming an isolated common-return star node.

    Parameters
    ----------
    lifted : Model
        The inner N-port model to be wrapped and lifted from the global ground.

    Reference
    ----------------------
    Constructs a $2N \times 2N$ floating S-matrix by superimposing the signal S-matrix 
    onto the even ports and a parallel star-node (common return) S-matrix onto the odd ports.
    """
    #: The inner N-port model to be wrapped.
    lifted: Model

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        n = self.lifted.nports

        # Evaluate the inner model at a uniform, scalar reference impedance
        z0_eval = 50.0
        s_inner = self.lifted.s(freq, z0=z0_eval)

        # The lifted ground forms an ideal common return node (an N-port parallel junction).
        s_ret = jnp.full((freq.npoints, n, n), 2.0 / n, dtype=jnp.complex128)
        i = jnp.arange(n)
        s_ret = s_ret.at[..., i, i].add(-1.0)

        # Superimpose signal ports (even) and return ports (odd)
        s_out = jnp.zeros((freq.npoints, 2 * n, 2 * n), dtype=jnp.complex128)
        
        s_out = s_out.at[..., 0::2, 0::2].set(s_inner)
        s_out = s_out.at[..., 1::2, 1::2].set(s_ret)

        # Renormalize to the requested z0
        return renormalize_s(s_out, z0_eval, z0, 'power', 'power')
    

class GroundExposed(Model):
    r"""
    A wrapper that converts an N-port grounded model into an (N+1)-port model
    by exposing the global ground as a single, accessible terminal.

    The original signal ports remain at indices 0 to N-1.
    The new exposed ground port is at index N.

    Parameters
    ----------
    exposed : Model
        The inner N-port model whose ground is to be exposed.

    Reference
    ----------------------
    Uses the Indefinite Admittance Matrix (IAM) transformation. Because the sum of 
    currents entering a subcircuit must be zero, the exposed global node is calculated 
    so that the rows and columns of the expanded Y-matrix sum to zero.
    """
    #: The inner N-port model to be wrapped.
    exposed: Model

    def y(self, freq: Frequency) -> jnp.ndarray:
        # Fetch the intrinsic admittance matrix
        y_inner = self.exposed.y(freq)

        col_sums = jnp.sum(y_inner, axis=-1, keepdims=True)
        row_sums = jnp.sum(y_inner, axis=-2, keepdims=True)
        total_sum = jnp.sum(y_inner, axis=(-2, -1), keepdims=True)

        top_block = jnp.concatenate([y_inner, -col_sums], axis=-1)
        bottom_block = jnp.concatenate([-row_sums, total_sum], axis=-1)
        y_exposed = jnp.concatenate([top_block, bottom_block], axis=-2)

        return y_exposed
        

class Shunt(Model):
    r"""
    Represents a 1-port network connected in parallel (shunt) across a 2-port line.

    Parameters
    ----------
    shunt : Model
        The 1-port model to be connected in shunt.

    Reference
    ----------------------
    Maps the reflection coefficient ($\Gamma$ or $S_{11}$) of a 1-port component 
    into a 2-port transmission matrix. Avoids division by zero (e.g., ideal opens/shorts) 
    by directly calculating $S_{11}$ and $S_{21}$ using $S_{11, 2port} = (\Gamma - 1) / (\Gamma + 3)$.
    """
    #: The 1-port model to be connected in shunt.
    shunt: Model
    
    def __post_init__(self):
        if self.shunt.nports != 1:
            raise ValueError(f"Shunt requires a 1-port model. Received a {self.shunt.nports}-port model.")

    def s(self, freq: Frequency, z0: ArrayLike = 50.0) -> jnp.ndarray:
        # Evaluate the 1-port shunt at a common uniform reference
        z0_eval = 50.0
        s_1p = self.shunt.s(freq, z0=z0_eval)
        gamma = s_1p[:, 0, 0]
        
        # Map Gamma into a 2-port transmission matrix at that same reference
        denom = gamma + 3.0
        s11 = (gamma - 1.0) / denom
        s21 = 2.0 * (1.0 + gamma) / denom
        
        S_shunt = jnp.array([
            [s11, s21],
            [s21, s11],
        ]).transpose(2, 0, 1)
        
        # Renormalize to the requested z0 
        return renormalize_s(S_shunt, z0_eval, z0, 'power', 'power')
    
    
import numpy as np
import jax.numpy as jnp
from typing import Any

class CoupledOnePorts(Model):
    r"""
    (experimental) Wraps N 1-port models (e.g. inductors) and couples them via a given K-matrix.
    
    Parameters
    ----------
    coupled : list[Model]
        The sequence of 1-port models to couple.
    coupling : Any
        The coupling definition between the elements. Meaning depends on `method`.
        For fixed coupling, pass Python collections and numpy arrays.
        For variable coupling, pass parameters from `pmrf.parameters`.
    method : str, default='coefficients'
        The meaning of `coupling`. Options are ('coefficients', 'matrix').
        For 'coefficients', must be a list of tuples (model_i, model_j, k_factor).
        For 'matrix', must be an NxN coupling matrix which is symmetric, has 1.0 on the 
        diagonals, and is positive semi-definite. 

    Reference
    ---------
    Creates an N-port model where the off-diagonal interactions are defined 
    by the mutual admittance relation: 
    $$ Y_{ij} = k_{ij} \sqrt{Y_{ii} Y_{jj}} $$
    """
    #: The sequence of 1-port models to couple.
    coupled: list['Model']
    
    #: The coupling definition (list of tuples or array-like matrix).
    coupling: Any
    
    #: The method used to interpret the coupling definition.
    method: str = 'coefficients'

    @property
    def coupling_matrix(self) -> jnp.ndarray:
        """
        Evaluates the coupling definition based on the method and returns the NxN coupling matrix.
        
        Returns
        -------
        jnp.ndarray
            The full, symmetric NxN coupling matrix.
        """
        n = len(self.coupled)
        
        if self.method == 'matrix':
            return jnp.asarray(self.coupling)
            
        elif self.method == 'coefficients':
            k = jnp.eye(n)
            seen = set()
            
            for i, j, k_val in self.coupling:
                if (i, j) in seen or (j, i) in seen:
                    raise ValueError(f"Duplicate coupling pair provided for indices ({i}, {j}).")
                
                seen.add((i, j))
                seen.add((j, i))
                
                # JAX compatible array update
                k = k.at[i, j].set(k_val)
                k = k.at[j, i].set(k_val)
                
            return k
            
        else:
            raise ValueError(f"Unknown method '{self.method}'. Must be 'coefficients' or 'matrix'.")

    def __post_init__(self):
        for i, m in enumerate(self.coupled):
            if m.nports != 1:
                raise ValueError(f"CoupledOnePorts requires 1-port models. Model {i} has {m.nports} ports.")
        
        n = len(self.coupled)
        k = self.coupling_matrix
        
        if k.shape != (n, n):
            raise ValueError(f"Coupling matrix must be shape ({n}, {n}), got {k.shape}")

        if not jnp.allclose(k, k.T):
            raise ValueError("Coupling matrix must be symmetric.")
        if not jnp.allclose(jnp.diag(k), 1.0):
            raise ValueError("Coupling matrix diagonals must be exactly 1.0 (self-coupling).")
        
        eigenvalues = jnp.linalg.eigvalsh(k)
        if jnp.any(eigenvalues < -1e-10):
            raise ValueError("Coupling matrix must be positive semi-definite to represent a physical system.")

    def y(self, freq: 'Frequency') -> jnp.ndarray:
        n = len(self.coupled)
        
        y_diags = []
        for m in self.coupled:
            y_i = m.y(freq) 
            y_diags.append(y_i[..., 0, 0])
            
        y_diag = jnp.stack(y_diags, axis=-1)
        
        y_outer = y_diag[..., :, jnp.newaxis] * y_diag[..., jnp.newaxis, :]
        
        k_mat = self.coupling_matrix
        y_coupled = k_mat * jnp.sqrt(y_outer)
        
        i = jnp.arange(n)
        y_coupled = y_coupled.at[..., i, i].set(y_diag)
        
        return y_coupled


class CoupledTwoPorts(Model):
    r"""
    (experimental) Wraps N 2-port models (e.g., Inductors) and couples them via a given K-matrix.
    
    Returns a 2N-port model where Model 1 occupies ports (0, 1), 
    Model 2 occupies ports (2, 3), and so on.

    Parameters
    ----------
    coupled : list[Model]
        The sequence of 2-port models to couple.
    coupling : Any
        The coupling definition between the elements. Meaning depends on `method`.
        For fixed coupling, pass Python collections and numpy arrays.
        For variable coupling, pass parameters from `pmrf.parameters`.
    method : str, default='coefficients'
        The meaning of `coupling`. Options are ('coefficients', 'matrix').
        For 'coefficients', must be a list of tuples (model_i, model_j, k_factor).
        For 'matrix', must be an NxN coupling matrix which is symmetric, has 1.0 on the 
        diagonals, and is positive semi-definite. 

    Reference
    ---------
    Uses Modified Nodal Analysis (MNA). Extracts the branch impedance ($Z_b$) for each 
    component, creates a mutually coupled branch matrix $Z_{ij} = k_{ij} \sqrt{Z_{ii} Z_{jj}}$, 
    and translates it to a $2N \times 2N$ nodal admittance matrix using an incidence matrix ($A$):
    $$ Y_{nodal} = A Z_b^{-1} A^T $$
    """
    #: The sequence of 2-port series models to couple.
    coupled: list['Model']
    
    #: The coupling definition (list of tuples or array-like matrix).
    coupling: Any
    
    #: The method used to interpret the coupling definition.
    method: str = 'coefficients'

    @property
    def coupling_matrix(self) -> jnp.ndarray:
        """
        Evaluates the coupling definition based on the method and returns the NxN coupling matrix.
        
        Returns
        -------
        jnp.ndarray
            The full, symmetric NxN coupling matrix.
        """
        n = len(self.coupled)
        
        if self.method == 'matrix':
            return jnp.asarray(self.coupling)
            
        elif self.method == 'coefficients':
            k = jnp.eye(n)
            seen = set()
            
            for i, j, k_val in self.coupling:
                if (i, j) in seen or (j, i) in seen:
                    raise ValueError(f"Duplicate coupling pair provided for indices ({i}, {j}).")
                
                seen.add((i, j))
                seen.add((j, i))
                
                # JAX compatible array update
                k = k.at[i, j].set(k_val)
                k = k.at[j, i].set(k_val)
                
            return k
            
        else:
            raise ValueError(f"Unknown method '{self.method}'. Must be 'coefficients' or 'matrix'.")

    def __post_init__(self):
        for i, m in enumerate(self.coupled):
            if m.nports != 2:
                raise ValueError(f"CoupledTwoPorts requires 2-port models. Model {i} has {m.nports} ports.")
        
        n = len(self.coupled)
        k = self.coupling_matrix
        
        if k.shape != (n, n):
            raise ValueError(f"Coupling matrix must be shape ({n}, {n}), got {k.shape}")

        if not jnp.allclose(k, k.T):
            raise ValueError("Coupling matrix must be symmetric.")
        if not jnp.allclose(jnp.diag(k), 1.0):
            raise ValueError("Coupling matrix diagonals must be exactly 1.0 (self-coupling).")
        
        eigenvalues = jnp.linalg.eigvalsh(k)
        if jnp.any(eigenvalues < -1e-10):
            raise ValueError("Coupling matrix must be positive semi-definite to represent a physical system.")

    def y(self, freq: 'Frequency') -> jnp.ndarray:
        n = len(self.coupled)
        
        z_branch_list = []
        for m in self.coupled:
            y_i = m.y(freq)
            z_series = -1.0 / y_i[..., 0, 1]
            z_branch_list.append(z_series)
            
        z_branch = jnp.stack(z_branch_list, axis=-1)
        
        z_outer = z_branch[..., :, jnp.newaxis] * z_branch[..., jnp.newaxis, :]
        
        k_mat = self.coupling_matrix
        z_b_matrix = k_mat * jnp.sqrt(z_outer)
        
        i = jnp.arange(n)
        z_b_matrix = z_b_matrix.at[..., i, i].set(z_branch)
        
        y_b_matrix = jnp.linalg.inv(z_b_matrix)
        
        A = jnp.zeros((2 * n, n), dtype=jnp.float64)
        A = A.at[0::2, :].set(jnp.eye(n))
        A = A.at[1::2, :].set(-jnp.eye(n))
        
        y_nodal = jnp.einsum('pi,...ij,qj->...pq', A, y_b_matrix, A)
        
        return y_nodal