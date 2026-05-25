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
    """
    #: The inner N-port model to be wrapped.
    lifted: Model

    def y(self, freq: Frequency) -> jnp.ndarray:
        n = self.lifted.nports
        
        # Get the intrinsic admittance matrix
        y_inner = self.lifted.y(freq)

        # Use the IAM to expose the ground as a single (N+1) port 
        col_sums = jnp.sum(y_inner, axis=-1, keepdims=True)
        row_sums = jnp.sum(y_inner, axis=-2, keepdims=True)
        total_sum = jnp.sum(y_inner, axis=(-2, -1), keepdims=True)

        top_block = jnp.concatenate([y_inner, -col_sums], axis=-1)
        bottom_block = jnp.concatenate([-row_sums, total_sum], axis=-1)
        y_exposed = jnp.concatenate([top_block, bottom_block], axis=-2)
        
        # Create an incidence matrix A of shape (2N, N+1)
        A = jnp.zeros((2 * n, n + 1), dtype=jnp.complex128)
        
        # Map even ports (0, 2, 4...) to the original signal ports (0, 1, 2... N-1)
        A = A.at[0::2, 0:n].set(jnp.eye(n))
        
        # Map odd ports (1, 3, 5...) all to the single exposed ground port (index N)
        A = A.at[1::2, n].set(1.0)
        
        # Map the admittances: Y_out = A * Y_exposed * A^T
        y_out = jnp.einsum('pi,...ij,qj->...pq', A, y_exposed, A)
        
        return y_out
    

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
        
        b_diag = jnp.imag(y_diag)
        b_outer = b_diag[..., :, jnp.newaxis] * b_diag[..., jnp.newaxis, :]
        
        k_mat = self.coupling_matrix
        y_mutual = 1j * k_mat * jnp.sqrt(b_outer + 0j)
        
        y_coupled = y_mutual
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
        y_shunt1_list = []
        y_shunt2_list = []
        
        for m in self.coupled:
            y_i = m.y(freq)
            
            # Decompose the Pi-Network
            # Series branch
            y_series = -y_i[..., 0, 1]
            z_series = 1.0 / y_series
            z_branch_list.append(z_series)
            
            # Shunt branches to ground
            y_shunt1_list.append(y_i[..., 0, 0] + y_i[..., 0, 1])
            y_shunt2_list.append(y_i[..., 1, 1] + y_i[..., 1, 0])
            
        z_branch = jnp.stack(z_branch_list, axis=-1)
        y_p1 = jnp.stack(y_shunt1_list, axis=-1)
        y_p2 = jnp.stack(y_shunt2_list, axis=-1)
        
        # Couple the series branches exactly as before
        x_branch = jnp.imag(z_branch)
        x_outer = x_branch[..., :, jnp.newaxis] * x_branch[..., jnp.newaxis, :]
        
        k_mat = self.coupling_matrix
        z_mutual = 1j * k_mat * jnp.sqrt(x_outer + 0j)
        
        z_b_matrix = z_mutual
        i = jnp.arange(n)
        z_b_matrix = z_b_matrix.at[..., i, i].set(z_branch)
        
        y_b_matrix = jnp.linalg.inv(z_b_matrix)
        
        A = jnp.zeros((2 * n, n), dtype=jnp.float64)
        A = A.at[0::2, :].set(jnp.eye(n))
        A = A.at[1::2, :].set(-jnp.eye(n))
        
        y_nodal = jnp.einsum('pi,...ij,qj->...pq', A, y_b_matrix, A)
        
        # Glue the substrate/shunt parasitics back onto the final nodes
        # Model 'm' has Port 1 at index 2*m, and Port 2 at index 2*m + 1
        even_indices = jnp.arange(0, 2 * n, 2)
        odd_indices = jnp.arange(1, 2 * n, 2)
        
        y_nodal = y_nodal.at[..., even_indices, even_indices].add(y_p1)
        y_nodal = y_nodal.at[..., odd_indices, odd_indices].add(y_p2)
        
        return y_nodal