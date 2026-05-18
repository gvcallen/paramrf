"""
Models that alter the nodal environment of wrapped models.

This includes adding/removing ground, introducing coupling, etc.
"""
from typing import Self

import jax.numpy as jnp
import numpy as np

from pmrf.models import Model
from pmrf.frequency import Frequency
from pmrf.rf.conversions import s2y, y2s

class GroundLifted(Model):
    r"""
    A wrapper that converts an N-port grounded model into a 2N-port ungrounded model.

    The inner component's signal paths map to the even ports (0, 2, 4, ..., 2N-2).
    The inner component's ground is lifted and connected to the odd ports (1, 3, 5, ..., 2N-1),
    forming an isolated common-return star node.

    Parameters
    ----------
    model : Model
        The inner N-port model to be wrapped and lifted from the global ground.

    Reference
    ----------------------
    Constructs a $2N \times 2N$ floating S-matrix by superimposing the signal S-matrix 
    onto the even ports and a parallel star-node (common return) S-matrix onto the odd ports.
    """
    #: The inner N-port model to be wrapped.
    model: Model

    def s(self, freq: Frequency) -> jnp.ndarray:
        n = self.model.nports

        # TODO currently we do not support mixing internal characteristic impedances.
        # We therefore could broadcast self.z0, though this code is just left for reference
        if jnp.isscalar(self.z0):
            inner_z0 = self.z0
            z_ret = jnp.full((freq.npoints, n), self.z0, dtype=jnp.complex128)
        else:
            if self.z0.shape[-1] >= 2 * n:
                inner_z0 = self.z0[..., 0:2*n:2]
                z_ret = self.z0[..., 1:2*n:2]
            else:
                inner_z0 = jnp.repeat(self.z0[..., 0:1], n, axis=-1)
                z_ret = jnp.repeat(
                    self.z0[..., 1:2] if self.z0.shape[-1] > 1 else self.z0[..., 0:1], 
                    n, axis=-1
                )

        s_inner = self.model.s(freq)

        y_ret = 1.0 / z_ret
        r_ret = z_ret.real
        y_tot = jnp.sum(y_ret, axis=-1, keepdims=True)

        term = jnp.sqrt(r_ret) * y_ret
        s_ret = 2.0 * jnp.einsum('...i,...j->...ij', term, term)
        s_ret = s_ret / y_tot[..., jnp.newaxis]

        diag_correction = jnp.conj(z_ret) / z_ret
        i = jnp.arange(n)
        s_ret = s_ret.at[..., i, i].add(-diag_correction)

        s_out = jnp.zeros(s_inner.shape[:-2] + (2 * n, 2 * n), dtype=jnp.complex128)
        
        s_out = s_out.at[..., 0::2, 0::2].set(s_inner)
        s_out = s_out.at[..., 1::2, 1::2].set(s_ret)

        return s_out
    

class GroundExposed(Model):
    r"""
    A wrapper that converts an N-port grounded model into an (N+1)-port model
    by exposing the global ground as a single, accessible terminal.

    The original signal ports remain at indices 0 to N-1.
    The new exposed ground port is at index N.

    Parameters
    ----------
    model : Model
        The inner N-port model whose ground is to be exposed.

    Reference
    ----------------------
    Uses the Indefinite Admittance Matrix (IAM) transformation. Because the sum of 
    currents entering a subcircuit must be zero, the exposed global node is calculated 
    so that the rows and columns of the expanded Y-matrix sum to zero.
    """
    #: The inner N-port model to be wrapped.
    model: Model

    def s(self, freq: Frequency) -> jnp.ndarray:
        if jnp.isscalar(self.z0):
            z0_inner = self.z0
            z0_new_port = self.z0
        else:
            z0_inner = self.z0[..., :-1]
            z0_new_port = self.z0[..., -1:]

        s_inner = self.model.s(freq)
        y_inner = s2y(s_inner, z0=z0_inner)

        col_sums = jnp.sum(y_inner, axis=-1, keepdims=True)
        row_sums = jnp.sum(y_inner, axis=-2, keepdims=True)
        total_sum = jnp.sum(y_inner, axis=(-2, -1), keepdims=True)

        top_block = jnp.concatenate([y_inner, -col_sums], axis=-1)
        bottom_block = jnp.concatenate([-row_sums, total_sum], axis=-1)
        y_exposed = jnp.concatenate([top_block, bottom_block], axis=-2)

        return y2s(y_exposed, z0=self.z0)
    

class Shunt(Model):
    r"""
    Represents a 1-port network connected in parallel (shunt) across a 2-port line.

    Parameters
    ----------
    model : Model
        The 1-port model to be connected in shunt.

    Reference
    ----------------------
    Maps the reflection coefficient ($\Gamma$ or $S_{11}$) of a 1-port component 
    into a 2-port transmission matrix. Avoids division by zero (e.g., ideal opens/shorts) 
    by directly calculating $S_{11}$ and $S_{21}$ using $S_{11, 2port} = (\Gamma - 1) / (\Gamma + 3)$.
    """
    #: The 1-port model to be connected in shunt.
    model: Model
    
    def __post_init__(self):
        if self.model.nports != 1:
            raise ValueError(f"Shunt requires a 1-port model. Received a {self.model.nports}-port model.")

    def s(self, freq: Frequency) -> jnp.ndarray:
        s_1p = self.model.s(freq)
        gamma = s_1p[:, 0, 0]
        
        denom = gamma + 3.0
        s11 = (gamma - 1.0) / denom
        s21 = 2.0 * (1.0 + gamma) / denom
        
        S_shunt = jnp.array([
            [s11, s21],
            [s21, s11],
        ]).transpose(2, 0, 1)
        
        return S_shunt
    
    
class CoupledOnePorts(Model):
    r"""
    (experimental) Wraps N 1-port models (e.g. inductors) and couples them via a given K-matrix.
    
    Parameters
    ----------
    models : list[Model]
        The sequence of 1-port models to couple.
    k_matrix : jnp.ndarray
        The NxN coupling coefficient matrix. Must be symmetric, have 1.0 on the 
        diagonals, and be positive semi-definite.

    Reference
    ----------------------
    Creates an N-port model where the off-diagonal interactions are defined 
    by the mutual admittance relation: 
    $$ Y_{ij} = k_{ij} \sqrt{Y_{ii} Y_{jj}} $$
    """
    #: The sequence of 1-port models to couple.
    models: list[Model]
    #: The NxN coupling coefficient matrix.
    k_matrix: jnp.ndarray 

    def __post_init__(self):
        for i, m in enumerate(self.models):
            if m.nports != 1:
                raise ValueError(f"CoupledOnePorts requires 1-port models. Model {i} has {m.nports} ports.")
        
        n = len(self.models)
        if self.k_matrix.shape != (n, n):
            raise ValueError(f"k_matrix must be shape ({n}, {n}), got {self.k_matrix.shape}")

        k = np.asarray(self.k_matrix)
        if not np.allclose(k, k.T):
            raise ValueError("k_matrix must be symmetric.")
        if not np.allclose(np.diag(k), 1.0):
            raise ValueError("k_matrix diagonals must be exactly 1.0 (self-coupling).")
        
        eigenvalues = np.linalg.eigvalsh(k)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("k_matrix must be positive semi-definite to represent a physical system.")

    def y(self, freq: Frequency) -> jnp.ndarray:
        n = len(self.models)
        
        y_diags = []
        for m in self.models:
            y_i = m.y(freq) 
            y_diags.append(y_i[..., 0, 0])
            
        y_diag = jnp.stack(y_diags, axis=-1)
        
        y_outer = y_diag[..., :, jnp.newaxis] * y_diag[..., jnp.newaxis, :]
        y_coupled = self.k_matrix * jnp.sqrt(y_outer)
        
        i = jnp.arange(n)
        y_coupled = y_coupled.at[..., i, i].set(y_diag)
        
        return y_coupled
    
    @classmethod
    def from_couplings(cls, models: list[Model], couplings: list[tuple[int, int, float]], **kwargs) -> Self:
        """
        Builds of model of coupled one-ports from a list of couplings coefficients between them.
        
        Parameters:
        - models: The models to be coupled.
        - defined_couplings: A list of tuples (model_i, model_j, k_factor).
        """
        seen = set()
        
        n_components = len(models)
        K = np.eye(n_components)
        
        for i, j, k_val in couplings:
            if (i, j) in seen:
                raise Exception(f"Same coupling pairs passed twice. Indices {i} and {j}")
            seen.add((i, j))
            
            K[i, j] = k_val
            K[j, i] = k_val
            
        return CoupledOnePorts(models, K, **kwargs)    


class CoupledTwoPorts(Model):
    r"""
    (experimental) Wraps N 2-port models (e.g., Inductors) and couples them via a given K-matrix.
    
    Returns a 2N-port model where Model 1 occupies ports (0, 1), 
    Model 2 occupies ports (2, 3), and so on.

    Parameters
    ----------
    models : list[Model]
        The sequence of 2-port series models to couple.
    k_matrix : jnp.ndarray
        The NxN coupling coefficient matrix. Must be symmetric, have 1.0 on the 
        diagonals, and be positive semi-definite.

    Reference
    ----------------------
    Uses Modified Nodal Analysis (MNA). Extracts the branch impedance ($Z_b$) for each 
    component, creates a mutually coupled branch matrix $Z_{ij} = k_{ij} \sqrt{Z_{ii} Z_{jj}}$, 
    and translates it to a $2N \times 2N$ nodal admittance matrix using an incidence matrix ($A$):
    $$ Y_{nodal} = A Z_b^{-1} A^T $$
    """
    #: The sequence of 2-port series models to couple.
    models: list[Model]
    #: The NxN coupling coefficient matrix (k).
    k_matrix: jnp.ndarray 

    def __post_init__(self):
        for i, m in enumerate(self.models):
            if m.nports != 2:
                raise ValueError(f"CoupledTwoPorts requires 2-port models. Model {i} has {m.nports} ports.")
        
        n = len(self.models)
        if self.k_matrix.shape != (n, n):
            raise ValueError(f"k_matrix must be shape ({n}, {n}), got {self.k_matrix.shape}")

        k = np.asarray(self.k_matrix)
        if not np.allclose(k, k.T):
            raise ValueError("k_matrix must be symmetric.")
        if not np.allclose(np.diag(k), 1.0):
            raise ValueError("k_matrix diagonals must be exactly 1.0 (self-coupling).")
        
        eigenvalues = np.linalg.eigvalsh(k)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("k_matrix must be positive semi-definite to represent a physical system.")

    def y(self, freq: Frequency) -> jnp.ndarray:
        n = len(self.models)
        
        z_branch_list = []
        for m in self.models:
            y_i = m.y(freq)
            z_series = -1.0 / y_i[..., 0, 1]
            z_branch_list.append(z_series)
            
        z_branch = jnp.stack(z_branch_list, axis=-1)
        
        z_outer = z_branch[..., :, jnp.newaxis] * z_branch[..., jnp.newaxis, :]
        z_b_matrix = self.k_matrix * jnp.sqrt(z_outer)
        
        i = jnp.arange(n)
        z_b_matrix = z_b_matrix.at[..., i, i].set(z_branch)
        
        y_b_matrix = jnp.linalg.inv(z_b_matrix)
        
        A = jnp.zeros((2 * n, n), dtype=jnp.float64)
        A = A.at[0::2, :].set(jnp.eye(n))
        A = A.at[1::2, :].set(-jnp.eye(n))
        
        y_nodal = jnp.einsum('pi,...ij,qj->...pq', A, y_b_matrix, A)
        
        return y_nodal
    
    @classmethod
    def from_couplings(cls, models: list[Model], couplings: list[tuple[int, int, float]], **kwargs) -> Self:
        """
        Builds of model of coupled two-ports from a list of couplings coefficients between them.
        
        Parameters:
        - models: The models to be coupled.
        - defined_couplings: A list of tuples (model_i, model_j, k_factor).
        """
        seen = set()
        
        n_components = len(models)
        K = np.eye(n_components)
        
        for i, j, k_val in couplings:
            if (i, j) in seen:
                raise Exception(f"Same coupling pairs passed twice. Indices {i} and {j}")
            seen.add((i, j))
            
            K[i, j] = k_val
            K[j, i] = k_val
            
        return CoupledTwoPorts(models, K, **kwargs)