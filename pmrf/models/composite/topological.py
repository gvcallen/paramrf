import jax.numpy as jnp

from pmrf.models import Model
from pmrf.frequency import Frequency

class PiTopology(Model):
    """
    (experimental) A 2-port Pi-network built from nested sub-models.

    The models placed in the branches are treated as 1-port admittances. 
    If a 2-port model is passed, its input admittance (Y11) is extracted, 
    effectively treating it as if its second port is shorted to ground.

    Parameters
    ----------
    M1 : Model
        The model representing the first shunt branch.
    M2 : Model
        The model representing the second shunt branch.
    M3 : Model
        The model representing the series branch connecting the two shunt branches.
    """
    M1: Model
    M2: Model
    M3: Model

    def __post_init__(self):
        for name, m in [('M1', self.M1), ('M2', self.M2), ('M3', self.M3)]:
            if m.nports not in (1, 2):
                raise ValueError(f"PiTopology {name} must be a 1-port or 2-port model, got {m.nports}-port.")

    def a(self, freq: Frequency) -> jnp.ndarray:
        # Extract the Y-matrices for each child model at the given frequencies
        Y_mat1 = self.M1.y(freq)
        Y_mat2 = self.M2.y(freq)
        Y_mat3 = self.M3.y(freq)

        # Extract the Y11 component (index 0,0) to use as the branch admittance.
        Y1 = Y_mat1[:, 0, 0]
        Y2 = Y_mat2[:, 0, 0]
        Y3 = Y_mat3[:, 0, 0]

        # Apply the stable Pi-network math
        Y3_safe = jnp.where(Y3 == 0.0, jnp.finfo(float).eps, Y3)

        A = 1 + Y2 / Y3_safe
        B = 1 / Y3_safe
        C = Y1 + Y2 + (Y1 * Y2) / Y3_safe
        D = 1 + Y1 / Y3_safe

        return jnp.array([
            [A, B],
            [C, D],
        ]).transpose(2, 0, 1)


class TTopology(Model):
    """
    (experimental) A 2-port Tee-network built from nested sub-models.

    The models placed in the branches are treated as 1-port impedances. 
    If a 2-port model is passed, its input impedance (Z11) is extracted, 
    effectively treating it as if its second port is open-circuited.

    Parameters
    ----------
    M1 : Model
        The model representing the first series branch.
    M2 : Model
        The model representing the second series branch.
    M3 : Model
        The model representing the shunt branch to ground.
    """
    M1: Model
    M2: Model
    M3: Model

    def __post_init__(self):
        for name, m in [('M1', self.M1), ('M2', self.M2), ('M3', self.M3)]:
            if m.nports not in (1, 2):
                raise ValueError(f"TTopology {name} must be a 1-port or 2-port model, got {m.nports}-port.")

    def a(self, freq: Frequency) -> jnp.ndarray:
        # Extract the Z-matrices for each child model
        Z_mat1 = self.M1.z(freq)
        Z_mat2 = self.M2.z(freq)
        Z_mat3 = self.M3.z(freq)

        # Extract the Z11 component (index 0,0) to use as the branch impedance.
        Z1 = Z_mat1[:, 0, 0]
        Z2 = Z_mat2[:, 0, 0]
        Z3 = Z_mat3[:, 0, 0]

        # Apply the stable T-network math
        Z3_safe = jnp.where(Z3 == 0.0, jnp.finfo(float).eps, Z3)

        A = 1 + Z1 / Z3_safe
        B = Z1 + Z2 + (Z1 * Z2) / Z3_safe
        C = 1 / Z3_safe
        D = 1 + Z2 / Z3_safe

        return jnp.array([
            [A, B],
            [C, D],
        ]).transpose(2, 0, 1)


class LTopology(Model):
    """
    (experimental) A 2-port L-section network built from nested sub-models.
    
    Uses a series model followed by a shunt model. The series branch extracts 
    input impedance (Z11), while the shunt branch extracts input admittance (Y11).

    Parameters
    ----------
    M1 : Model
        The model representing the series branch.
    M2 : Model
        The model representing the shunt branch.
    """
    M1: Model
    M2: Model

    def __post_init__(self):
        for name, m in [('M1', self.M1), ('M2', self.M2)]:
            if m.nports not in (1, 2):
                raise ValueError(f"LTopology {name} must be a 1-port or 2-port model, got {m.nports}-port.")

    def a(self, freq: Frequency) -> jnp.ndarray:
        # Extract Z-matrix for the series branch and Y-matrix for the shunt branch
        Z_mat = self.M1.z(freq)
        Y_mat = self.M2.y(freq)

        # Extract Z11 and Y11 respectively
        Z = Z_mat[:, 0, 0]
        Y = Y_mat[:, 0, 0]

        ones = jnp.ones_like(Z)

        # Apply the perfectly stable L-section math
        A = 1 + Z * Y
        B = Z
        C = Y
        D = ones

        return jnp.array([
            [A, B],
            [C, D],
        ]).transpose(2, 0, 1)