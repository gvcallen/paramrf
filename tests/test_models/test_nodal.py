# tests/test_models/test_nodal.py
import pytest
import numpy as np
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.rf import s2y

from pmrf.models import (
    Model, GroundLifted, GroundExposed, Shunt, 
    Resistor, Short, Open, Inductor
)
from pmrf.models.composite.nodal import CoupledOnePorts, CoupledTwoPorts


@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

# ---------------------------------------------------------
# GroundLifted Tests
# ---------------------------------------------------------

def test_ground_lifted(basic_freq):
    """
    Test lifting the ground of an N-port network.
    The signal paths should map cleanly to the even indices.
    """
    base_model = Resistor(R=50.0) # 2-port network
    lifted_model = GroundLifted(lifted=base_model)
    
    s_lifted = lifted_model.s(basic_freq)
    
    # A grounded 2-port becomes a 4-port (2 signals, 2 returns)
    assert s_lifted.shape == (5, 4, 4)
    assert not jnp.any(jnp.isnan(s_lifted))
    
    # The signal matrix (even ports 0 and 2) should exactly equal the original matrix
    s_base = base_model.s(basic_freq)
    assert jnp.allclose(s_lifted[:, 0::2, 0::2], s_base)

# ---------------------------------------------------------
# GroundExposed Tests
# ---------------------------------------------------------

def test_ground_exposed_iam_property(basic_freq):
    """
    Test exposing the ground as the N+1 port.
    By definition of the Indefinite Admittance Matrix (IAM), the sum of 
    all rows and all columns in the resulting Y-matrix must equal zero.
    """
    base_model = Resistor(R=50.0) # 2-port network
    exposed_model = GroundExposed(exposed=base_model)
    
    s_exposed = exposed_model.s(basic_freq)
    
    # A grounded 2-port becomes a 3-port
    assert s_exposed.shape == (5, 3, 3)
    assert not jnp.any(jnp.isnan(s_exposed))
    
    # Convert back to Y parameters to check the IAM math
    y_exposed = s2y(s_exposed, z0=50.0)
    
    # Sum across the columns (axis=-1) for each row should be effectively zero
    row_sums = jnp.sum(y_exposed, axis=-1)
    assert jnp.allclose(row_sums, 0.0 + 0.0j, atol=1e-12)
    
    # Sum across the rows (axis=-2) for each column should be effectively zero
    col_sums = jnp.sum(y_exposed, axis=-2)
    assert jnp.allclose(col_sums, 0.0 + 0.0j, atol=1e-12)

# ---------------------------------------------------------
# Shunt Tests
# ---------------------------------------------------------

def test_shunt_open_circuit(basic_freq):
    """
    Test shunting an Open circuit (Gamma = 1).
    Placing an open circuit in parallel with a line should have no effect.
    """
    open_model = Open()
    shunt_open = Shunt(shunt=open_model)
    s_thru = shunt_open.s(basic_freq)
    
    assert s_thru.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s_thru))
    
    # S11 and S22 should be 0 (no reflection)
    assert jnp.allclose(s_thru[:, 0, 0], 0.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s_thru[:, 1, 1], 0.0 + 0.0j, atol=1e-6)
    
    # S21 and S12 should be 1 (perfect transmission)
    assert jnp.allclose(s_thru[:, 1, 0], 1.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s_thru[:, 0, 1], 1.0 + 0.0j, atol=1e-6)

def test_shunt_short_circuit(basic_freq):
    """
    Test shunting a Short circuit (Gamma = -1).
    Placing a dead short to ground across a transmission line should 
    fully reflect all power and block all transmission.
    """
    short_model = Short()
    shunt_short = Shunt(shunt=short_model)
    s_block = shunt_short.s(basic_freq)
    
    assert s_block.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s_block))
    
    # S11 and S22 should be -1 (full reflection, out of phase)
    assert jnp.allclose(s_block[:, 0, 0], -1.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s_block[:, 1, 1], -1.0 + 0.0j, atol=1e-6)
    
    # S21 and S12 should be 0 (zero transmission)
    assert jnp.allclose(s_block[:, 1, 0], 0.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s_block[:, 0, 1], 0.0 + 0.0j, atol=1e-6)

def test_shunt_invalid_port_count():
    """Ensure Shunt raises an error if passed a multi-port network."""
    res_model = Resistor(50.0) # 2-port
    
    with pytest.raises(ValueError, match="Shunt requires a 1-port model"):
        Shunt(shunt=res_model)

# ---------------------------------------------------------
# CoupledOnePorts Tests
# ---------------------------------------------------------

class DummyOnePort(Model):
    """A safe, finite 1-port model to isolate matrix math tests."""
    @property
    def nports(self):
        return 1

    def y(self, freq: Frequency) -> jnp.ndarray:
        return jnp.full((freq.npoints, 1, 1), 1.0 + 2.0j, dtype=jnp.complex128)


def test_coupled_one_ports_validation():
    """Ensure matrix validation catches physical impossibilities and shape mismatches."""
    m1, m2 = DummyOnePort(), DummyOnePort()
    m_2port = Resistor(R=50.0)

    with pytest.raises(ValueError, match="requires 1-port models"):
        CoupledOnePorts(coupled=[m1, m_2port], coupling=np.eye(2), method='matrix')

    with pytest.raises(ValueError, match="Coupling matrix must be shape"):
        CoupledOnePorts(coupled=[m1, m2], coupling=np.eye(3), method='matrix')

    with pytest.raises(ValueError, match="symmetric"):
        CoupledOnePorts(coupled=[m1, m2], coupling=np.array([[1.0, 0.5], [0.2, 1.0]]), method='matrix')

    with pytest.raises(ValueError, match="diagonals must be exactly 1.0"):
        CoupledOnePorts(coupled=[m1, m2], coupling=np.array([[0.9, 0.1], [0.1, 0.9]]), method='matrix')

    with pytest.raises(ValueError, match="positive semi-definite"):
        # k_12 > 1.0 forces a negative eigenvalue
        CoupledOnePorts(coupled=[m1, m2], coupling=np.array([[1.0, 1.5], [1.5, 1.0]]), method='matrix')
        
    with pytest.raises(ValueError, match="Duplicate coupling pair"):
        CoupledOnePorts(coupled=[m1, m2], coupling=[(0, 1, 0.5), (1, 0, 0.5)], method='coefficients')

    with pytest.raises(ValueError, match="Unknown method"):
        CoupledOnePorts(coupled=[m1, m2], coupling=[], method='invalid')


def test_coupled_one_ports_zero_coupling_matrix(basic_freq):
    """
    Test that with an identity matrix, the admittances map cleanly 
    onto an isolated block-diagonal matrix.
    """
    m1, m2 = DummyOnePort(), DummyOnePort()
    k_zero = np.eye(2)
    
    coupled = CoupledOnePorts(coupled=[m1, m2], coupling=k_zero, method='matrix')
    y_coupled = coupled.y(basic_freq)
    
    assert y_coupled.shape == (basic_freq.npoints, 2, 2)
    
    # Self-admittances should be unchanged
    assert jnp.allclose(y_coupled[..., 0, 0], m1.y(basic_freq)[..., 0, 0])
    assert jnp.allclose(y_coupled[..., 1, 1], m2.y(basic_freq)[..., 0, 0])
    
    # Mutual admittances should be strictly zero
    assert jnp.allclose(y_coupled[..., 0, 1], 0.0 + 0.0j)
    assert jnp.allclose(y_coupled[..., 1, 0], 0.0 + 0.0j)


def test_coupled_one_ports_coefficients(basic_freq):
    """
    Test that coefficient definitions map correctly to the resulting admittances.
    """
    m1, m2 = DummyOnePort(), DummyOnePort()
    
    # Couple with k=0.5
    coupled = CoupledOnePorts(coupled=[m1, m2], coupling=[(0, 1, 0.5)], method='coefficients')
    
    # A 0.5 coefficient should evaluate symmetrically
    assert jnp.allclose(coupled.coupling_matrix, np.array([[1.0, 0.5], [0.5, 1.0]]))
    
    y_coupled = coupled.y(basic_freq)
    y11 = m1.y(basic_freq)[..., 0, 0]
    
    # Mutual admittance Y_12 = k_12 * sqrt(Y_11 * Y_22)
    expected_mutual = 0.5 * jnp.sqrt(y11 * y11)
    
    assert jnp.allclose(y_coupled[..., 0, 1], expected_mutual)
    assert jnp.allclose(y_coupled[..., 1, 0], expected_mutual)


# ---------------------------------------------------------
# CoupledTwoPorts Tests
# ---------------------------------------------------------

def test_coupled_two_ports_validation():
    """Ensure bounds checking applies correctly to 2-port arrays."""
    m1, m2 = Inductor(L=1e-9), Inductor(L=1e-9)
    m_1port = DummyOnePort()

    with pytest.raises(ValueError, match="requires 2-port models"):
        CoupledTwoPorts(coupled=[m1, m_1port], coupling=np.eye(2), method='matrix')

    with pytest.raises(ValueError, match="symmetric"):
        CoupledTwoPorts(coupled=[m1, m2], coupling=np.array([[1.0, 0.5], [0.2, 1.0]]), method='matrix')
        
    with pytest.raises(ValueError, match="positive semi-definite"):
        CoupledTwoPorts(coupled=[m1, m2], coupling=np.array([[1.0, 1.5], [1.5, 1.0]]), method='matrix')
        
    with pytest.raises(ValueError, match="Duplicate coupling pair"):
        CoupledTwoPorts(coupled=[m1, m2], coupling=[(0, 1, 0.2), (0, 1, 0.5)], method='coefficients')


def test_coupled_two_ports_zero_coupling_matrix(basic_freq):
    """
    Test that with zero off-diagonal coupling, the resulting 2N-port network 
    behaves exactly like N completely isolated 2-port networks.
    """
    ind1 = Inductor(L=1e-9)
    ind2 = Inductor(L=2e-9)
    k_zero = np.eye(2)
    
    coupled = CoupledTwoPorts(coupled=[ind1, ind2], coupling=k_zero, method='matrix')
    y_coupled = coupled.y(basic_freq)
    
    # 2 models * 2 ports each = 4-port network matrix
    assert y_coupled.shape == (basic_freq.npoints, 4, 4)
    
    y1 = ind1.y(basic_freq)
    y2 = ind2.y(basic_freq)
    
    # Model 1 occupies ports 0 and 1
    assert jnp.allclose(y_coupled[..., 0:2, 0:2], y1)
    
    # Model 2 occupies ports 2 and 3
    assert jnp.allclose(y_coupled[..., 2:4, 2:4], y2)
    
    # Isolation cross-terms between Model 1 and Model 2 must be strictly zero
    assert jnp.allclose(y_coupled[..., 0:2, 2:4], 0.0 + 0.0j)
    assert jnp.allclose(y_coupled[..., 2:4, 0:2], 0.0 + 0.0j)


def test_coupled_two_ports_coefficients():
    """Verify that the constructor handles the coefficients method smoothly."""
    ind1 = Inductor(L=1e-9)
    ind2 = Inductor(L=2e-9)
    
    coupled = CoupledTwoPorts(coupled=[ind1, ind2], coupling=[(0, 1, 0.1)], method='coefficients')
    assert jnp.allclose(coupled.coupling_matrix, np.array([[1.0, 0.1], [0.1, 1.0]]))