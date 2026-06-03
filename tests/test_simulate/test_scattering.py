"""tests/test_scattering_solvers.py"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Adjust imports based on your project structure
from pmrf.simulate.solvers.scattering import (
    SequentialScatteringCascader,
    ScatteringTerminator,
    GlobalScatteringReducer,
    SequentialScatteringReducer,
    HierarchicalScatteringReducer
)
from pmrf.simulate.base import PortRepresentation


def test_sequential_cascader_matched_attenuators():
    """
    Cascade two ideal 2-port attenuators (S11=S22=0, S21=S12=0.5).
    Result should be a single attenuator with S21=S12=0.25.
    """
    solver = SequentialScatteringCascader()
    
    S_ideal = jnp.array([[0.0, 0.5], 
                         [0.5, 0.0]], dtype=jnp.complex128)
    z0_ideal = jnp.array([50.0, 50.0])
    
    s_stacked = jnp.stack([S_ideal, S_ideal])
    z0_stacked = jnp.stack([z0_ideal, z0_ideal])
    
    result = solver.run(s_stacked, z0_stacked)
    
    expected_s = jnp.array([[0.0, 0.25], 
                            [0.25, 0.0]], dtype=jnp.complex128)
    
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)
    np.testing.assert_allclose(result.z0, z0_ideal, atol=1e-7)


def test_scattering_terminator_dimension_fix():
    """
    Terminate a 3-port network with a 1-port load.
    Ensures slicing uses surviving ports (K).
    """
    solver = ScatteringTerminator()
    
    S_3port = jnp.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.3, 0.4, 0.1]
    ], dtype=jnp.complex128)
    z0_3port = jnp.array([50.0, 50.0, 50.0])
    
    S_load = jnp.array([[0.0]], dtype=jnp.complex128)
    z0_load = jnp.array([50.0])
    
    result = solver.run(S_3port, z0_3port, S_load, z0_load)
    
    expected_s = jnp.array([
        [0.1, 0.2],
        [0.2, 0.1]
    ], dtype=jnp.complex128)
    
    assert result.s.shape == (2, 2)
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)


@pytest.mark.parametrize("SolverClass", [
    GlobalScatteringReducer, 
    SequentialScatteringReducer, 
    HierarchicalScatteringReducer
])
def test_reducers_simple_connection(SolverClass):
    """
    Test generic reducers on a basic 2-block connection topology.
    Blocks: Two perfectly matched lines with 0.5 transmission.
    """
    solver = SolverClass()
    
    S_block1 = jnp.array([[0.0, 0.5], [0.5, 0.0]], dtype=jnp.complex128)
    S_block2 = jnp.array([[0.0, 0.5], [0.5, 0.0]], dtype=jnp.complex128)
    
    s_bd = jax.scipy.linalg.block_diag(S_block1, S_block2)
    z0_ports = jnp.array([50.0, 50.0, 50.0, 50.0])
    
    # Topology: Port 0 (ext), Port 1 (int) -> connected to -> Port 2 (int), Port 3 (ext)
    topology = PortRepresentation(
        ext_idx=np.array([0, 3]),
        int_idx=np.array([1, 2]),
        port_to_net_map=np.array([0, 1, 1, 2])
    )
            
    result = solver.run(s_bd, z0_ports, topology)
    
    expected_s = jnp.array([[0.0, 0.25], 
                            [0.25, 0.0]], dtype=jnp.complex128)
    
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)
    np.testing.assert_allclose(result.z0, jnp.array([50.0, 50.0]), atol=1e-7)


def test_hierarchical_reducer_complex_chain():
    """
    Tests HierarchicalScatteringReducer with 5 internal nets (pairs).
    """
    solver = HierarchicalScatteringReducer()

    # 6 components, each a 2-port with S11=S22=0, S21=S12=0.5
    S_block = jnp.array([[0.0, 0.5], [0.5, 0.0]], dtype=jnp.complex128)
    s_bd = jax.scipy.linalg.block_diag(*[S_block for _ in range(6)])
    z0_ports = jnp.full(12, 50.0)

    # 12 total ports. Chain connection.
    # Ext: Port 0 and 11
    # Int: Ports 1-10 paired up sequentially
    topology = PortRepresentation(
        ext_idx=np.array([0, 11]),
        int_idx=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        port_to_net_map=np.array([
            0,        # Port 0 -> Net 0 (ext)
            1, 1,     # Port 1, 2 -> Net 1
            2, 2,     # Port 3, 4 -> Net 2
            3, 3,     # Port 5, 6 -> Net 3
            4, 4,     # Port 7, 8 -> Net 4
            5, 5,     # Port 9, 10 -> Net 5
            6         # Port 11 -> Net 6 (ext)
        ])
    )

    result = solver.run(s_bd, z0_ports, topology)

    # Total transmission = 0.5 ^ 6 = 0.015625
    expected_s = jnp.array([
        [0.0, 0.015625],
        [0.015625, 0.0]
    ], dtype=jnp.complex128)

    assert result.s.shape == (2, 2)
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)
    np.testing.assert_allclose(result.z0, jnp.array([50.0, 50.0]), atol=1e-7)

@pytest.mark.parametrize("SolverClass", [
    GlobalScatteringReducer, 
    HierarchicalScatteringReducer,
    SequentialScatteringReducer
])
def test_star_junction_3port(SolverClass):
    """
    Connects three identical 2-port lines to a single central node (a star junction).
    Validates that the X-matrix generalization correctly handles N >= 3 ports 
    meeting at a single net.
    """
    solver = SolverClass()
    
    # Three ideal, matched, lossless lines
    S_line = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    s_bd = jax.scipy.linalg.block_diag(S_line, S_line, S_line)
    z0_ports = jnp.full(6, 50.0)
    
    # Topology: 
    # Line 1: Port 0 (Ext 0), Port 1 (Int)
    # Line 2: Port 2 (Int), Port 3 (Ext 1)
    # Line 3: Port 4 (Int), Port 5 (Ext 2)
    # Internal ports 1, 2, and 4 all meet at Net 1.
    topology = PortRepresentation(
        ext_idx=np.array([0, 3, 5]),
        int_idx=np.array([1, 2, 4]),
        port_to_net_map=np.array([
            0,  # Port 0 -> Net 0
            1,  # Port 1 -> Net 1 (Star Center)
            1,  # Port 2 -> Net 1 (Star Center)
            2,  # Port 3 -> Net 2
            1,  # Port 4 -> Net 1 (Star Center)
            3   # Port 5 -> Net 3
        ])
    )
    
    result = solver.run(s_bd, z0_ports, topology)
    
    # For three equal admittance lines meeting at a node, the reflection 
    # coefficient at the junction is -1/3, and transmission is 2/3.
    expected_s = jnp.array([
        [-1/3,  2/3,  2/3],
        [ 2/3, -1/3,  2/3],
        [ 2/3,  2/3, -1/3]
    ], dtype=jnp.complex128)
    
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)


@pytest.mark.parametrize("SolverClass", [
    GlobalScatteringReducer, 
    HierarchicalScatteringReducer,
    SequentialScatteringReducer
])
def test_impedance_step_mismatch(SolverClass):
    """
    Connects a 50-ohm line to a 75-ohm line.
    Validates that the solver intrinsically handles the reflection 
    caused by differing characteristic impedances at a connection node.
    """
    solver = SolverClass()
    
    # Two ideal lines
    S_line = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    s_bd = jax.scipy.linalg.block_diag(S_line, S_line)
    
    # Line 1 is 50 ohms, Line 2 is 75 ohms
    z0_ports = jnp.array([50.0, 50.0, 75.0, 75.0])
    
    # Connect Port 1 to Port 2
    topology = PortRepresentation(
        ext_idx=np.array([0, 3]),
        int_idx=np.array([1, 2]),
        port_to_net_map=np.array([0, 1, 1, 2])
    )
    
    result = solver.run(s_bd, z0_ports, topology)
    
    # Gamma = (75 - 50) / (75 + 50) = 25 / 125 = 0.2
    # Transmission = sqrt(1 - Gamma^2) (for power waves)
    gamma = 0.2
    tau = np.sqrt(1 - gamma**2)
    
    expected_s = jnp.array([
        [gamma, tau],
        [tau, -gamma]
    ], dtype=jnp.complex128)
    
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)
    np.testing.assert_allclose(result.z0, jnp.array([50.0, 75.0]), atol=1e-7)


def test_solver_parity_complex_ring():
    """
    Creates a 4-node ring network and solves it using all three reducers.
    Ensures mathematical equivalence across the different formulations.
    """
    # 4 lines forming a ring. We use slightly lossy lines (0.5) to prevent 
    # the mathematically singular infinite resonance of a perfectly lossless loop.
    S_line = jnp.array([[0.0, 0.5], [0.5, 0.0]], dtype=jnp.complex128)
    s_bd = jax.scipy.linalg.block_diag(S_line, S_line, S_line, S_line)
    z0_ports = jnp.full(8, 50.0)
    
    # Ring Topology:
    # Node A (Ext 1): Port 0, Port 7
    # Node B (Int)  : Port 1, Port 2
    # Node C (Ext 2): Port 3, Port 4
    # Node D (Int)  : Port 5, Port 6
    topology = PortRepresentation(
        ext_idx=np.array([0, 3]),  # Probing Node A and Node C
        int_idx=np.array([1, 2, 4, 5, 6, 7]),
        port_to_net_map=np.array([
            0,  # P0 -> Node A (Ext)
            1,  # P1 -> Node B
            1,  # P2 -> Node B
            2,  # P3 -> Node C (Ext)
            2,  # P4 -> Node C (Ext)
            3,  # P5 -> Node D
            3,  # P6 -> Node D
            0   # P7 -> Node A (Ext)
        ])
    )
    
    res_global = GlobalScatteringReducer().run(s_bd, z0_ports, topology)
    res_hier = HierarchicalScatteringReducer().run(s_bd, z0_ports, topology)
    res_seq = SequentialScatteringReducer().run(s_bd, z0_ports, topology)
    
    # Now that the system is well-conditioned, they will match perfectly
    np.testing.assert_allclose(res_hier.s, res_global.s, atol=1e-7)
    np.testing.assert_allclose(res_seq.s, res_global.s, atol=1e-7)