"""tests/test_scattering_solvers.py"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Adjust imports based on your project structure
from pmrf.models import (
    GlobalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver,
    HierarchicalScatteringCircuitSolver,
    PortRepresentation,
)


@pytest.mark.parametrize("SolverClass", [
    GlobalScatteringCircuitSolver, 
    SequentialScatteringCircuitSolver, 
    HierarchicalScatteringCircuitSolver
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
    z0_ext = jnp.array([50.0, 50.0])
    topology = PortRepresentation(
        port_to_net_map=np.array([0, 1, 1, 2]),
        ext_net_ids=np.array([0, 2])
    )
            
    result = solver.run(s_bd, z0_ports, z0_ext, topology)
    
    expected_s = jnp.array([[0.0, 0.25], 
                            [0.25, 0.0]], dtype=jnp.complex128)
    
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)
    np.testing.assert_allclose(result.z0, jnp.array([50.0, 50.0]), atol=1e-7)


def test_hierarchical_reducer_complex_chain():
    """
    Tests HierarchicalScatteringReducer with 5 internal nets (pairs).
    """
    solver = HierarchicalScatteringCircuitSolver()

    # 6 components, each a 2-port with S11=S22=0, S21=S12=0.5
    S_block = jnp.array([[0.0, 0.5], [0.5, 0.0]], dtype=jnp.complex128)
    s_bd = jax.scipy.linalg.block_diag(*[S_block for _ in range(6)])
    z0_ports = jnp.full(12, 50.0)

    # 12 total ports. Chain connection.
    # Ext: Port 0 and 11
    # Int: Ports 1-10 paired up sequentially
    z0_ext = jnp.array([50.0, 50.0])
    topology = PortRepresentation(
        port_to_net_map=np.array([
            0,        # Port 0 -> Net 0 (ext)
            1, 1,     # Port 1, 2 -> Net 1
            2, 2,     # Port 3, 4 -> Net 2
            3, 3,     # Port 5, 6 -> Net 3
            4, 4,     # Port 7, 8 -> Net 4
            5, 5,     # Port 9, 10 -> Net 5
            6         # Port 11 -> Net 6 (ext)
        ]),
        ext_net_ids=np.array([0, 6])
    )

    result = solver.run(s_bd, z0_ports, z0_ext, topology)

    # Total transmission = 0.5 ^ 6 = 0.015625
    expected_s = jnp.array([
        [0.0, 0.015625],
        [0.015625, 0.0]
    ], dtype=jnp.complex128)

    assert result.s.shape == (2, 2)
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)
    np.testing.assert_allclose(result.z0, jnp.array([50.0, 50.0]), atol=1e-7)

@pytest.mark.parametrize("SolverClass", [
    GlobalScatteringCircuitSolver, 
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver
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
    z0_ext = jnp.array([50.0, 50.0, 50.0])
    topology = PortRepresentation(
        port_to_net_map=np.array([
            0,  # Port 0 -> Net 0
            1,  # Port 1 -> Net 1 (Star Center)
            1,  # Port 2 -> Net 1 (Star Center)
            2,  # Port 3 -> Net 2
            1,  # Port 4 -> Net 1 (Star Center)
            3   # Port 5 -> Net 3
        ]),
        ext_net_ids=np.array([0, 2, 3])
    )
    
    result = solver.run(s_bd, z0_ports, z0_ext, topology)
    
    # For three equal admittance lines meeting at a node, the reflection 
    # coefficient at the junction is -1/3, and transmission is 2/3.
    expected_s = jnp.array([
        [-1/3,  2/3,  2/3],
        [ 2/3, -1/3,  2/3],
        [ 2/3,  2/3, -1/3]
    ], dtype=jnp.complex128)
    
    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)


@pytest.mark.parametrize("SolverClass", [
    GlobalScatteringCircuitSolver, 
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver
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
    z0_ext = jnp.array([50.0, 75.0])
    topology = PortRepresentation(
        port_to_net_map=np.array([0, 1, 1, 2]),
        ext_net_ids=np.array([0, 2])
    )
    
    result = solver.run(s_bd, z0_ports, z0_ext, topology)
    
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
    z0_ext = jnp.array([50.0, 50.0])
    topology = PortRepresentation(
        port_to_net_map=np.array([
            0,  # P0 -> Node A (Ext)
            1,  # P1 -> Node B
            1,  # P2 -> Node B
            2,  # P3 -> Node C (Ext)
            2,  # P4 -> Node C (Ext)
            3,  # P5 -> Node D
            3,  # P6 -> Node D
            0   # P7 -> Node A (Ext)
        ]),
        ext_net_ids=np.array([0, 2])  # Probing Node A and Node C
    )
    
    res_global = GlobalScatteringCircuitSolver().run(s_bd, z0_ports, z0_ext, topology)
    res_hier = HierarchicalScatteringCircuitSolver().run(s_bd, z0_ports, z0_ext, topology)
    res_seq = SequentialScatteringCircuitSolver().run(s_bd, z0_ports, z0_ext, topology)
    
    # Now that the system is well-conditioned, they will match perfectly
    np.testing.assert_allclose(res_hier.s, res_global.s, atol=1e-7)
    np.testing.assert_allclose(res_seq.s, res_global.s, atol=1e-7)


@pytest.mark.parametrize("SolverClass", [
    GlobalScatteringCircuitSolver, 
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver
])
def test_dangling_ext_with_internal_termination(SolverClass):
    """
    Tests a 1-port external boundary coupled with an internal termination.
    This explicitly validates the virtual probe injection on a single 
    dangling port while an internal net gets heavily reduced.
    """
    solver = SolverClass()

    # Component 1: Attenuator (3dB / 0.5 voltage transmission)
    S_att = jnp.array([[0.0, 0.5], 
                       [0.5, 0.0]], dtype=jnp.complex128)
    
    # Component 2: Short circuit (-1 reflection)
    S_short = jnp.array([[-1.0]], dtype=jnp.complex128)

    s_bd = jax.scipy.linalg.block_diag(S_att, S_short)
    z0_ports = jnp.array([50.0, 50.0, 50.0])

    # Topology:
    # Attenuator: Port 0 (ext), Port 1 (int)
    # Short: Port 2 (int)
    # Net 0: Port 0 (Dangling External, Count = 1)
    # Net 1: Port 1, Port 2 (Internal connection, Count = 2)
    z0_ext = jnp.array([50.0])
    topology = PortRepresentation(
        port_to_net_map=np.array([0, 1, 1]),
        ext_net_ids=np.array([0])
    )

    result = solver.run(s_bd, z0_ports, z0_ext, topology)

    # Signal enters Port 0 -> drops by 0.5 -> hits short (-1.0) -> drops by 0.5 returning.
    # Expected S11 = 0.5 * -1.0 * 0.5 = -0.25
    expected_s = jnp.array([[-0.25]], dtype=jnp.complex128)

    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)
    np.testing.assert_allclose(result.z0, jnp.array([50.0]), atol=1e-7)


@pytest.mark.parametrize("SolverClass", [
    GlobalScatteringCircuitSolver, 
    SequentialScatteringCircuitSolver, 
    HierarchicalScatteringCircuitSolver
])
def test_multiple_dangling_ext_branches(SolverClass):
    """
    Tests a network with multiple dangling external ports and multiple 
    internal nets, mimicking the complex branching of higher-level circuits.
    Ensures that multiple padding operations don't misalign indices.
    """
    solver = SolverClass()

    # Component 1: 3-Port Star Junction
    S_star = jnp.array([
        [-1/3,  2/3,  2/3],
        [ 2/3, -1/3,  2/3],
        [ 2/3,  2/3, -1/3]
    ], dtype=jnp.complex128)

    # Components 2 & 3: Ideal, transparent delay-less lines
    S_line = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)

    s_bd = jax.scipy.linalg.block_diag(S_star, S_line, S_line)
    
    # 7 total local ports
    z0_ports = jnp.full(7, 50.0)

    # Topology:
    # Star: Port 0 (Ext), Port 1 (Int), Port 2 (Int)
    # Line 1: Port 3 (Int), Port 4 (Ext)
    # Line 2: Port 5 (Int), Port 6 (Ext)
    z0_ext = jnp.array([50.0, 50.0, 50.0])
    topology = PortRepresentation(
        port_to_net_map=np.array([
            0,  # P0 -> Net 0 (Dangling Ext)
            1,  # P1 -> Net 1 (Int)
            2,  # P2 -> Net 2 (Int)
            1,  # P3 -> Net 1 (Int)
            3,  # P4 -> Net 3 (Dangling Ext)
            2,  # P5 -> Net 2 (Int)
            4   # P6 -> Net 4 (Dangling Ext)
        ]),
        ext_net_ids=np.array([0, 3, 4])
    )

    result = solver.run(s_bd, z0_ports, z0_ext, topology)

    # Because the lines are transparent, the final 3-port reduced network 
    # should mathematically collapse to behave exactly like the original star junction.
    expected_s = S_star

    np.testing.assert_allclose(result.s, expected_s, atol=1e-7)
    np.testing.assert_allclose(result.z0, jnp.array([50.0, 50.0, 50.0]), atol=1e-7)