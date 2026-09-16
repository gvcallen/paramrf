# tests/test_models/test_circuit_port_order.py
"""
A `Circuit` orders its external ports by `Port` declaration: the first `Port` found
scanning `connections` node by node, left to right, is port 0. This must hold for
every solver and must not depend on how internal components are wired.
"""
import pytest
import numpy as np
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models import (
    Circuit, Port, Ground, Resistor,
    GlobalScatteringCircuitSolver,
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver,
    GlobalNodalCircuitSolver,
    GlobalMNACircuitSolver,
)

SOLVERS = [
    GlobalScatteringCircuitSolver,
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver,
    GlobalNodalCircuitSolver,
    GlobalMNACircuitSolver,
]

Z0 = 50.0


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=10.0, npoints=3, unit='GHz')


def _y_to_s(Y, z0=Z0):
    """Independent reference: S = (I - z0 Y)(I + z0 Y)^-1 for a real, uniform z0."""
    eye = np.eye(Y.shape[0])
    return (eye - z0 * Y) @ np.linalg.inv(eye + z0 * Y)


# Asymmetric two-port: node A has a 10 Ω shunt, node B a 1000 Ω shunt, with a 5 Ω
# series resistor between them. Nodal Y written down by hand, ordered (A, B).
R_A, R_B, R_SERIES = 10.0, 1000.0, 5.0
Y_AB = np.array([
    [1 / R_A + 1 / R_SERIES, -1 / R_SERIES],
    [-1 / R_SERIES, 1 / R_B + 1 / R_SERIES],
])
S_AB = _y_to_s(Y_AB)


def _asymmetric_two_port(solver, a_first: bool, series_flipped: bool, **kwargs):
    shunt_a, shunt_b, series = Resistor(R=R_A), Resistor(R=R_B), Resistor(R=R_SERIES)
    port_a, port_b, ground = Port(), Port(), Ground()

    series_a, series_b = ((series, 1), (series, 0)) if series_flipped else ((series, 0), (series, 1))
    node_a = [(port_a, 0), (shunt_a, 0), series_a]
    node_b = [(port_b, 0), (shunt_b, 0), series_b]
    nodes = [node_a, node_b] if a_first else [node_b, node_a]

    return Circuit(
        nodes + [[(ground, 0), (shunt_a, 1), (shunt_b, 1)]],
        solver=solver,
        **kwargs,
    )


def test_asymmetric_two_port_reference_values():
    # Values quoted in the issue, as a guard on the hand-written reference.
    assert np.isclose(S_AB[0, 0], -0.7122, atol=1e-4)
    assert np.isclose(S_AB[1, 1], -0.5833, atol=1e-4)


@pytest.mark.parametrize("series_flipped", [False, True])
@pytest.mark.parametrize("a_first", [True, False])
@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_ports_follow_declaration_order(solver_cls, a_first, series_flipped, freq):
    circuit = _asymmetric_two_port(solver_cls(), a_first, series_flipped)
    expected = S_AB if a_first else S_AB[::-1, ::-1]

    s = circuit.s(freq, z0=Z0)

    np.testing.assert_allclose(s, np.broadcast_to(expected, s.shape), atol=1e-8)


def _kron_reduce(Y, keep):
    """Independent reference: eliminate every node not in `keep` by Schur complement."""
    keep = np.asarray(keep)
    drop = np.setdiff1d(np.arange(Y.shape[0]), keep)
    return Y[np.ix_(keep, keep)] - Y[np.ix_(keep, drop)] @ np.linalg.solve(
        Y[np.ix_(drop, drop)], Y[np.ix_(drop, keep)]
    )


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_nested_circuit_ports_follow_declaration_order(solver_cls, flatten, freq):
    # The inner circuit has its series element's port indices swapped.
    inner = _asymmetric_two_port(solver_cls(), a_first=True, series_flipped=True)

    # The outer circuit adds a 3 Ω series resistor on A and a 2 Ω one on B. The
    # resistor on B is declared first, so it is discovered before either `Port`,
    # but port A is still the first `Port` declared.
    R_TA, R_TB = 3.0, 2.0
    t_a, t_b = Resistor(R=R_TA), Resistor(R=R_TB)
    port_a, port_b = Port(), Port()
    outer = Circuit([
        [(t_b, 0), (inner, 1)],
        [(t_a, 0), (inner, 0)],
        [(port_a, 0), (t_a, 1)],
        [(port_b, 0), (t_b, 1)],
    ], solver=solver_cls(), flatten=flatten)

    # Nodes (A, B, port A, port B), by hand.
    Y = np.zeros((4, 4))
    Y[:2, :2] = Y_AB
    for i, j, R in [(0, 2, R_TA), (1, 3, R_TB)]:
        Y[np.ix_([i, j], [i, j])] += np.array([[1, -1], [-1, 1]]) / R
    expected = _y_to_s(_kron_reduce(Y, keep=[2, 3]))

    s = outer.s(freq, z0=Z0)

    np.testing.assert_allclose(s, np.broadcast_to(expected, s.shape), atol=1e-8)


def test_ports_sharing_a_net_raise():
    shunt, ground = Resistor(R=R_A), Ground()
    port_a, port_b = Port(name="port_a"), Port(name="port_b")

    with pytest.raises(ValueError, match="port_a.*port_b"):
        Circuit([
            [(port_a, 0), (port_b, 0), (shunt, 0)],
            [(ground, 0), (shunt, 1)],
        ])


def test_port_sharing_a_net_with_ground_raises():
    shunt, ground = Resistor(R=R_A), Ground()
    port_a, port_b = Port(name="port_a"), Port(name="port_b")

    with pytest.raises(ValueError, match="port_b.*Ground"):
        Circuit([
            [(port_a, 0), (shunt, 0)],
            [(ground, 0), (shunt, 1), (port_b, 0)],
        ])


@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_pi_clc_docstring_example_matches_skrf(solver_cls, freq):
    skrf = pytest.importorskip("skrf")
    from pmrf.models import Capacitor, Inductor

    # Exactly the `Circuit` docstring example: `(L, 1)` on the first node.
    C1, C2 = Capacitor(C=2e-12), Capacitor(C=1.5e-12)
    L = Inductor(L=3e-9)
    p0, p1, ground = Port(), Port(), Ground()
    pi_clc = Circuit([
        [(p0, 0), (C1, 1), (L, 1)],
        [(p1, 0), (C2, 1), (L, 0)],
        [(ground, 0), (C1, 0), (C2, 0)],
    ], solver=solver_cls())

    # Port 1 is the C1 side.
    media = skrf.media.DefinedGammaZ0(frequency=freq.to_skrf(), z0_port=Z0)
    expected = (
        media.shunt_capacitor(2e-12) ** media.inductor(3e-9) ** media.shunt_capacitor(1.5e-12)
    ).s

    np.testing.assert_allclose(pi_clc.s(freq, z0=Z0), expected, atol=1e-8)
