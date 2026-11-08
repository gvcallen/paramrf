# tests/test_models/test_circuit_failed_solve.py
"""
A circuit whose linear solve fails gives NaN S and NaN gradients instead of raising (#353).

lineax raises from inside compiled code on a failed solve, and its tangent and transpose
solves raise even with `throw=False`, so a fit that steps onto such a point used to die
before any ParamRF check could see it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmrf.frequency import Frequency
from pmrf.models import (
    Circuit, Port, Resistor,
    GlobalNodalCircuitSolver, GlobalMNACircuitSolver,
    GlobalScatteringCircuitSolver, HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver,
)

FREQ = Frequency(start=1.0, stop=2.0, npoints=3, unit='GHz')

SOLVERS = [
    GlobalNodalCircuitSolver,
    GlobalMNACircuitSolver,
    GlobalScatteringCircuitSolver,
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver,
]


def _series_circuit(x, solver, floating=None):
    """
    A resistance sqrt(3 - x) in series between two Ports, non-finite for x > 3.

    `floating` adds an otherwise unconnected sub-circuit that makes the system singular
    at eps = 0 while it stays finite:

    - `'ring'`: two resistors in a ring between two nets. The ring's node voltages are
      undetermined, so the nodal and MNA systems are singular. The scattering solvers
      see two series two-ports, which have no common mode, so it is regular for them.
    - `'self_loop'`: a 100 ohm resistor with both ends on one net. A wave entering one
      end leaves the other unchanged in sum (S11 + S21 = 1 exactly at 100 ohm, against
      the 50 ohm reference), so the scattering system is singular. The nodal solver
      sees a zero 1x1 block, which its pseudo-inverse solves, so it is regular for it.
    """
    r = Resistor(R=jnp.sqrt(3.0 - x))
    p0, p1 = Port(), Port()
    nets = [[(p0, 0), (r, 0)], [(r, 1), (p1, 0)]]
    if floating == 'ring':
        ra, rb = Resistor(R=10.0), Resistor(R=20.0)
        nets += [[(ra, 0), (rb, 0)], [(ra, 1), (rb, 1)]]
    elif floating == 'self_loop':
        loop = Resistor(R=100.0)
        nets.append([(loop, 0), (loop, 1)])
    return Circuit(nets, solver=solver)


def _loss(solver, floating=None):
    return lambda x: jnp.sum(jnp.abs(_series_circuit(x, solver, floating).s(FREQ)) ** 2)


@pytest.mark.parametrize("SolverClass", SOLVERS)
def test_nonfinite_stamp_gives_nan(SolverClass):
    solver = SolverClass()
    s = _series_circuit(3.5, solver).s(FREQ)
    assert np.all(np.isnan(s))
    assert np.all(np.isnan(jax.jit(lambda x: _series_circuit(x, solver).s(FREQ))(3.5)))

    loss = _loss(solver)
    assert np.isnan(jax.grad(loss)(3.5))
    assert np.isnan(jax.jit(jax.grad(loss))(3.5))


@pytest.mark.parametrize("SolverClass, floating", [
    (GlobalNodalCircuitSolver, 'ring'),
    (GlobalMNACircuitSolver, 'ring'),
    (GlobalScatteringCircuitSolver, 'self_loop'),
    (HierarchicalScatteringCircuitSolver, 'self_loop'),
    (SequentialScatteringCircuitSolver, 'self_loop'),
])
def test_finite_singular_system_gives_nan(SolverClass, floating):
    solver = SolverClass(eps=0.0)
    s = _series_circuit(2.5, solver, floating).s(FREQ)
    assert np.all(np.isnan(s))

    loss = _loss(solver, floating)
    assert np.isnan(jax.grad(loss)(2.5))
    assert np.isnan(jax.jit(jax.grad(loss))(2.5))


@pytest.mark.parametrize("SolverClass", SOLVERS)
def test_finite_point_unchanged(SolverClass):
    """At x = 2.5, S and its gradient are those of a series resistance between 50 ohm ports."""
    solver = SolverClass()
    x = 2.5
    R = np.sqrt(3.0 - x)
    s11, s21 = R / (R + 100.0), 100.0 / (R + 100.0)
    expected = np.broadcast_to([[s11, s21], [s21, s11]], (FREQ.npoints, 2, 2))
    # The nodal solver's GMIN of 1e-12 S sits on the port nodes too, against the 1/50 S
    # port load, so it moves S by about 50 * 1e-12. The others move S by about eps.
    atol = 1e-10 if SolverClass is GlobalNodalCircuitSolver else 1e-11
    np.testing.assert_allclose(_series_circuit(x, solver).s(FREQ), expected, rtol=0, atol=atol)

    # d/dx of 3 (2 s11² + 2 s21²), with dR/dx = -1 / (2R).
    ds11 = 100.0 / (R + 100.0) ** 2
    ds21 = -100.0 / (R + 100.0) ** 2
    expected_grad = 3 * (4 * s11 * ds11 + 4 * s21 * ds21) * (-1 / (2 * R))
    np.testing.assert_allclose(jax.grad(_loss(solver))(x), expected_grad, rtol=1e-9)
