# tests/test_models/test_circuit_z0.py
"""
A per-port `z0` works with every circuit solver (ADR-0006, issue #206).

A circuit solver evaluates its internal components at a fixed internal reference,
not at the caller's `z0`, and `Model` converts an explicit `z0` to an array once.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

from pmrf.frequency import Frequency
from pmrf.models import (
    Model, Circuit, Port, Ground, Resistor, Capacitor, Inductor,
    GlobalScatteringCircuitSolver,
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver,
    GlobalNodalCircuitSolver,
    GlobalMNACircuitSolver,
)

SOLVERS = [
    GlobalNodalCircuitSolver,
    GlobalMNACircuitSolver,
    GlobalScatteringCircuitSolver,
    HierarchicalScatteringCircuitSolver,
    SequentialScatteringCircuitSolver,
]

Z0_PER_PORT = [10.0, 50.0]


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')


def _reporter_circuit(solver):
    """Two isolated shunt resistors, each matched to its own Port's `z0` (issue #186)."""
    r10, r50 = Resistor(R=10.0), Resistor(R=50.0)
    p10, p50, ground = Port(z0=10.0), Port(z0=50.0), Ground()
    return Circuit([
        [(p10, 0), (r10, 0)],
        [(ground, 0), (r10, 1), (r50, 1)],
        [(p50, 0), (r50, 0)],
    ], solver=solver)


@pytest.mark.parametrize("solver_cls", SOLVERS)
@pytest.mark.parametrize("z0", [Z0_PER_PORT, np.array(Z0_PER_PORT)], ids=["list", "array"])
def test_reporter_circuit_is_matched_at_per_port_z0(solver_cls, z0, freq):
    s = _reporter_circuit(solver_cls()).s(freq, z0=z0)

    assert s.shape == (freq.npoints, 2, 2)
    np.testing.assert_allclose(s, 0.0, atol=1e-10)


# Series R followed by a shunt C, then an inductor into port 2.
R_SERIES, C_SHUNT, L_SERIES = 30.0, 1e-12, 2e-9


def _series_r_shunt_c_series_l(solver):
    r, c, l = Resistor(R=R_SERIES), Capacitor(C=C_SHUNT), Inductor(L=L_SERIES)
    p0, p1, ground = Port(), Port(), Ground()
    return Circuit([
        [(p0, 0), (r, 0)],
        [(r, 1), (c, 0), (l, 0)],
        [(ground, 0), (c, 1)],
        [(p1, 0), (l, 1)],
    ], solver=solver)


# Per-solver tolerance on |S| against scikit-rf.
SKRF_TOLERANCES = {
    GlobalNodalCircuitSolver: 1e-10,
    GlobalMNACircuitSolver: 1e-10,
    GlobalScatteringCircuitSolver: 1e-10,
    HierarchicalScatteringCircuitSolver: 1e-10,
    SequentialScatteringCircuitSolver: 1e-10,
}


@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_unequal_per_port_z0_matches_skrf_renormalize(solver_cls, freq):
    skrf = pytest.importorskip("skrf")
    z0 = [10.0, 75.0]

    media = skrf.media.DefinedGammaZ0(frequency=freq.to_skrf(), z0_port=50.0)
    expected = (
        media.resistor(R_SERIES) ** media.shunt_capacitor(C_SHUNT) ** media.inductor(L_SERIES)
    )
    expected.renormalize(z0)

    s = _series_r_shunt_c_series_l(solver_cls()).s(freq, z0=z0)

    np.testing.assert_allclose(s, expected.s, atol=SKRF_TOLERANCES[solver_cls])


def test_capacitor_accepts_list_z0(freq):
    cap = Capacitor(C=1e-12)

    np.testing.assert_allclose(
        cap.s(freq, z0=Z0_PER_PORT),
        cap.s(freq, z0=np.array(Z0_PER_PORT)),
    )


@pytest.mark.parametrize("model", [Capacitor(C=1e-12), Resistor(R=30.0)], ids=["capacitor", "resistor"])
def test_scalar_z0_takes_scalar_path(model, freq):
    # Reference written with the explicit uniform per-port value.
    expected = model.s(freq, z0=np.array([25.0, 25.0]))

    np.testing.assert_allclose(model.s(freq, z0=25.0), expected, atol=1e-14)
    np.testing.assert_allclose(model.s(freq, z0=jnp.asarray(25.0)), expected, atol=1e-14)
    np.testing.assert_allclose(model.s(freq, 25.0), expected, atol=1e-14)


class _RecordsZ0(Model):
    def s(self, freq: Frequency, z0=50.0):
        _RecordsZ0.seen.append(z0)
        return jnp.zeros((freq.npoints, 1, 1), dtype=jnp.complex128)


def test_z0_none_reaches_the_model_unchanged(freq):
    _RecordsZ0.seen = []
    _RecordsZ0().s(freq, z0=None)

    assert _RecordsZ0.seen == [None]


def test_explicit_z0_reaches_the_model_as_an_array(freq):
    _RecordsZ0.seen = []
    _RecordsZ0().s(freq, z0=[10.0])
    _RecordsZ0().s(freq, [10.0])

    assert all(isinstance(z0, jax.Array) for z0 in _RecordsZ0.seen)
