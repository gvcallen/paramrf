# tests/test_models/test_circuit_native_z0.py
"""
A Circuit's S-parameters default to its Ports' reference impedance (ADR-0006, issue #207).

`Circuit` and `Port` have a native reference and default to `z0=None`; a probe-only
model defaults to 50 and rejects `z0=None`. An explicit `z0` always wins.
"""
import dataclasses

import pytest
import numpy as np

from pmrf.frequency import Frequency
from pmrf.models import (
    Model, Circuit, Port, Ground, Load, Resistor, Capacitor, Inductor,
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


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')


def _reporter_circuit(solver, **kwargs):
    """Two isolated shunt resistors, each matched to its own Port's `z0` (issue #186)."""
    r10, r50 = Resistor(R=10.0), Resistor(R=50.0)
    p10, p50, ground = Port(z0=10.0), Port(z0=50.0), Ground()
    return Circuit([
        [(p10, 0), (r10, 0)],
        [(ground, 0), (r10, 1), (r50, 1)],
        [(p50, 0), (r50, 0)],
    ], solver=solver, **kwargs)


# Series R followed by a shunt C, then an inductor into port 2.
R_SERIES, C_SHUNT, L_SERIES = 30.0, 1e-12, 2e-9


def _series_r_shunt_c_series_l(solver, z0=(50.0, 50.0), **kwargs):
    r, c, l = Resistor(R=R_SERIES), Capacitor(C=C_SHUNT), Inductor(L=L_SERIES)
    p0, p1, ground = Port(z0=z0[0]), Port(z0=z0[1]), Ground()
    return Circuit([
        [(p0, 0), (r, 0)],
        [(r, 1), (c, 0), (l, 0)],
        [(ground, 0), (c, 1)],
        [(p1, 0), (l, 1)],
    ], solver=solver, **kwargs)


def _skrf_series_r_shunt_c_series_l(freq, z0):
    skrf = pytest.importorskip("skrf")
    media = skrf.media.DefinedGammaZ0(frequency=freq.to_skrf(), z0_port=50.0)
    ntwk = media.resistor(R_SERIES) ** media.shunt_capacitor(C_SHUNT) ** media.inductor(L_SERIES)
    ntwk.renormalize(z0)
    return ntwk


# ---- Circuit ---------------------------------------------------------------------------


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_reporter_circuit_is_matched_by_default(solver_cls, flatten, freq):
    s = _reporter_circuit(solver_cls(), flatten=flatten).s(freq)

    assert s.shape == (freq.npoints, 2, 2)
    np.testing.assert_allclose(s, 0.0, atol=1e-10)


@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_reporter_circuit_at_explicit_50_matches_skrf_renormalize(solver_cls, freq):
    skrf = pytest.importorskip("skrf")
    # At its Ports' impedances the circuit is matched, so scikit-rf's reference is a
    # zero S-matrix at (10, 50), renormalised to 50.
    expected = skrf.Network(
        frequency=freq.to_skrf(), s=np.zeros((freq.npoints, 2, 2)), z0=[10.0, 50.0],
    )
    expected.renormalize(50.0)

    s = _reporter_circuit(solver_cls()).s(freq, z0=50.0)

    np.testing.assert_allclose(s, expected.s, atol=1e-10)
    np.testing.assert_allclose(s[:, 0, 0], -2.0 / 3.0, atol=1e-10)


# Per-solver tolerance on |S| against scikit-rf.
SKRF_TOLERANCES = {
    GlobalNodalCircuitSolver: 1e-10,
    GlobalMNACircuitSolver: 1e-10,
    GlobalScatteringCircuitSolver: 1e-10,
    HierarchicalScatteringCircuitSolver: 1e-10,
    SequentialScatteringCircuitSolver: 1e-10,
}


@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_unequal_port_z0_matches_skrf(solver_cls, freq):
    z0 = (10.0, 75.0)
    circuit = _series_r_shunt_c_series_l(solver_cls(), z0=z0)
    tol = SKRF_TOLERANCES[solver_cls]

    np.testing.assert_allclose(
        circuit.s(freq), _skrf_series_r_shunt_c_series_l(freq, list(z0)).s, atol=tol,
    )
    np.testing.assert_allclose(
        circuit.s(freq, z0=50.0), _skrf_series_r_shunt_c_series_l(freq, 50.0).s, atol=tol,
    )


@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_default_ports_are_unchanged(solver_cls, freq):
    circuit = _series_r_shunt_c_series_l(solver_cls())

    np.testing.assert_array_equal(circuit.s(freq), circuit.s(freq, z0=50.0))

    ntwk = circuit.to_skrf(freq)
    np.testing.assert_array_equal(ntwk.s, np.asarray(circuit.s(freq, z0=50.0)))
    np.testing.assert_array_equal(ntwk.z0, 50.0)


@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_nested_circuit_is_probed(solver_cls, freq):
    # The same inner circuit, once with 50 Ω Ports and once with 10 Ω and 75 Ω ones.
    inner_default = _series_r_shunt_c_series_l(solver_cls())
    inner_odd = _series_r_shunt_c_series_l(solver_cls(), z0=(10.0, 75.0))

    def in_cascade(inner):
        return inner ** Capacitor(C=0.5e-12)

    def in_circuit(inner):
        shunt, p0, p1, ground = Resistor(R=100.0), Port(), Port(), Ground()
        return Circuit([
            [(p0, 0), (inner, 0)],
            [(inner, 1), (shunt, 0), (p1, 0)],
            [(ground, 0), (shunt, 1)],
        ], solver=solver_cls())

    for wrap in (in_cascade, in_circuit):
        np.testing.assert_allclose(
            wrap(inner_odd).s(freq), wrap(inner_default).s(freq), atol=1e-12,
        )


def test_ports_are_in_port_order():
    # Port B is declared first in `connections`, so it is port 0.
    shunt_a, shunt_b, series = Resistor(R=10.0), Resistor(R=1000.0), Resistor(R=5.0)
    port_a, port_b, ground = Port(name="a"), Port(name="b"), Ground()
    circuit = Circuit([
        [(port_b, 0), (shunt_b, 0), (series, 1)],
        [(port_a, 0), (shunt_a, 0), (series, 0)],
        [(ground, 0), (shunt_a, 1), (shunt_b, 1)],
    ])

    assert circuit.ports == [port_b, port_a]


# ---- Port ------------------------------------------------------------------------------


def test_port_is_matched_by_default(freq):
    np.testing.assert_allclose(Port(z0=10.0).s(freq), 0.0, atol=1e-14)


def test_port_at_explicit_z0_is_a_load(freq):
    expected = Load(gamma=0.0, z0=10.0).s(freq, z0=50.0)

    np.testing.assert_allclose(Port(z0=10.0).s(freq, z0=50.0), expected, atol=1e-14)
    np.testing.assert_allclose(expected, -2.0 / 3.0, atol=1e-14)


# ---- supports_native_z0 ----------------------------------------------------------------


def test_probe_only_model_rejects_z0_none(freq):
    with pytest.raises(ValueError, match=r"Resistor.*z0"):
        Resistor(R=10.0).s(freq, z0=None)


@pytest.mark.parametrize("cls", [Model, Resistor, Circuit, Port])
def test_supports_native_z0_is_not_a_field(cls):
    assert "supports_native_z0" not in {f.name for f in dataclasses.fields(cls)}


def test_supports_native_z0_flags():
    assert Model.supports_native_z0 is False
    assert Resistor.supports_native_z0 is False
    assert Circuit.supports_native_z0 is True
    assert Port.supports_native_z0 is True


# ---- to_skrf ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_cls", SOLVERS)
def test_to_skrf_reports_port_z0(solver_cls, freq):
    circuit = _reporter_circuit(solver_cls())

    ntwk = circuit.to_skrf(freq)

    np.testing.assert_array_equal(ntwk.z0, np.broadcast_to([10.0, 50.0], (freq.npoints, 2)))
    np.testing.assert_allclose(ntwk.s, 0.0, atol=1e-10)


def test_to_skrf_explicit_z0_wins(freq):
    circuit = _reporter_circuit(GlobalMNACircuitSolver())

    ntwk = circuit.to_skrf(freq, z0=50.0)

    np.testing.assert_array_equal(ntwk.z0, 50.0)
    np.testing.assert_allclose(ntwk.s, np.asarray(circuit.s(freq, z0=50.0)))


def test_to_skrf_port_reports_its_z0(freq):
    ntwk = Port(z0=10.0).to_skrf(freq)

    np.testing.assert_array_equal(ntwk.z0, 10.0)
    np.testing.assert_allclose(ntwk.s, 0.0, atol=1e-14)


def test_to_skrf_probe_only_model_is_50_ohm(freq):
    cap = Capacitor(C=1e-12)

    ntwk = cap.to_skrf(freq)

    np.testing.assert_array_equal(ntwk.z0, 50.0)
    np.testing.assert_allclose(ntwk.s, np.asarray(cap.s(freq, z0=50.0)))
