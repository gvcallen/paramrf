# tests/test_models/test_circuit_skrf_port_names.py
"""
`to_skrf` fills `skrf.Network.port_names` from a Circuit's Port names (ADR-0006, issue #208).

Names follow the external Ports in port order. If no Port is named, `port_names` is left
at scikit-rf's default; otherwise an unnamed Port is named by its 1-based index. A
`port_names` passed to `to_skrf` wins.
"""
import pytest
import numpy as np

from pmrf.frequency import Frequency
from pmrf.models import Circuit, Port, Ground, Resistor, Capacitor

skrf = pytest.importorskip("skrf")


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=10.0, npoints=3, unit='GHz')


def _two_port(port0: Port, port1: Port) -> Circuit:
    """A series resistor with a shunt resistor at `port0`; `port0` is port 0."""
    series, shunt, ground = Resistor(R=5.0), Resistor(R=100.0), Ground()
    return Circuit([
        [(port0, 0), (series, 0), (shunt, 0)],
        [(port1, 0), (series, 1)],
        [(ground, 0), (shunt, 1)],
    ])


def _default_port_names(freq, nports):
    return skrf.Network(
        frequency=freq.to_skrf(), s=np.zeros((freq.npoints, nports, nports)),
    ).port_names


def test_named_ports_give_port_names(freq):
    port_in, port_out = Port(name='in'), Port(name='out')

    assert _two_port(port_in, port_out).to_skrf(freq).port_names == ['in', 'out']
    assert _two_port(port_out, port_in).to_skrf(freq).port_names == ['out', 'in']


def test_unnamed_port_is_named_by_its_index(freq):
    ntwk = _two_port(Port(), Port(name='out')).to_skrf(freq)

    assert ntwk.port_names == ['1', 'out']


def test_no_named_ports_leaves_skrf_default(freq):
    ntwk = _two_port(Port(), Port()).to_skrf(freq)

    assert ntwk.port_names == _default_port_names(freq, 2)


def test_explicit_port_names_win(freq):
    circuit = _two_port(Port(name='in'), Port(name='out'))

    ntwk = circuit.to_skrf(freq, port_names=['a', 'b'])

    assert ntwk.port_names == ['a', 'b']


def test_probe_only_model_is_unchanged(freq):
    ntwk = Capacitor(C=1e-12).to_skrf(freq)

    assert ntwk.port_names == _default_port_names(freq, 2)
