# tests/test_models/test_circuit_shorted_ports.py
"""
The MNA solver gives exact S and gradients when ports are shorted together (ADR-0007, #222).

Two ports joined by a zero impedance have no finite Y, so the MNA solver returns S at the
probe reference, loading each port with it. The scattering solver never forms Y, so it is
the reference here.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmrf.frequency import Frequency
from pmrf.models import (
    Model, Circuit, Port, Resistor, PiSectionCLC,
    GlobalMNACircuitSolver, GlobalScatteringCircuitSolver,
)
from pmrf.rf import MNAStamp

FREQ = Frequency(start=1.0, stop=10.0, npoints=10, unit='GHz')

# The MNA solver's GMIN (1e-12 S to ground on every node) is a physical shunt that the
# scattering solver does not have. At a 50 ohm port it moves S by about GMIN * Z0 = 5e-11.
GMIN_ATOL = 1e-10


def _pi_circuit(L, solver=None, **kwargs):
    """`PiSectionCLC` between two Ports. At L = 0 the Ports are shorted together."""
    pi = PiSectionCLC(C1=1e-12, L=L, C2=1e-12)
    p0, p1 = Port(), Port()
    solver = solver or GlobalMNACircuitSolver()
    return Circuit([[(p0, 0), (pi, 0)], [(p1, 0), (pi, 1)]], solver=solver, **kwargs)


def _s11_power(circuit_of_L, z0):
    """Σ|S11|² over frequency, and its derivative with respect to L at L = 0."""
    def f(L):
        return jnp.sum(jnp.abs(circuit_of_L(L).s(FREQ, z0=z0)[:, 0, 0]) ** 2)
    return jax.value_and_grad(f)(0.0)


@pytest.mark.parametrize("z0", [
    50.0,
    jnp.array([50.0, 10.0 + 5.0j]),
], ids=["50", "complex-per-port"])
def test_shorted_ports_match_scattering(z0):
    value, grad = _s11_power(_pi_circuit, z0)
    ref_value, ref_grad = _s11_power(lambda L: _pi_circuit(L, GlobalScatteringCircuitSolver()), z0)

    # Before #222 the value was off by 3e-3 and dL by ten orders of magnitude.
    np.testing.assert_allclose(value, ref_value, rtol=GMIN_ATOL)
    np.testing.assert_allclose(grad, ref_grad, rtol=1e-8)


def test_nested_shorted_circuit_matches_flattened():
    """An unflattened sub-circuit reaches its parent through `Circuit.mna()`, at the hub reference."""
    def outer(L, flatten):
        inner = _pi_circuit(L)
        p0, p1 = Port(), Port()
        return Circuit([[(p0, 0), (inner, 0)], [(p1, 0), (inner, 1)]], flatten=flatten)

    z0 = jnp.array([50.0, 10.0 + 5.0j])
    nested = outer(0.0, flatten=False).s(FREQ, z0=z0)
    flat = outer(0.0, flatten=True).s(FREQ, z0=z0)
    # Each level of nesting adds its own GMIN.
    np.testing.assert_allclose(nested, flat, rtol=0, atol=GMIN_ATOL)

    value, grad = _s11_power(lambda L: outer(L, flatten=False), z0)
    ref_value, ref_grad = _s11_power(lambda L: outer(L, flatten=True), z0)
    np.testing.assert_allclose(value, ref_value, rtol=GMIN_ATOL)
    np.testing.assert_allclose(grad, ref_grad, rtol=1e-8)


def test_parallel_zero_impedance_branches_are_a_thru():
    """A loop of two zero-impedance branches is resolved by the auxiliary series resistance."""
    a = PiSectionCLC(C1=0.0, L=0.0, C2=0.0)
    b = PiSectionCLC(C1=0.0, L=0.0, C2=0.0)
    p0, p1 = Port(), Port()
    circuit = Circuit([[(p0, 0), (a, 0), (b, 0)], [(p1, 0), (a, 1), (b, 1)]])

    s = circuit.s(FREQ, z0=50.0)
    thru = np.broadcast_to(np.array([[0, 1], [1, 0]]), s.shape)
    np.testing.assert_allclose(s, thru, rtol=0, atol=GMIN_ATOL)


def test_shorted_ports_y_is_not_finite_but_mna_is():
    """Y of shorted ports is only as finite as GMIN makes it; the stamp at the hub reference is well scaled."""
    circuit = _pi_circuit(0.0)
    # `s2y` of a thru, regularised only by GMIN: about 1 / (GMIN * Z0^2).
    assert np.all(np.abs(circuit.y(FREQ)) > 1e10)

    stamp = circuit.mna(FREQ)
    for block in (stamp.Y, stamp.B, stamp.C, stamp.D):
        assert np.all(np.isfinite(block))


def test_y_matches_scattering_when_finite():
    """Away from the short, `Circuit.y()` agrees across solvers."""
    y = _pi_circuit(1e-9).y(FREQ)
    ref = _pi_circuit(1e-9, GlobalScatteringCircuitSolver()).y(FREQ)
    np.testing.assert_allclose(y, ref, rtol=0, atol=GMIN_ATOL)


class _SeriesBranch(Model):
    """A series impedance stamped as a branch current, the form a zero impedance needs."""
    R: float = 0.0

    def mna(self, freq: Frequency) -> MNAStamp:
        nf = freq.npoints
        b = jnp.broadcast_to(jnp.array([[1.0], [-1.0]], dtype=complex), (nf, 2, 1))
        return MNAStamp(
            Y=jnp.zeros((nf, 2, 2), dtype=complex),
            B=b,
            C=jnp.swapaxes(b, 1, 2),
            D=jnp.full((nf, 1, 1), -self.R, dtype=complex),
        )


@pytest.mark.parametrize("R", [0.0, 25.0])
def test_model_s_from_mna_primary_domain(R):
    """`Model.s()` converts an MNA primary domain with `mna2s`."""
    z0 = jnp.array([50.0, 10.0 + 5.0j])
    s = _SeriesBranch(R=R).s(FREQ, z0=z0)
    # PiSectionCLC goes through ABCD, which is finite at zero impedance.
    ref = Resistor(R=R).s(FREQ, z0=z0) if R > 0 else PiSectionCLC(C1=0.0, L=0.0, C2=0.0).s(FREQ, z0=z0)
    np.testing.assert_allclose(s, ref, rtol=0, atol=1e-12)
