# tests/test_models/test_topologies.py
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from skrf.media import DefinedGammaZ0

from pmrf.frequency import Frequency
from pmrf.models import (
    PiSectionCLC, BoxSectionCLCC, TSectionLCL, LSectionLC,
    Circuit, Port, Ground, Capacitor, Inductor,
    GlobalMNACircuitSolver, GlobalScatteringCircuitSolver,
)

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

# ---------------------------------------------------------
# Pi-CLC Tests
# ---------------------------------------------------------

def test_piclc_general(basic_freq):
    """Test standard Pi-CLC execution and reciprocity."""
    model = PiSectionCLC(C1=1e-12, L=1e-9, C2=2e-12)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))
    
    # Passive lumped network should be strictly reciprocal (S21 == S12)
    assert jnp.allclose(s[:, 1, 0], s[:, 0, 1], atol=1e-6)

def test_piclc_zero_inductance(basic_freq):
    """Test the edge case where L = 0."""
    # When L=0, the network is just C1 and C2 in parallel to ground
    model = PiSectionCLC(C1=1e-12, L=0.0, C2=2e-12)
    a_mat = model.a(basic_freq)
    
    # ABCD for a shunt admittance Y is [[1, 0], [Y, 1]]
    Y_total = 1j * basic_freq.w * (1e-12 + 2e-12)
    
    assert jnp.allclose(a_mat[:, 0, 0], 1.0)
    assert jnp.allclose(a_mat[:, 0, 1], 0.0)
    assert jnp.allclose(a_mat[:, 1, 0], Y_total)
    assert jnp.allclose(a_mat[:, 1, 1], 1.0)

# ---------------------------------------------------------
# Box-CLCC Tests
# ---------------------------------------------------------

def test_boxclcc_general(basic_freq):
    """Test standard Box-CLCC execution (4-port)."""
    model = BoxSectionCLCC(C1=1e-12, L=1e-9, C2=1e-12, C3=0.5e-12)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 4, 4)
    assert not jnp.any(jnp.isnan(s))

def test_boxclcc_zero_inductance(basic_freq):
    """Test the edge case where L = 0."""
    model = BoxSectionCLCC(L=0.0, C1=1e-12, C2=1e-12, C3=1e-12)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 4, 4)
    assert not jnp.any(jnp.isnan(s))

# ---------------------------------------------------------
# Tee-LCL Tests
# ---------------------------------------------------------

def test_teelcl_general(basic_freq):
    """Test standard Tee-LCL execution and reciprocity."""
    model = TSectionLCL(L1=1e-9, C=1e-12, L2=2e-9)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))
    assert jnp.allclose(s[:, 1, 0], s[:, 0, 1], atol=1e-6)

def test_teelcl_zero_capacitance(basic_freq):
    """Test the jax.lax.cond edge case where C = 0."""
    # When C=0, the network is just L1 and L2 in series
    model = TSectionLCL(L1=1e-9, C=0.0, L2=2e-9)
    a_mat = model.a(basic_freq)
    
    # ABCD for a series impedance Z is [[1, Z], [0, 1]]
    Z_total = 1j * basic_freq.w * (1e-9 + 2e-9)
    
    assert jnp.allclose(a_mat[:, 0, 0], 1.0)
    assert jnp.allclose(a_mat[:, 0, 1], Z_total)
    assert jnp.allclose(a_mat[:, 1, 0], 0.0)
    assert jnp.allclose(a_mat[:, 1, 1], 1.0)

# ---------------------------------------------------------
# L-Section Tests
# ---------------------------------------------------------

def test_lsection_general(basic_freq):
    """Test standard L-Section execution."""
    model = LSectionLC(L=1e-9, C=1e-12)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))

def test_lsection_thru(basic_freq):
    """Test the ideal thru case where L=0 and C=0."""
    model = LSectionLC(L=0.0, C=0.0)
    s = model.s(basic_freq)
    
    # An ideal thru-line has S11=0 and S21=1
    assert jnp.allclose(s[:, 0, 0], 0.0 + 0.0j, atol=1e-6)
    assert jnp.allclose(s[:, 1, 0], 1.0 + 0.0j, atol=1e-6)


# ---------------------------------------------------------
# Exact at zero (ADR-0007, #215, #223)
# ---------------------------------------------------------

#: The #215 reproduction's grid.
ZERO_FREQ = Frequency(start=0.1, stop=1.0, npoints=3, unit='GHz')

# The MNA solver's GMIN (1e-12 S to every node) moves S by about GMIN * Z0 = 5e-11 at a
# 50 ohm port, and the scattering solver's own eps (1e-12) by about 1e-12.
REG_ATOL = 1e-10
# dL is a difference of nearly cancelling terms at a short, so the regularisation shows
# up about two orders larger in it; observed agreement is ~1e-10 relative.
GRAD_RTOL = 1e-8
# A central difference of scikit-rf with h = 1e-13 H: truncation error of order
# (w h / 2 Z0)^2 ~ 1e-10 relative, and roundoff of order eps / (h dS/dL) ~ 1e-11.
FD_STEP_L = 1e-13
FD_RTOL = 1e-7


def _loss_and_grad(circuit_of_L, L0=0.0):
    """The #215 reproduction's loss, Σ|S11|², and its derivative with respect to L."""
    def f(L):
        return jnp.sum(jnp.abs(circuit_of_L(L).s(ZERO_FREQ)[:, 0, 0]) ** 2)
    return jax.value_and_grad(f)(L0)


def _skrf_pi_loss_and_grad():
    """Σ|S11|² of C - L - C in scikit-rf at L = 0, and a central difference in L."""
    media = DefinedGammaZ0(ZERO_FREQ.to_skrf(), z0=50.0)
    def f(L):
        ntwk = media.shunt_capacitor(1e-12) ** media.inductor(L) ** media.shunt_capacitor(1e-12)
        return np.sum(np.abs(ntwk.s[:, 0, 0]) ** 2)
    return f(0.0), (f(FD_STEP_L) - f(-FD_STEP_L)) / (2 * FD_STEP_L)


def _pi_circuit(L, solver):
    """The #215 reproduction: `PiSectionCLC` between two 50 ohm Ports."""
    pi = PiSectionCLC(C1=1e-12, L=L, C2=1e-12)
    p0, p1 = Port(z0=50.0), Port(z0=50.0)
    return Circuit([[(p0, 0), (pi, 0)], [(pi, 1), (p1, 0)]], solver=solver)


def _box_circuit(L, solver):
    """`BoxSectionCLCC` with ports 1 and 3 grounded, which is the Pi-section above."""
    box = BoxSectionCLCC(C1=1e-12, L=L, C2=1e-12, C3=1e-12)
    p0, p1, g = Port(z0=50.0), Port(z0=50.0), Ground()
    return Circuit([[(p0, 0), (box, 0)], [(p1, 0), (box, 2)], [(g, 0), (box, 1), (box, 3)]], solver=solver)


@pytest.mark.parametrize("circuit", [_pi_circuit, _box_circuit], ids=["PiSectionCLC", "BoxSectionCLCC"])
def test_clc_at_zero_inductance_matches_scattering_and_skrf(circuit):
    """Under MNA, `PiSectionCLC(L=0)` was 0.2% off with dL = -2.8e17, and `BoxSectionCLCC(L=0)` had dL = 0."""
    value, grad = _loss_and_grad(lambda L: circuit(L, GlobalMNACircuitSolver()))
    ref_value, ref_grad = _loss_and_grad(lambda L: circuit(L, GlobalScatteringCircuitSolver()))
    skrf_value, skrf_grad = _skrf_pi_loss_and_grad()

    np.testing.assert_allclose(value, ref_value, rtol=REG_ATOL)
    np.testing.assert_allclose(grad, ref_grad, rtol=GRAD_RTOL)
    np.testing.assert_allclose(value, skrf_value, rtol=REG_ATOL)
    np.testing.assert_allclose(grad, skrf_grad, rtol=FD_RTOL)
    # About -4.8e7 here.
    assert grad < -4e7


def _discrete_box(L):
    """`BoxSectionCLCC` built from discrete elements, in a scattering circuit."""
    c1, c2, c3, ind = Capacitor(C=1e-12), Capacitor(C=2e-12), Capacitor(C=0.5e-12), Inductor(L=L)
    ports = [Port(z0=50.0) for _ in range(4)]
    return Circuit([
        [(ports[0], 0), (c1, 0), (ind, 0)],
        [(ports[1], 0), (c1, 1), (c3, 0)],
        [(ports[2], 0), (c2, 0), (ind, 1)],
        [(ports[3], 0), (c2, 1), (c3, 1)],
    ], solver=GlobalScatteringCircuitSolver())


@pytest.mark.parametrize("L0", [0.0, 1e-9], ids=["L=0", "L=1nH"])
def test_boxclcc_matches_discrete_elements(L0):
    """`BoxSectionCLCC.s()` comes from its MNA stamp, and is exact, with its dS/dL, at L = 0 and DC."""
    freq = Frequency(start=0.0, stop=1.0, npoints=5, unit='GHz')
    box_s = lambda L: BoxSectionCLCC(C1=1e-12, L=L, C2=2e-12, C3=0.5e-12).s(freq)
    s, ds = jax.jvp(box_s, (L0,), (1.0,))
    ref_s, ref_ds = jax.jvp(lambda L: _discrete_box(L).s(freq), (L0,), (1.0,))

    assert np.all(np.isfinite(s)) and np.all(np.isfinite(ds))
    np.testing.assert_allclose(s, ref_s, rtol=0, atol=REG_ATOL)
    # dS/dL is of order w Z0 ~ 3e11 at 1 GHz; the scattering eps moves it by about eps.
    np.testing.assert_allclose(ds, ref_ds, rtol=0, atol=GRAD_RTOL * np.max(np.abs(ref_ds)))


@pytest.mark.parametrize("L", [-1e-9, 0.0, 1e-9])
def test_boxclcc_is_finite_at_dc(L):
    freq = Frequency(start=0.0, stop=1.0, npoints=3, unit='GHz')
    s = BoxSectionCLCC(C1=1e-12, L=L, C2=1e-12, C3=1e-12).s(freq)
    assert np.all(np.isfinite(s))
