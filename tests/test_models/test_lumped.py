import pytest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from skrf.media import DefinedGammaZ0
from pmrf.frequency import Frequency

from pmrf.models import (
    Short, Open, Match,
    Resistor, Capacitor, Inductor,
    ShuntResistor, ShuntCapacitor, ShuntInductor,
    CapacitorQ, InductorQ,
    Circuit, Port, GlobalMNACircuitSolver, GlobalScatteringCircuitSolver,
)

@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')

def test_ideal_loads(basic_freq):
    """Test 1-port ideal static loads (gamma representations)."""
    # Short circuit (Gamma = -1)
    s_short = Short().s(basic_freq)
    assert s_short.shape == (5, 1, 1)
    assert jnp.allclose(s_short, -1.0 + 0.0j)

    # Open circuit (Gamma = +1)
    s_open = Open().s(basic_freq)
    assert jnp.allclose(s_open, 1.0 + 0.0j)

    # Matched load (Gamma = 0)
    s_match = Match().s(basic_freq)
    assert jnp.allclose(s_match, 0.0 + 0.0j)

def test_series_resistor(basic_freq):
    """Test known RF limits for a series resistor."""
    res = Resistor(R=50.0)
    s = res.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    
    # In a 50 ohm system, a 50 ohm series resistor yields:
    # S21 = 2*Z0 / (R + 2*Z0) = 100 / 150 = 2/3
    assert jnp.allclose(s[:, 1, 0], 2.0/3.0, atol=1e-5)
    # S11 = R / (R + 2*Z0) = 50 / 150 = 1/3
    assert jnp.allclose(s[:, 0, 0], 1.0/3.0, atol=1e-5)

def test_shunt_resistor(basic_freq):
    """Test known RF limits for a shunt resistor."""
    res = ShuntResistor(R=50.0)
    s = res.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    
    # In a 50 ohm system, a 50 ohm shunt resistor yields:
    # S21 = 2*R / (2*R + Z0) = 100 / 150 = 2/3
    assert jnp.allclose(s[:, 1, 0], 2.0/3.0, atol=1e-5)
    # S11 = -Z0 / (2*R + Z0) = -50 / 150 = -1/3
    assert jnp.allclose(s[:, 0, 0], -1.0/3.0, atol=1e-5)

@pytest.mark.parametrize("model_class, param_kwargs", [
    (Capacitor, {'C': 1e-12}),
    (Inductor, {'L': 1e-9}),
    (ShuntCapacitor, {'C': 1e-12}),
    (ShuntInductor, {'L': 1e-9}),
])
def test_reactive_elements_execution(model_class, param_kwargs, basic_freq):
    """Ensure reactive lumped elements evaluate properly without shape or NaN errors."""
    model = model_class(**param_kwargs)
    s = model.s(basic_freq)
    
    assert s.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s))

def test_q_components_execution(basic_freq):
    """Ensure lumped elements with finite Quality Factor evaluate properly."""
    cap_q = CapacitorQ(C=1e-12, Q=50.0)
    ind_q = InductorQ(L=1e-9, Q=50.0)
    
    s_c = cap_q.s(basic_freq)
    s_i = ind_q.s(basic_freq)
    
    assert s_c.shape == (5, 2, 2)
    assert s_i.shape == (5, 2, 2)
    assert not jnp.any(jnp.isnan(s_c))
    assert not jnp.any(jnp.isnan(s_i))

# ---------------------------------------------------------
# Exact at zero (ADR-0007, #215, #223)
# ---------------------------------------------------------

#: The #215 reproduction's grid.
ZERO_FREQ = Frequency(start=0.1, stop=1.0, npoints=3, unit='GHz')

# The MNA solver's GMIN (1e-12 S to every node) moves S by about GMIN * Z0 = 5e-11 at a
# 50 ohm port; the scattering solver's own eps (1e-12) moves it by about 1e-12.
REG_ATOL = 1e-10
# dS/dL at L = 0 is of order w / (2 Z0) ~ 6e7; the regularisation shows up about two
# orders larger in it than in S.
GRAD_RTOL = 1e-8
# A central difference of scikit-rf with h = 1e-13 H has truncation error of order
# (w h / 2 Z0)^2 ~ 1e-10 relative, and roundoff of order eps / (h dS/dL) ~ 1e-11.
FD_STEP_L = 1e-13
FD_RTOL = 1e-7


def _two_port(element, solver):
    """`element` between two 50 ohm Ports, as in #215."""
    p0, p1 = Port(z0=50.0), Port(z0=50.0)
    return Circuit([[(p0, 0), (element, 0)], [(element, 1), (p1, 0)]], solver=solver)


def _s_and_ds(make_element, solver, x0=0.0):
    """S of the #215 two-port circuit, and its derivative with respect to the element value."""
    s_of = lambda x: _two_port(make_element(x), solver).s(ZERO_FREQ)
    return jax.jvp(s_of, (x0,), (1.0,))


def _loss_and_grad(make_element, solver, x0=0.0):
    """The #215 reproduction's loss, Σ|S11|², and its gradient."""
    def f(x):
        return jnp.sum(jnp.abs(_two_port(make_element(x), solver).s(ZERO_FREQ)[:, 0, 0]) ** 2)
    return jax.value_and_grad(f)(x0)


def _skrf_s_and_ds(network_of):
    """S of a scikit-rf network at a value of 0, and a central difference in the value."""
    s = network_of(0.0).s
    ds = (network_of(FD_STEP_L).s - network_of(-FD_STEP_L).s) / (2 * FD_STEP_L)
    return s, ds


_MEDIA = DefinedGammaZ0(ZERO_FREQ.to_skrf(), z0=50.0)
_W = ZERO_FREQ.w
_Q = 30.0

#: Series elements that short at a value of zero, and the same impedance built in
#: scikit-rf. `InductorQ` is compared with a series impedance of $\omega L (1/Q + j)$:
#: scikit-rf's `inductor_q` models a different loss, with a DC resistance and Q fixed
#: at one frequency.
ZERO_IMPEDANCE_ELEMENTS = {
    "Inductor": (lambda L: Inductor(L=L), lambda L: _MEDIA.inductor(L)),
    "InductorQ": (
        lambda L: InductorQ(L=L, Q=_Q),
        lambda L: _MEDIA.resistor(np.asarray(_W * L * (1 / _Q + 1j))),
    ),
}


@pytest.mark.parametrize("name", ZERO_IMPEDANCE_ELEMENTS)
def test_series_inductor_at_zero_matches_scattering_and_skrf(name):
    make_element, network_of = ZERO_IMPEDANCE_ELEMENTS[name]
    s, ds = _s_and_ds(make_element, GlobalMNACircuitSolver())
    ref_s, ref_ds = _s_and_ds(make_element, GlobalScatteringCircuitSolver())
    skrf_s, skrf_ds = _skrf_s_and_ds(network_of)

    assert np.all(np.isfinite(s)) and np.all(np.isfinite(ds))
    np.testing.assert_allclose(s, ref_s, rtol=0, atol=REG_ATOL)
    np.testing.assert_allclose(ds, ref_ds, rtol=GRAD_RTOL)
    np.testing.assert_allclose(s, skrf_s, rtol=0, atol=REG_ATOL)
    np.testing.assert_allclose(ds, skrf_ds, rtol=FD_RTOL)


@pytest.mark.parametrize("name", ZERO_IMPEDANCE_ELEMENTS)
def test_series_inductor_at_zero_215_loss(name):
    """The #215 reproduction: Σ|S11|² was NaN, with a NaN gradient, under MNA."""
    make_element, _ = ZERO_IMPEDANCE_ELEMENTS[name]
    value, grad = _loss_and_grad(make_element, GlobalMNACircuitSolver())
    ref_value, ref_grad = _loss_and_grad(make_element, GlobalScatteringCircuitSolver())

    # At a thru the loss is quadratic in the regularisation, and its gradient is
    # 2 Re(S11* dS11/dL), with S11 of order REG_ATOL.
    _, ds = _s_and_ds(make_element, GlobalScatteringCircuitSolver())
    grad_atol = 2 * REG_ATOL * np.sum(np.abs(ds[:, 0, 0]))
    np.testing.assert_allclose(value, ref_value, rtol=0, atol=REG_ATOL**2)
    np.testing.assert_allclose(grad, ref_grad, rtol=0, atol=grad_atol)


def test_capacitor_q_at_zero_has_finite_gradient_under_mna():
    """`CapacitorQ.y()` divided by C, which gave a NaN gradient at C = 0."""
    make_element = lambda C: CapacitorQ(C=C, Q=_Q)
    s, ds = _s_and_ds(make_element, GlobalMNACircuitSolver())
    ref_s, ref_ds = _s_and_ds(make_element, GlobalScatteringCircuitSolver())

    assert np.all(np.isfinite(ds))
    np.testing.assert_allclose(s, ref_s, rtol=0, atol=REG_ATOL)
    # dS/dC at C = 0 is of order w Z0 ~ 3e11; GMIN moves it relatively by about GMIN * Z0.
    np.testing.assert_allclose(ds, ref_ds, rtol=GRAD_RTOL)


@pytest.mark.parametrize("model", [lambda: InductorQ(L=1e-9, Q=0.0), lambda: CapacitorQ(C=1e-12, Q=0.0)],
                         ids=["InductorQ", "CapacitorQ"])
def test_quality_factor_is_positive(model):
    with pytest.raises(eqx.EquinoxRuntimeError, match='Positive'):
        model()


def test_inductor_at_dc_stamps_a_short():
    """At DC an `Inductor` is a short for any L; its stamp stays finite."""
    freq = Frequency(start=0.0, stop=1.0, npoints=3, unit='GHz')
    s = _two_port(Inductor(L=1e-9), GlobalMNACircuitSolver()).s(freq)
    np.testing.assert_allclose(s[0], [[0, 1], [1, 0]], rtol=0, atol=REG_ATOL)
