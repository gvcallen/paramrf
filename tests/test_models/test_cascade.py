# tests/test_models/test_cascade.py
"""
Scattering `Cascade` and `RepeatedCascade` at singular junctions.

Floating multi-conductor networks leave a common mode at each junction that the
scattering reduction must not invert. `Circuit` has no such junction, so it is
the reference for S and its JAX gradient. A real resonance must survive the
same reduction untouched; there the reference is the ABCD cascade of the
resonant channel, which has no junction inverse at all.

JAX gradients are checked against the reference gradient first. Central finite
differences are only a sanity check: before the fix, S carried noise at about
1e-5, so finite differences alone could not tell which answer was right.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pmrf as prf
from pmrf.models import (
    Capacitor, Cascade, Circuit, FloatingLine, FloatingTwoPort, GroundExposed,
    Isolator, Model, RepeatedCascade, RLGCLine, Tee,
)

FREQ = prf.Frequency(0.05, 3, 301, 'GHz')

# A fixed, dense linear functional of S, so a gradient sees every entry.
WEIGHTS = jnp.cos(jnp.arange(301 * 16)).reshape(301, 4, 4)


def loss(s, weights=WEIGHTS):
    return jnp.sum(weights * jnp.real(s)) + jnp.sum(weights[::-1] * jnp.imag(s))


def central_difference(fn, x, h):
    return (float(fn(x + h)) - float(fn(x - h))) / (2 * h)


def floating_transition():
    """A grounded line joined to a Tee: a floating 4-port."""
    line = RLGCLine(R=0.3, L=280e-9, G=3e-6, C=110e-12, length=0.02)
    t4 = Circuit.from_connection(GroundExposed(line), 2, Tee(), 0)
    return t4.renumbered((0, 1, 2, 3), (0, 2, 1, 3))


def floating_chain(length, middle=None):
    """Floating line, then a floating block, then another floating line."""
    first = RLGCLine(R=1.0, L=220e-9, G=1e-5, C=90e-12, length=length)
    last = RLGCLine(R=4.0, L=310e-9, G=4e-5, C=120e-12, length=0.11)
    middle = floating_transition() if middle is None else middle
    return (FloatingLine(first), middle, FloatingLine(last))


# ---------------------------------------------------------
# Floating chains
# ---------------------------------------------------------

def test_floating_chain_matches_circuit():
    chain = floating_chain(0.07)

    s_cascade = Cascade(chain).s(FREQ)
    s_circuit = Circuit.from_chain(chain).s(FREQ)

    # The pinned common mode is exactly unobservable, so only rounding remains.
    assert np.abs(s_cascade - s_circuit).max() < 1e-9


def test_floating_chain_gradient_matches_circuit():
    cascade = lambda x: loss(Cascade(floating_chain(x)).s(FREQ))
    circuit = lambda x: loss(Circuit.from_chain(floating_chain(x)).s(FREQ))

    g_cascade = float(jax.grad(cascade)(0.07))
    g_circuit = float(jax.grad(circuit)(0.07))
    g_fd = central_difference(cascade, 0.07, 1e-6)

    # Before the fix the cascade gradient was about +2e8 against -32.1.
    assert g_cascade == pytest.approx(g_circuit, rel=1e-7)
    assert g_cascade == pytest.approx(g_fd, rel=1e-6)


def test_non_reciprocal_floating_block_matches_circuit():
    """
    The pinning argument is shown for reciprocal floating modes; a
    non-reciprocal block at the floating junction is covered by regression.
    """
    chain = lambda x: floating_chain(x, middle=FloatingTwoPort(Isolator(isolation=10.0)))
    cascade = lambda x: loss(Cascade(chain(x)).s(FREQ))
    circuit = lambda x: loss(Circuit.from_chain(chain(x)).s(FREQ))

    s_cascade = Cascade(chain(0.07)).s(FREQ)
    s_circuit = Circuit.from_chain(chain(0.07)).s(FREQ)
    assert np.abs(s_cascade - s_circuit).max() < 1e-9

    g_cascade = float(jax.grad(cascade)(0.07))
    assert g_cascade == pytest.approx(float(jax.grad(circuit)(0.07)), rel=1e-7)
    assert g_cascade == pytest.approx(central_difference(cascade, 0.07, 1e-6), rel=1e-6)


def test_repeated_floating_sections_match_cascade_and_circuit():
    section = lambda l: FloatingLine(RLGCLine(R=1.0, L=220e-9, G=1e-5, C=90e-12, length=l))
    member = section(0.05)
    base = jnp.linspace(0.03, 0.07, 4)

    def sections(scale):
        return [section(scale * l) for l in base]

    repeated = lambda scale: RepeatedCascade(member, {'floating.length': scale * base}, method='s')
    reference = lambda scale: Circuit.from_chain(sections(scale))

    s_repeated = repeated(1.0).s(FREQ)
    assert np.abs(s_repeated - Cascade(sections(1.0)).s(FREQ)).max() < 1e-12
    assert np.abs(s_repeated - reference(1.0).s(FREQ)).max() < 1e-9

    g_repeated = float(jax.grad(lambda x: loss(repeated(x).s(FREQ)))(1.0))
    g_reference = float(jax.grad(lambda x: loss(reference(x).s(FREQ)))(1.0))
    g_fd = central_difference(lambda x: loss(repeated(x).s(FREQ)), 1.0, 1e-6)
    assert g_repeated == pytest.approx(g_reference, rel=1e-7)
    assert g_repeated == pytest.approx(g_fd, rel=1e-6)


# ---------------------------------------------------------
# Lossless resonance
# ---------------------------------------------------------

#: Series capacitance of the resonator; small enough that a 1e-8 singular-value
#: cutoff at the junction drops the resonance entirely.
C_RES = 1e-16

#: Peak of |S21| of the resonator below, located by grid refinement.
F_PEAK = 1333315555.7927082


def resonant_line(length=0.15):
    return RLGCLine(R=0.0, L=250e-9, G=0.0, C=100e-12, length=length)


def resonator(length=0.15):
    return (Capacitor(C_RES), resonant_line(length), Capacitor(C_RES))


class Paired(Model):
    """Two independent 2-port channels as one 4-port, (1a, 1b, 2a, 2b)."""
    first: Model
    second: Model

    @property
    def number_of_ports(self):
        return 4

    def s(self, freq, z0=50.0):
        s = jnp.zeros((freq.npoints, 4, 4), dtype=complex)
        s = s.at[:, 0::2, 0::2].set(self.first.s(freq, z0=z0))
        return s.at[:, 1::2, 1::2].set(self.second.s(freq, z0=z0))


def resonator_beside_channel(length=0.15):
    """The resonator in the first channel, a normal channel in the second."""
    normal = (Capacitor(1e-11), resonant_line(), Capacitor(1e-11))
    return tuple(Paired(r, n) for r, n in zip(resonator(length), normal))


@pytest.fixture
def peak():
    return prf.Frequency(F_PEAK, F_PEAK, 1, 'Hz')


def test_single_resonator_at_peak(peak):
    s_scattering = Cascade(resonator()).s(peak)
    s_abcd = Cascade(resonator(), method='a').s(peak)

    assert abs(s_abcd[0, 1, 0]) > 0.999
    # Conditioning at a Q this high, not the junction handling, limits agreement.
    assert np.abs(s_scattering - s_abcd).max() < 1e-7


def test_resonance_beside_normal_channel_at_peak(peak):
    """
    The resonance is observable from the ports, so it is never pinned.

    `Circuit.from_chain` is not the reference here: its nodal solve loses about
    3% of S at this peak with or without a cascade in the loop. The ABCD
    cascade of the resonant channel has no junction inverse at all.
    """
    s = Cascade(resonator_beside_channel()).s(peak)
    s_ref = Cascade(resonator(), method='a').s(peak)

    # A pinned resonance would give |S21| near 0 against a reference near 1.
    assert np.abs(s[:, 0::2, 0::2] - s_ref).max() < 1e-7


def test_resonance_beside_normal_channel_gradient(peak):
    weights = jnp.cos(jnp.arange(4.0)).reshape(1, 2, 2)
    cascade = lambda x: loss(Cascade(resonator_beside_channel(x)).s(peak)[:, 0::2, 0::2], weights)
    reference = lambda x: loss(Cascade(resonator(x), method='a').s(peak), weights)

    g_cascade = float(jax.grad(cascade)(0.15))
    assert g_cascade == pytest.approx(float(jax.grad(reference)(0.15)), rel=1e-6)
    # The peak is about 1 Hz wide at 1.3 GHz, so the step must be tiny and the
    # difference is limited by rounding in S.
    assert g_cascade == pytest.approx(central_difference(cascade, 0.15, 1e-13), rel=1e-4)
