# tests/test_models/test_cascade.py
"""
Scattering `Cascade` and `RepeatedCascade` at singular junctions.

Floating multi-conductor networks leave a common mode at each junction that the
scattering reduction must not invert. The S reference for floating chains is a
50-digit nodal reduction of the ideal floating chain (`floating_chain_reference`).
`Circuit.from_chain` is not: its nodal solve is nearly singular on a floating
chain, so its 1e-12 GMIN and rounding move S by up to about 1e-8, differently
on different platforms. A real resonance must survive the same reduction
untouched; there the reference is the ABCD cascade of the resonant channel,
which has no junction inverse at all.

On floating chains, JAX gradients are checked against central finite
differences of the same model. `Circuit` stays the reference, for S and
gradient, on a grounded chain, where its solve is well conditioned.
"""
import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest

import pmrf as prf
from pmrf.models import (
    Capacitor, Cascade, Circuit, FloatingLine, FloatingTwoPort,
    GlobalScatteringCircuitSolver, GroundExposed, Isolator, Model,
    RepeatedCascade, RLGCLine, Tee,
)

FREQ = prf.Frequency(0.05, 3, 301, 'GHz')

#: Every tenth point of `FREQ`, for the slow high-precision reference. The
#: floating-mode error is flat in frequency, so the subsample loses nothing.
REFERENCE_FREQ = prf.Frequency(0.05, 3, 31, 'GHz')

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
# High-precision floating-chain reference
# ---------------------------------------------------------

def floating_modes(model):
    """
    Terminal sets a block's common mode lives on: each terminal pair of a
    floating two-port lift, and the whole block otherwise.
    """
    if isinstance(model, (FloatingLine, FloatingTwoPort)):
        return ((0, 1), (2, 3))
    return ((0, 1, 2, 3),)


def floating_chain_reference(models, freq, z0=50.0, dps=50):
    """
    S of a chain of floating 4-ports, reduced nodally at `dps` digits.

    Each block's S is evaluated in float64 by the model itself; everything after
    that is mpmath. The chain's terminals become nodes (ports 2, 3 of one block
    are ports 0, 1 of the next), and each block's S becomes a nodal admittance.

    The floating modes are made exact rather than regularised. Each block's
    admittance is projected so its `floating_modes` carry no current. This
    removes what leaks them in float64: the 1e-12 diagonal nudge in
    `renormalize_s` inside `FloatingLine.s`, and the 1e-12 GMIN inside the
    `Circuit` of `floating_transition`. The ideal floating chain then leaves
    each internal island of terminals at an arbitrary common potential that
    no port can observe. Grounding one node per island fixes that potential
    and changes no port quantity, so it is a choice of gauge, not an
    approximation. The rest of the nodal system is nonsingular and is
    eliminated exactly.

    None of this shares code or formulation with `Cascade`, which reduces
    scattering matrices pairwise and pins junction modes by an SVD.
    """
    blocks = [np.asarray(m.s(freq)) for m in models]
    n = 2 * len(models) + 2
    external = [0, 1, n - 2, n - 1]

    # Internal islands: terminals a floating mode joins, away from any port.
    island = list(range(n))
    def root(i):
        while island[i] != i:
            i = island[i]
        return i
    for b, model in enumerate(models):
        for mode in floating_modes(model):
            nodes = [2 * b + t for t in mode]
            if not set(nodes) & set(external):
                for node in nodes[1:]:
                    island[root(node)] = root(nodes[0])
    grounded = {root(i) for i in range(2, n - 2)}
    internal = [i for i in range(2, n - 2) if i not in grounded]

    out = np.empty(blocks[0].shape, dtype=complex)
    with mpmath.workdps(dps):
        eye = mpmath.eye(4)
        projectors = []
        for model in models:
            p = mpmath.eye(4)
            for mode in floating_modes(model):
                v = mpmath.matrix([int(t in mode) for t in range(4)])
                p -= v * v.T / len(mode)
            projectors.append(p)

        for k in range(len(out)):
            y = mpmath.zeros(n)
            for b, (s, p) in enumerate(zip(blocks, projectors)):
                s = mpmath.matrix(s[k].tolist())
                yb = p * (eye - s) * mpmath.inverse(eye + s) * p / z0
                for r in range(4):
                    for c in range(4):
                        y[2 * b + r, 2 * b + c] += yb[r, c]

            sub = lambda rows, cols: mpmath.matrix([[y[r, c] for c in cols] for r in rows])
            y_ii = sub(internal, internal)
            y_ii_inv = mpmath.inverse(y_ii)
            # A floating mode the islands missed would leave this singular to
            # working precision instead of failing loudly.
            assert mpmath.mnorm(y_ii, 1) * mpmath.mnorm(y_ii_inv, 1) < mpmath.mpf(10) ** (dps // 2)
            y_ext = sub(external, external) - sub(external, internal) * (y_ii_inv * sub(internal, external))
            out[k] = np.array(((eye - z0 * y_ext) * mpmath.inverse(eye + z0 * y_ext)).tolist(), dtype=complex)
    return out


# ---------------------------------------------------------
# Floating chains
# ---------------------------------------------------------

FLOATING_MIDDLES = {
    'transition': floating_transition,
    # The pinning argument is shown for reciprocal floating modes; a
    # non-reciprocal block at the floating junction is covered by regression.
    'non_reciprocal': lambda: FloatingTwoPort(Isolator(isolation=10.0)),
}

#: Bound on max |S_cascade - S_reference|, per middle block.
#:
#: The reference is the ideal floating chain. `Cascade` pins each junction
#: mode of the blocks as given, and those modes are not exactly floating, so
#: dropping their leak moves S by about the leak. `FloatingLine` leaks through
#: the 1e-12 diagonal nudge in `renormalize_s`, measured at 5.0e-13 of |Y|;
#: that alone sets the `non_reciprocal` error, 5.0e-13 at every length within
#: +-30% of the first line's. `floating_transition` also leaks through its
#: `Circuit`'s 1e-12 GMIN, which lifts the joint mode's junction singular value
#: to about 5e-12; the error over the same sweep is at most 1.1e-12. Each bound
#: is about 200 times its worst case. A pinning regression gives noise near
#: 1e-5, far above either.
FLOATING_S_TOL = {'transition': 2e-10, 'non_reciprocal': 1e-10}

#: Four `FloatingLine` sections: three junctions, each leaked only by the
#: `renormalize_s` nudge. Worst over a +-30% length scale is 5.05e-13.
REPEATED_S_TOL = 1e-10

#: Relative bound on a JAX gradient against a central difference of step
#: `FD_STEP`. Truncation, h**2 |f'''| / 6, dominates: at most 7.8e-9 over the
#: sweeps above, and 100 times that at h = 1e-5, as h**2 predicts. Rounding,
#: eps |f| / h, is about 1e-9. The bound is over 100 times their sum, and an
#: unpinned junction mode misses it by orders of magnitude: the transition
#: chain's gradient was once about +2e8 against -32.1.
FD_STEP = 1e-6
FD_REL_TOL = 1e-6


@pytest.mark.parametrize('name', FLOATING_MIDDLES)
def test_floating_chain_matches_reference(name):
    chain = lambda x: floating_chain(x, middle=FLOATING_MIDDLES[name]())

    s_cascade = Cascade(chain(0.07)).s(REFERENCE_FREQ)
    s_reference = floating_chain_reference(chain(0.07), REFERENCE_FREQ)
    assert np.abs(s_cascade - s_reference).max() < FLOATING_S_TOL[name]

    cascade = lambda x: loss(Cascade(chain(x)).s(FREQ))
    g_cascade = float(jax.grad(cascade)(0.07))
    assert g_cascade == pytest.approx(central_difference(cascade, 0.07, FD_STEP), rel=FD_REL_TOL)


def test_repeated_floating_sections_match_cascade_and_reference():
    section = lambda l: FloatingLine(RLGCLine(R=1.0, L=220e-9, G=1e-5, C=90e-12, length=l))
    member = section(0.05)
    base = jnp.linspace(0.03, 0.07, 4)

    def sections(scale):
        return [section(scale * l) for l in base]

    repeated = lambda scale: RepeatedCascade(member, {'floating.length': scale * base}, method='s')

    # Same junction reductions in the same order, so only rounding separates
    # them: at most 3.6e-15 over the length sweep.
    s_repeated = repeated(1.0).s(FREQ)
    assert np.abs(s_repeated - Cascade(sections(1.0)).s(FREQ)).max() < 1e-12

    # Three junctions, each pinning a `FloatingLine` mode leaked by the
    # renormalisation nudge, as for `non_reciprocal` above.
    s_reference = floating_chain_reference(sections(1.0), REFERENCE_FREQ)
    assert np.abs(repeated(1.0).s(REFERENCE_FREQ) - s_reference).max() < REPEATED_S_TOL

    repeated_loss = lambda x: loss(repeated(x).s(FREQ))
    g_repeated = float(jax.grad(repeated_loss)(1.0))
    assert g_repeated == pytest.approx(central_difference(repeated_loss, 1.0, FD_STEP), rel=FD_REL_TOL)


def test_grounded_chain_matches_circuit():
    """
    Without a floating mode, `Circuit` is a sound reference for S and gradient.

    The scattering solver without regularisation reads the same block S as
    `Cascade` and solves a well-conditioned system, so only rounding separates
    them: over +-30% of the first line's length, at most 8.0e-16 in S and
    1.9e-13 relative in the gradient. Each bound is about 100 times that. The
    default nodal solver is not used: `RLGCLine.y` and `RLGCLine.s` differ by up
    to about 1e-10, which would swamp the comparison.
    """
    def chain(length):
        return (
            RLGCLine(R=1.0, L=220e-9, G=1e-5, C=90e-12, length=length),
            Capacitor(1e-12),
            RLGCLine(R=4.0, L=310e-9, G=4e-5, C=120e-12, length=0.11),
        )

    solver = GlobalScatteringCircuitSolver(eps=0.0)
    weights = WEIGHTS[:, :2, :2]
    cascade = lambda x: loss(Cascade(chain(x)).s(FREQ), weights)
    circuit = lambda x: loss(Circuit.from_chain(chain(x), solver=solver).s(FREQ), weights)

    s_cascade = Cascade(chain(0.07)).s(FREQ)
    s_circuit = Circuit.from_chain(chain(0.07), solver=solver).s(FREQ)
    assert np.abs(s_cascade - s_circuit).max() < 1e-13

    g_cascade = float(jax.grad(cascade)(0.07))
    assert g_cascade == pytest.approx(float(jax.grad(circuit)(0.07)), rel=2e-11)


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
