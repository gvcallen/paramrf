import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from pmrf.frequency import Frequency
from pmrf.parameters import Unconstrained
from pmrf.models import (
    Transformer, CentreTappedTransformer, Autotransformer, Balun,
    SourceConverter, MixedModeConverter, CoupledInductors, Inductor,
)


@pytest.fixture
def basic_freq():
    return Frequency(start=1.0, stop=10.0, npoints=5, unit='GHz')


def terminal_waves(s_mat, z0=50.0, ntrials=4):
    """Excite an ideal network with random incident waves and return the (V, I) it allows."""
    n = s_mat.shape[0]
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, ntrials)) + 1j * rng.normal(size=(n, ntrials))
    b = np.asarray(s_mat) @ a

    v = np.sqrt(z0) * (a + b)
    i = (a - b) / np.sqrt(z0)
    return v, i


def terminated(s_mat, port, gamma):
    """Reduce an N-port by terminating a single port in a reflection coefficient."""
    s_mat = np.asarray(s_mat)
    keep = [k for k in range(s_mat.shape[0]) if k != port]

    coupling = np.outer(s_mat[keep, port], s_mat[port, keep])
    return s_mat[np.ix_(keep, keep)] + coupling * gamma / (1.0 - s_mat[port, port] * gamma)


def assert_lossless_and_reciprocal(s_mat):
    n = s_mat.shape[0]
    assert np.allclose(np.asarray(s_mat).conj().T @ np.asarray(s_mat), np.eye(n))
    assert np.allclose(s_mat, np.asarray(s_mat).T)


def test_transformer_terminal_relations(basic_freq):
    """An ideal 1:N transformer steps up voltage by N and steps down current by N."""
    N = 2.0
    s = Transformer(N=N).s(basic_freq)

    assert s.shape == (5, 4, 4)
    assert_lossless_and_reciprocal(s[0])

    v, i = terminal_waves(s[0])

    # V3 - V4 = N (V1 - V2)
    assert np.allclose(v[2] - v[3], N * (v[0] - v[1]))

    # Both windings are isolated, and the secondary current is the primary's over N
    assert np.allclose(i[0], -i[1])
    assert np.allclose(i[2], -i[3])
    assert np.allclose(i[2], -i[0] / N)


def test_transformer_is_impedance_independent(basic_freq):
    """The S-parameters of an ideal transformer do not depend on the reference impedance."""
    model = Transformer(N=3.0)
    assert np.allclose(model.s(basic_freq, z0=50.0), model.s(basic_freq, z0=75.0))


def test_transformer_turns_ratio_is_tunable(basic_freq):
    """The turns ratio is a parameter, so it can be differentiated and fitted."""
    single_freq = Frequency(start=1.0, stop=1.0, npoints=1, unit='GHz')

    def s13_real(n_val):
        return jnp.real(Transformer(N=n_val).s(single_freq)[0, 0, 2])

    N = Unconstrained(2.0)

    # S13 = N / (1 + N^2), so dS13/dN = (1 - N^2) / (1 + N^2)^2
    expected = (1.0 - N**2) / (1.0 + N**2)**2
    assert jnp.isclose(jax.grad(s13_real)(N), expected, atol=1e-6)


def test_balun_terminal_relations(basic_freq):
    """A 1:N balun converts the single-ended port 1 into the balanced pair (2, 3)."""
    N = 3.0
    s = Balun(N=N).s(basic_freq)

    assert s.shape == (5, 3, 3)
    assert_lossless_and_reciprocal(s[0])

    v, i = terminal_waves(s[0])

    # V2 - V3 = N * V1, with the balanced pair isolated from the single-ended side
    assert np.allclose(v[1] - v[2], N * v[0])
    assert np.allclose(i[1], -i[2])
    assert np.allclose(i[1], -i[0] / N)


def test_balun_matches_source_converter(basic_freq):
    """A 1:1 balun is the library's existing source converter."""
    assert np.allclose(Balun(N=1.0).s(basic_freq), SourceConverter().s(basic_freq))


def test_autotransformer_terminal_relations(basic_freq):
    """An autotransformer taps a single winding, so it provides no isolation."""
    N = 4.0
    s = Autotransformer(N=N).s(basic_freq)

    assert s.shape == (5, 3, 3)
    assert_lossless_and_reciprocal(s[0])

    v, i = terminal_waves(s[0])

    # V1 - V3 = N (V2 - V3) across the full winding and its tapped section
    assert np.allclose(v[0] - v[2], N * (v[1] - v[2]))

    # A single winding has no separate return path, so the terminal currents sum to zero
    assert np.allclose(i.sum(axis=0), 0.0)


def test_centre_tapped_transformer_divides_the_primary(basic_freq):
    """The tap splits the primary in the ratio set by `tap`."""
    N, tap = 2.0, 0.3
    s = CentreTappedTransformer(N=N, tap=tap).s(basic_freq)

    assert s.shape == (5, 5, 5)
    assert_lossless_and_reciprocal(s[0])

    v, _ = terminal_waves(s[0])
    secondary = v[2] - v[3]

    # The upper section carries `tap` of the primary turns, and the lower the rest
    assert np.allclose(v[0] - v[4], (tap / N) * secondary)
    assert np.allclose(v[4] - v[1], ((1.0 - tap) / N) * secondary)


def test_centre_tapped_transformer_reduces_to_transformer(basic_freq):
    """With the tap left open, the centre-tapped transformer is a plain transformer."""
    s = CentreTappedTransformer(N=2.0, tap=0.5).s(basic_freq)
    reduced = terminated(s[0], port=4, gamma=1.0)

    assert np.allclose(reduced, Transformer(N=2.0).s(basic_freq)[0])


def test_coupled_inductors_structure(basic_freq):
    """Coupled inductors form a lossless, reciprocal 4-terminal network of isolated pairs."""
    model = CoupledInductors(L1=1e-9, L2=4e-9, k=0.5)

    y = model.y(basic_freq)
    s = model.s(basic_freq)

    assert s.shape == (5, 4, 4)
    assert not np.any(np.isnan(s))
    assert_lossless_and_reciprocal(s[0])

    # Each winding is floating, so no terminal pair carries a net current
    assert np.allclose(y.sum(axis=-1), 0.0)
    assert np.allclose(y.sum(axis=-2), 0.0)


def test_coupled_inductors_uncoupled(basic_freq):
    """At k = 0 the windings reduce to two independent inductors."""
    y = np.asarray(CoupledInductors(L1=1e-9, L2=4e-9, k=0.0).y(basic_freq))

    assert np.allclose(y[:, :2, :2], Inductor(L=1e-9).y(basic_freq))
    assert np.allclose(y[:, 2:, 2:], Inductor(L=4e-9).y(basic_freq))
    assert np.allclose(y[:, :2, 2:], 0.0)


def test_coupled_inductors_approach_ideal_transformer(basic_freq):
    """Perfect coupling with a large magnetizing inductance is an ideal transformer."""
    L1, L2 = 1e-3, 4e-3
    coupled = CoupledInductors(L1=L1, L2=L2, k=1.0 - 1e-9)
    ideal = Transformer(N=np.sqrt(L2 / L1))

    assert np.allclose(coupled.s(basic_freq), ideal.s(basic_freq), atol=1e-3)


def cascade_through_modes(s_a, s_b, nmodal=2):
    """Cascade two 4-ports by tying the last `nmodal` ports of each together."""
    s_a, s_b = np.asarray(s_a), np.asarray(s_b)
    p = slice(0, s_a.shape[0] - nmodal)
    m = slice(s_a.shape[0] - nmodal, s_a.shape[0])
    npp = s_a.shape[0] - nmodal

    # Internal waves satisfy b_m = S_mp a_p + S_mm a_m, with a_m of one side the b_m of the other
    lhs = np.block([
        [np.eye(nmodal), -s_a[m, m]],
        [-s_b[m, m], np.eye(nmodal)],
    ])
    rhs = np.block([
        [s_a[m, p], np.zeros((nmodal, npp))],
        [np.zeros((nmodal, npp)), s_b[m, p]],
    ])
    b_int = np.linalg.solve(lhs, rhs)

    # External ports see their own reflection plus what returns from the interface
    ext = np.block([
        [s_a[p, p], np.zeros((npp, npp))],
        [np.zeros((npp, npp)), s_b[p, p]],
    ])
    coupling = np.block([
        [s_a[p, m], np.zeros((npp, nmodal))],
        [np.zeros((npp, nmodal)), s_b[p, m]],
    ])
    return ext + coupling @ b_int[[*range(nmodal, 2 * nmodal), *range(nmodal)], :]


def test_mixed_mode_converter_is_unitary_and_reciprocal(basic_freq):
    """The converter is lossless, reciprocal and matched at every port."""
    s = MixedModeConverter().s(basic_freq)

    assert s.shape == (5, 4, 4)
    assert_lossless_and_reciprocal(s[0])

    # Both the physical and the modal pair are perfectly matched
    assert np.allclose(s[0, :2, :2], 0.0)
    assert np.allclose(s[0, 2:, 2:], 0.0)


def test_mixed_mode_converter_modal_selectivity(basic_freq):
    """Even excitation appears only at the common port, and odd only at the differential."""
    s = np.asarray(MixedModeConverter().s(basic_freq)[0])
    root2 = np.sqrt(2.0)

    # Ports 1 and 2 driven in phase excite port 3 alone
    b_even = s @ np.array([1.0, 1.0, 0.0, 0.0])
    assert np.allclose(b_even, [0.0, 0.0, root2, 0.0])

    # Ports 1 and 2 driven in antiphase excite port 4 alone
    b_odd = s @ np.array([1.0, -1.0, 0.0, 0.0])
    assert np.allclose(b_odd, [0.0, 0.0, 0.0, root2])

    # Driving the common port drives the physical pair in phase, and vice versa
    assert np.allclose(s @ np.array([0.0, 0.0, 1.0, 0.0]), [1 / root2, 1 / root2, 0.0, 0.0])
    assert np.allclose(s @ np.array([0.0, 0.0, 0.0, 1.0]), [1 / root2, -1 / root2, 0.0, 0.0])


def test_mixed_mode_converter_round_trip_is_identity(basic_freq):
    """Two converters back-to-back through their modal ports form a through connection."""
    s = MixedModeConverter().s(basic_freq)[0]
    cascaded = cascade_through_modes(s, s)

    expected = np.block([
        [np.zeros((2, 2)), np.eye(2)],
        [np.eye(2), np.zeros((2, 2))],
    ])
    assert np.allclose(cascaded, expected)


def test_mixed_mode_converter_broadcasts_over_frequency():
    """The S-parameters are frequency-independent but span the whole grid."""
    freq = Frequency(start=0.1, stop=20.0, npoints=17, unit='GHz')
    s = np.asarray(MixedModeConverter().s(freq))

    assert s.shape == (17, 4, 4)
    assert np.allclose(s, s[0])


def test_mixed_mode_converter_requires_equal_reference_impedances(basic_freq):
    """The modal transform is only power-normalized when every port shares one z0."""
    model = MixedModeConverter()

    # A single shared impedance is fine, whatever its value
    assert np.allclose(model.s(basic_freq, z0=50.0), model.s(basic_freq, z0=75.0))
    assert np.allclose(model.s(basic_freq, z0=[50.0] * 4), model.s(basic_freq, z0=50.0))

    with pytest.raises(eqx.EquinoxRuntimeError, match="must be equal"):
        model.s(basic_freq, z0=[50.0, 75.0, 50.0, 50.0])
