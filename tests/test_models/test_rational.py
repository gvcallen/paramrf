# tests/test_models/test_rational.py
import numpy as np
import pytest

from pmrf import Frequency
from pmrf.models import (
    BarycentricRational, PoleResidue, PolynomialRatio, StateSpace,
)
from pmrf.rf.conversions import renormalize_s


@pytest.fixture
def freq():
    # Contains 60 and 140 MHz exactly.
    return Frequency(50, 200, 151, unit='MHz')


def jw(freq):
    return 1j * np.asarray(freq.w)


def barycentric_setup():
    s_i = 1j * 2 * np.pi * np.array([60e6, 100e6, 140e6])
    w_i = np.array([1.0, -2.0 + 0.5j, 1.5])
    f_i = np.array([0.3 + 0.1j, -0.2 + 0.4j, 0.5 - 0.3j])
    return s_i, w_i, f_i


def barycentric_reference(s, s_i, w_i, f_i):
    basis = w_i[None, :] / (s[:, None] - s_i[None, :])
    return (basis @ f_i) / basis.sum(axis=1)


@pytest.mark.parametrize('cls', [PolynomialRatio, PoleResidue, StateSpace, BarycentricRational])
def test_is_concrete(cls):
    assert not cls.__abstractmethods__


def test_barycentric_matches_closed_form(freq):
    s_i, w_i, f_i = barycentric_setup()
    model = BarycentricRational(support_points=s_i, weights=w_i, values=f_i)

    s = jw(freq)
    off_support = ~np.isclose(s[:, None], s_i[None, :], rtol=0, atol=1.0).any(axis=1)
    ref = barycentric_reference(s[off_support], s_i, w_i, f_i)

    H = np.asarray(model.s(freq))[:, 0, 0]
    np.testing.assert_allclose(H[off_support], ref, rtol=1e-10, atol=1e-12)


def test_barycentric_interpolates_support_points():
    s_i, w_i, f_i = barycentric_setup()
    model = BarycentricRational(support_points=s_i, weights=w_i, values=f_i)

    support_freq = Frequency.from_f(np.array([60e6, 100e6, 140e6]), unit='Hz')
    H = np.asarray(model.s(support_freq))[:, 0, 0]
    # The eps guard makes the support term dominate rather than selecting
    # f_i outright, so the result is f_i to within an ulp, not bit-exact.
    np.testing.assert_allclose(H, f_i, rtol=0, atol=1e-15)


def test_pole_residue_matches_closed_form(freq):
    p = np.array([-1e8 + 6e8j, -1e8 - 6e8j, -5e8])
    r = np.array([2e7 + 1e7j, 2e7 - 1e7j, 1e8])
    D = 0.1
    model = PoleResidue(poles=p, residues=r, D=D)

    s = jw(freq)
    ref = D + (r[None, :] / (s[:, None] - p[None, :])).sum(axis=1)

    H = np.asarray(model.s(freq))[:, 0, 0]
    np.testing.assert_allclose(H, ref, rtol=1e-10, atol=1e-12)


def test_polynomial_ratio_matches_closed_form(freq):
    # Coefficients in increasing order of degree; s scaled so terms are O(1).
    A = np.array([0.2, 1e-9, 3e-19])
    B = np.array([1.0, 2e-9, 1e-18])
    model = PolynomialRatio(A=A, B=B)

    s = jw(freq)
    ref = np.polyval(A[::-1], s) / np.polyval(B[::-1], s)

    H = np.asarray(model.s(freq))[:, 0, 0]
    np.testing.assert_allclose(H, ref, rtol=1e-10, atol=1e-12)


def test_state_space_matches_closed_form(freq):
    A = np.array([[-1e8, 6e8], [-6e8, -1e8]])
    B = np.array([[1e8, 0.0], [0.0, 5e7]])
    C = np.array([[0.5, 0.2], [0.1, 0.4]])
    D = np.array([[0.1, 0.0], [0.0, -0.05]])
    model = StateSpace(A=A, B=B, C=C, D=D)

    s = jw(freq)
    I = np.eye(2)
    ref = np.stack([C @ np.linalg.solve(sk * I - A, B) + D for sk in s])

    H = np.asarray(model.s(freq))
    np.testing.assert_allclose(H, ref, rtol=1e-10, atol=1e-12)


def test_renormalises_to_requested_z0(freq):
    s_i, w_i, f_i = barycentric_setup()
    model = BarycentricRational(support_points=s_i, weights=w_i, values=f_i, z0=50.0)

    intrinsic = np.asarray(model.s(freq, z0=50.0))
    renormalised = np.asarray(model.s(freq, z0=75.0))
    ref = np.asarray(renormalize_s(intrinsic, 50.0, 75.0, 'power', 'power'))

    assert not np.allclose(renormalised, intrinsic)
    np.testing.assert_allclose(renormalised, ref, rtol=1e-10, atol=1e-12)
