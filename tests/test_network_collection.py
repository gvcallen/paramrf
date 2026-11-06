"""Tests for `NetworkCollection`."""

import signal
from contextlib import contextmanager

import numpy as np
import pytest
import skrf as rf

import pmrf as prf
from pmrf import NetworkCollection


@contextmanager
def deadline(seconds):
    """Fails the test instead of hanging if the block runs longer than `seconds`.
    NumPy coercion of a `pmrf.Frequency` never returns (#345)."""
    def expire(signum, frame):
        raise TimeoutError(f"did not return within {seconds} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def make_collection():
    f = np.linspace(1e6, 500e6, 201)
    return NetworkCollection([
        rf.Network(f=f, s=0.1 * np.exp(-1j * f / 1e8), z0=50, name="a"),
        rf.Network(f=f, s=0.2 * np.exp(-2j * f / 1e8), z0=50, name="b"),
    ])


def test_interpolate_self_accepts_pmrf_frequency():
    coll = make_collection()
    with deadline(10):
        coll.interpolate_self(prf.Frequency(50, 200, 151, "MHz"))
    for ntwk in coll:
        assert ntwk.s.shape == (151, 1, 1)
        assert np.allclose(ntwk.f[[0, -1]], [50e6, 200e6])
        assert ntwk.frequency.unit == "MHz"


@pytest.mark.parametrize("start, stop, unit", [(50, 200, "MHz"), (0.05, 0.2, "GHz")])
def test_interpolate_pmrf_frequency_matches_skrf_frequency(start, stop, unit):
    coll = make_collection()
    with deadline(10):
        from_pmrf = coll.interpolate(prf.Frequency(start, stop, 151, unit))
    from_skrf = coll.interpolate(rf.Frequency(start, stop, 151, unit))
    for got, want in zip(from_pmrf, from_skrf):
        assert got.frequency == want.frequency
        assert got.frequency.unit == want.frequency.unit == unit
        np.testing.assert_allclose(got.s, want.s)
