import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmrf.stats import LogUniform


def test_loguniform_requires_positive_ordered_bounds():
    for low, high in ((0, 10), (-1, 10), (10, 10), (10, 1)):
        with pytest.raises(ValueError, match="0 < low < high"):
            LogUniform(low, high)


def test_loguniform_density_and_closed_support():
    dist = LogUniform(1.0, 100.0)
    x = jnp.array([0.5, 1.0, 10.0, 100.0, 101.0])
    expected = [0.0, 1 / np.log(100), 1 / (10 * np.log(100)),
                1 / (100 * np.log(100)), 0.0]
    np.testing.assert_allclose(dist.prob(x), expected, rtol=1e-12)
    np.testing.assert_allclose(jnp.exp(dist.log_prob(x)), expected, rtol=1e-12)
    assert dist.support == (dist.low, dist.high)


def test_loguniform_unit_cube_and_cdf():
    dist = LogUniform(1.0, 100.0)
    np.testing.assert_allclose(dist.icdf(jnp.array([0.0, 0.5, 1.0])), [1, 10, 100])
    np.testing.assert_allclose(dist.cdf(jnp.array([0.5, 1.0, 10.0, 100.0, 101.0])),
                               [0, 0, 0.5, 1, 1])
    assert np.isneginf(dist.log_cdf(0.5))
    narrow = LogUniform(1e-12, 1e-9)
    assert narrow.icdf(0.0) == narrow.low
    assert narrow.icdf(1.0) == narrow.high


def test_loguniform_samples_have_equal_mass_per_decade():
    dist = LogUniform(1, 1000)
    keys = jax.random.split(jax.random.key(338), 6000)
    samples = jax.vmap(dist.sample)(keys)
    assert bool(jnp.all((samples >= 1.0) & (samples <= 1000.0)))
    counts = np.histogram(np.asarray(samples), bins=[1, 10, 100, 1000])[0]
    np.testing.assert_allclose(counts / counts.sum(), [1 / 3] * 3, atol=0.025)
    sample, log_prob = dist.sample_and_log_prob(jax.random.key(1))
    np.testing.assert_allclose(log_prob, dist.log_prob(sample))
