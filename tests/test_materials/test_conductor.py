import jax
import jax.numpy as jnp
import pytest
from scipy.constants import mu_0

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.materials import (
    BulkConductor,
    ConductorProperties,
    HammerstadRoughness,
    RoughConductor,
    as_conductor,
)


def test_conductor_exposes_one_property_record(freq):
    conductor = BulkConductor(sigma=1 / 1.68e-8, mu_r=2.0)

    properties = conductor.properties(freq)

    assert isinstance(properties, ConductorProperties)
    assert jnp.allclose(properties.sigma, 1 / 1.68e-8)
    assert jnp.allclose(properties.mu_r, 2.0)
    assert not hasattr(conductor, "surface_impedance")
    assert not hasattr(conductor, "skin_depth")


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=10.0, npoints=11, unit='GHz')


def test_bulk_surface_impedance(freq):
    sigma = 1 / 1.68e-8
    zs = BulkConductor(sigma).properties(freq).zs

    expected = jnp.sqrt(1j * freq.w * mu_0 / sigma)
    assert zs.shape == (freq.npoints,)
    assert jnp.allclose(zs, expected)
    # Real and imaginary parts are equal in the strong skin-effect regime.
    assert jnp.allclose(jnp.real(zs), jnp.imag(zs))


def test_bulk_sigma(freq):
    sigma = 1 / 1.68e-8
    conductor = BulkConductor(sigma)
    assert jnp.allclose(conductor.properties(freq).sigma, sigma)


def test_bulk_permeability(freq):
    assert jnp.allclose(BulkConductor(1 / 1.68e-8, mu_r=4.0).properties(freq).mu_r, 4.0)


def test_bulk_from_rho(freq):
    assert jnp.allclose(BulkConductor.from_rho(1 / 5.8e7).sigma, 5.8e7)


def test_bulk_reproduces_tesche_skin_terms(freq):
    """Re(Zs)/(2*pi*a) is the Tesche per-conductor skin resistance."""
    sigma, a = 1 / 1.68e-8, 0.56e-3

    zs = BulkConductor(sigma).properties(freq).zs
    R_skin = (1 / (2 * jnp.pi * a)) * jnp.sqrt(freq.w * mu_0 / (2 * sigma))
    L_skin = (1 / (2 * jnp.pi * a)) * jnp.sqrt(mu_0 / (2 * freq.w * sigma))

    assert jnp.allclose(jnp.real(zs) / (2 * jnp.pi * a), R_skin)
    assert jnp.allclose(jnp.imag(zs) / (2 * jnp.pi * a * freq.w), L_skin)


def test_bulk_reproduces_wheeler_resistance(freq):
    """Wheeler's R = (1/W)*sqrt(2*mu_0*rho*w) is two squares of Re(Zs)."""
    rho, W = 1.68e-8, 3e-3
    zs = BulkConductor(1 / rho).properties(freq).zs
    R_wheeler = (1 / W) * jnp.sqrt(2 * mu_0 * rho * freq.w)
    assert jnp.allclose(2 * jnp.real(zs) / W, R_wheeler)


def test_bulk_permeability_scaling(freq):
    plain = BulkConductor(1 / 1.68e-8).properties(freq).zs
    magnetic = BulkConductor(1 / 1.68e-8, mu_r=4.0).properties(freq).zs
    assert jnp.allclose(magnetic, 2.0 * plain)


def test_smooth_rough_conductor_is_bulk(freq):
    sigma = 1 / 1.68e-8
    rough = RoughConductor(sigma, roughness=0.0)
    assert jnp.allclose(
        rough.properties(freq).zs, BulkConductor(sigma).properties(freq).zs
    )


def test_roughness_increases_loss_and_saturates(freq):
    sigma = 1 / 1.68e-8
    rough = RoughConductor(sigma, roughness=1e-6)
    properties = rough.properties(freq)
    factor = rough.roughness.factor(freq, properties.sigma, properties.mu_r)

    assert jnp.all(factor > 1.0)
    assert jnp.all(factor < 2.0)
    # Rougher surfaces lose more, and the factor grows with frequency.
    assert jnp.all(jnp.diff(factor) > 0)
    assert jnp.allclose(rough.properties(freq).zs, factor * BulkConductor(sigma).properties(freq).zs)


def test_conductor_gradients_are_finite(freq):
    dc = Frequency.from_f(jnp.array([0.0, 1e9]))

    for material in (BulkConductor(1 / 1.68e-8), RoughConductor(1 / 1.68e-8, roughness=1e-6)):
        for axis in (freq, dc):
            grads = jax.grad(
                lambda m: jnp.sum(jnp.abs(m.properties(axis).zs))
            )(material)
            leaves = jax.tree_util.tree_leaves(prf.unwrap(grads))
            assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_unconstrained_gradients_are_finite_at_dc():
    """The bare sqrt at w = 0 must be guarded, not just masked by a constraint."""
    dc = Frequency.from_f(jnp.array([0.0, 1e9]))

    def loss(sigma):
        conductor = BulkConductor(sigma=prf.Unconstrained(sigma))
        return jnp.sum(jnp.real(conductor.properties(dc).zs))

    assert jnp.isfinite(jax.grad(loss)(1 / 1.68e-8))


def test_abstract_conductor_is_an_abc():
    with pytest.raises(TypeError):
        from pmrf.materials import AbstractConductor
        AbstractConductor()


def test_roughness_formulation_is_swappable():
    conductor = RoughConductor(1 / 1.68e-8, roughness=HammerstadRoughness(2e-6))
    assert isinstance(conductor.roughness, HammerstadRoughness)
    assert jnp.allclose(conductor.roughness.rms, 2e-6)


def test_as_conductor_converters():
    assert jnp.allclose(as_conductor(5.8e7).sigma, 5.8e7)
    rough = RoughConductor(5.8e7, roughness=1e-6)
    assert as_conductor(rough) is rough


def test_as_conductor_rejects_resistivity_regime_scalar():
    with pytest.raises(ValueError, match="from_rho"):
        as_conductor(1.68e-8)


def test_as_conductor_rejects_zero():
    """0 was the old rho idiom for a perfect conductor; as a sigma it means the opposite."""
    with pytest.raises(ValueError, match="ambiguous"):
        as_conductor(0.0)
