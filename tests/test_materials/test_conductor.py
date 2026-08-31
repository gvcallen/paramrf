import jax
import jax.numpy as jnp
import pytest
from scipy.constants import mu_0

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.materials import BulkConductor, Hammerstad, RoughConductor, as_conductor


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=10.0, npoints=11, unit='GHz')


def test_bulk_surface_impedance(freq):
    rho = 1.68e-8
    zs = BulkConductor(rho).surface_impedance(freq)

    expected = jnp.sqrt(1j * freq.w * mu_0 / (1 / rho))
    assert zs.shape == (freq.npoints,)
    assert jnp.allclose(zs, expected)
    # Real and imaginary parts are equal in the strong skin-effect regime.
    assert jnp.allclose(jnp.real(zs), jnp.imag(zs))


def test_bulk_sigma_and_skin_depth(freq):
    rho = 1.68e-8
    conductor = BulkConductor(rho)
    assert jnp.allclose(conductor.sigma(freq), 1 / rho)
    assert jnp.allclose(
        conductor.skin_depth(freq), jnp.sqrt(2 * rho / (freq.w * mu_0))
    )


def test_bulk_permeability(freq):
    assert jnp.allclose(BulkConductor(1.68e-8, mu_rel=4.0).mu_r(freq), 4.0)


def test_bulk_from_sigma(freq):
    assert jnp.allclose(BulkConductor.from_sigma(5.8e7).rho, 1 / 5.8e7)


def test_bulk_reproduces_tesche_skin_terms(freq):
    """Re(Zs)/(2*pi*a) is the Tesche per-conductor skin resistance."""
    rho, a = 1.68e-8, 0.56e-3
    sigma = 1 / rho

    zs = BulkConductor(rho).surface_impedance(freq)
    R_skin = (1 / (2 * jnp.pi * a)) * jnp.sqrt(freq.w * mu_0 / (2 * sigma))
    L_skin = (1 / (2 * jnp.pi * a)) * jnp.sqrt(mu_0 / (2 * freq.w * sigma))

    assert jnp.allclose(jnp.real(zs) / (2 * jnp.pi * a), R_skin)
    assert jnp.allclose(jnp.imag(zs) / (2 * jnp.pi * a * freq.w), L_skin)


def test_bulk_reproduces_wheeler_resistance(freq):
    """Wheeler's R = (1/W)*sqrt(2*mu_0*rho*w) is two squares of Re(Zs)."""
    rho, W = 1.68e-8, 3e-3
    zs = BulkConductor(rho).surface_impedance(freq)
    R_wheeler = (1 / W) * jnp.sqrt(2 * mu_0 * rho * freq.w)
    assert jnp.allclose(2 * jnp.real(zs) / W, R_wheeler)


def test_bulk_permeability_scaling(freq):
    plain = BulkConductor(1.68e-8).surface_impedance(freq)
    magnetic = BulkConductor(1.68e-8, mu_rel=4.0).surface_impedance(freq)
    assert jnp.allclose(magnetic, 2.0 * plain)


def test_smooth_rough_conductor_is_bulk(freq):
    rough = RoughConductor(1.68e-8, roughness=0.0)
    assert jnp.allclose(
        rough.surface_impedance(freq), BulkConductor(1.68e-8).surface_impedance(freq)
    )


def test_roughness_increases_loss_and_saturates(freq):
    rho = 1.68e-8
    rough = RoughConductor(rho, roughness=1e-6)
    factor = rough.roughness.factor(rough.skin_depth(freq))

    assert jnp.all(factor > 1.0)
    assert jnp.all(factor < 2.0)
    # Rougher surfaces lose more, and the factor grows with frequency.
    assert jnp.all(jnp.diff(factor) > 0)
    assert jnp.allclose(rough.surface_impedance(freq), factor * BulkConductor(rho).surface_impedance(freq))


def test_conductor_gradients_are_finite(freq):
    dc = Frequency.from_f(jnp.array([0.0, 1e9]))

    for material in (BulkConductor(1.68e-8), RoughConductor(1.68e-8, roughness=1e-6)):
        for axis in (freq, dc):
            grads = jax.grad(
                lambda m: jnp.sum(jnp.abs(m.surface_impedance(axis)))
            )(material)
            leaves = jax.tree_util.tree_leaves(prf.unwrap(grads))
            assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_unconstrained_gradients_are_finite_at_dc():
    """The bare sqrt at w = 0 must be guarded, not just masked by a constraint."""
    dc = Frequency.from_f(jnp.array([0.0, 1e9]))

    def loss(rho):
        conductor = BulkConductor(rho=prf.Unconstrained(rho))
        return jnp.sum(jnp.real(conductor.surface_impedance(dc)))

    assert jnp.isfinite(jax.grad(loss)(1.68e-8))


def test_abstract_conductor_is_an_abc():
    with pytest.raises(TypeError):
        from pmrf.materials import AbstractConductor
        AbstractConductor()


def test_roughness_formulation_is_swappable():
    conductor = RoughConductor(1.68e-8, roughness=Hammerstad(2e-6))
    assert isinstance(conductor.roughness, Hammerstad)
    assert jnp.allclose(conductor.roughness.roughness, 2e-6)


def test_as_conductor_converters():
    assert jnp.allclose(as_conductor(1.68e-8).rho, 1.68e-8)
    rough = RoughConductor(1.68e-8, roughness=1e-6)
    assert as_conductor(rough) is rough
