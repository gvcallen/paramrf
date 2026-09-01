import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import epsilon_0

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.materials import (
    DielectricProperties,
    ColeCole,
    ConstantDielectric,
    DebyePole,
    DjordjevicSarkar,
    MultipoleDebye,
    TabulatedDielectric,
    as_dielectric,
)


def test_dielectric_properties_separate_static_conductivity(freq):
    material = ConstantDielectric(ep_r=4.3, tand=0.02, sigma=0.01, mu_r=2.0)

    properties = material.properties(freq)

    assert isinstance(properties, DielectricProperties)
    assert jnp.allclose(properties.ep_r, 4.3 * (1 - 0.02j))
    assert jnp.allclose(properties.mu_r, 2.0)
    assert jnp.allclose(properties.sigma, 0.01)
    assert not hasattr(material, "epsilon_r")


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=10.0, npoints=11, unit='GHz')


@pytest.fixture
def wideband_freq():
    return Frequency.from_f(jnp.logspace(3, 11, 201))


def test_constant_dielectric(freq):
    eps = ConstantDielectric(4.3, 0.02).properties(freq).ep_r
    assert eps.shape == (freq.npoints,)
    assert jnp.allclose(eps, 4.3 * (1 - 0.02j))


def test_constant_dielectric_loss_tangent(freq):
    eps = ConstantDielectric(4.3, 0.02).properties(freq).ep_r
    assert jnp.allclose(-jnp.imag(eps) / jnp.real(eps), 0.02)


def test_conductivity_gives_frequency_independent_conductance(freq):
    """Static conductivity is retained separately from permittivity."""
    sigma = 0.01
    properties = ConstantDielectric(4.3, sigma=sigma).properties(freq)
    assert jnp.allclose(jnp.imag(properties.ep_r), 0.0)
    assert jnp.allclose(properties.sigma, sigma)


def test_loss_terms_are_additive(freq):
    both = ConstantDielectric(4.3, 0.02, 0.01).properties(freq).ep_r
    tand_only = ConstantDielectric(4.3, 0.02).properties(freq).ep_r
    sigma_only = ConstantDielectric(4.3, sigma=0.01).properties(freq).ep_r
    assert jnp.allclose(both, tand_only + sigma_only - 4.3)


def test_conductivity_guarded_at_dc():
    dc = Frequency.from_f(jnp.array([0.0, 1e9]))
    eps = ConstantDielectric(4.3, sigma=0.01).properties(dc).ep_r
    assert jnp.all(jnp.isfinite(eps))

    grad = jax.grad(
        lambda s: jnp.sum(jnp.imag(ConstantDielectric(4.3, sigma=s).properties(dc).ep_r))
    )(0.01)
    assert jnp.isfinite(grad)


def test_djordjevic_sarkar_matches_skrf(wideband_freq):
    skrf = pytest.importorskip("skrf")
    from skrf.media.mline import MLine

    ep_r, tand = 4.3, 0.02
    f_low, f_high, f_ref = 1e3, 1e12, 1e9
    f = np.asarray(wideband_freq.f)

    expected, _ = MLine.analyse_dielectric(
        None, ep_r=ep_r, tand=tand, f_low=f_low, f_high=f_high,
        f_epr_tand=f_ref, f=f, diel='djordjevicsvensson',
    )
    got = DjordjevicSarkar(ep_r, tand, f_low, f_high, f_ref).properties(wideband_freq).ep_r

    assert jnp.allclose(got, jnp.asarray(expected), rtol=1e-10, atol=0.0)


def test_djordjevic_sarkar_matches_target_at_reference():
    freq = Frequency.from_f(jnp.array([1e9]))
    eps = DjordjevicSarkar(4.3, 0.02, f_ref=1e9).properties(freq).ep_r
    assert jnp.allclose(jnp.real(eps), 4.3, rtol=1e-6)
    assert jnp.allclose(-jnp.imag(eps) / jnp.real(eps), 0.02, rtol=1e-6)


def test_djordjevic_sarkar_is_dispersive(wideband_freq):
    eps = DjordjevicSarkar(4.3, 0.02).properties(wideband_freq).ep_r
    # Permittivity falls monotonically with frequency inside the relaxation band.
    assert jnp.all(jnp.diff(jnp.real(eps)) < 0)


def test_multipole_debye_limits():
    lo = Frequency.from_f(jnp.array([1e3]))
    hi = Frequency.from_f(jnp.array([1e15]))
    material = MultipoleDebye(ep_inf=2.0, poles=[(1.0, 1e9), (0.5, 1e10)])

    assert jnp.allclose(jnp.real(material.properties(lo).ep_r), 3.5, rtol=1e-5)
    assert jnp.allclose(jnp.real(material.properties(hi).ep_r), 2.0, rtol=1e-5)


def test_multipole_debye_coerces_pairs():
    material = MultipoleDebye(ep_inf=2.0, poles=[(1.0, 1e9)])
    assert isinstance(material.poles[0], DebyePole)
    assert jnp.allclose(material.poles[0].dep_r, 1.0)


def test_cole_cole_reduces_to_debye(freq):
    cole = ColeCole(ep_inf=2.0, dep_r=1.0, f_relax=1e9, alpha=0.0)
    debye = MultipoleDebye(ep_inf=2.0, poles=[(1.0, 1e9)])
    assert jnp.allclose(cole.properties(freq).ep_r, debye.properties(freq).ep_r)


def test_cole_cole_finite_at_dc():
    dc = Frequency.from_f(jnp.array([0.0, 1e9]))
    eps = ColeCole(2.0, 1.0, 1e9, 0.3).properties(dc).ep_r
    assert jnp.all(jnp.isfinite(eps))
    assert jnp.allclose(eps[0], 3.0)


def test_tabulated_dielectric_interpolates():
    material = TabulatedDielectric(
        f=jnp.array([1e9, 2e9, 3e9]),
        ep_r=jnp.array([4.0 - 0.1j, 3.8 - 0.2j, 3.6 - 0.3j]),
    )
    freq = Frequency.from_f(jnp.array([1e9, 1.5e9, 3e9]))
    eps = material.properties(freq).ep_r
    assert jnp.allclose(eps, jnp.array([4.0 - 0.1j, 3.9 - 0.15j, 3.6 - 0.3j]))


@pytest.mark.parametrize(
    ("f", "ep_r", "message"),
    [
        ([], [], "nonempty"),
        ([[1e9, 2e9]], [[4.0, 3.9]], "one-dimensional"),
        ([1e9, 1e9], [4.0, 3.9], "strictly increasing"),
        ([2e9, 1e9], [4.0, 3.9], "strictly increasing"),
        ([1e9], [4.0, 3.9], "same shape"),
    ],
)
def test_tabulated_dielectric_rejects_invalid_tables(f, ep_r, message):
    with pytest.raises(ValueError, match=message):
        TabulatedDielectric(f=f, ep_r=ep_r)


def test_tabulated_dielectric_validates_shapes():
    with pytest.raises(ValueError):
        TabulatedDielectric(f=jnp.array([1e9, 2e9]), ep_r=jnp.array([4.0]))


def test_as_dielectric_converters():
    assert jnp.allclose(as_dielectric(ConstantDielectric(2.0)).ep_r, 2.0)

    scalar = as_dielectric(4.3)
    assert isinstance(scalar, ConstantDielectric)
    assert jnp.allclose(scalar.ep_r, 4.3)

    pair = as_dielectric((4.3, 0.02))
    assert jnp.allclose(pair.tand, 0.02)

    triple = as_dielectric((4.3, 0.02, 0.01))
    assert jnp.allclose(triple.sigma, 0.01)


@pytest.mark.parametrize("material", [
    ConstantDielectric(4.3, 0.02, 0.01),
    DjordjevicSarkar(4.3, 0.02),
    MultipoleDebye(ep_inf=2.0, poles=[(1.0, 1e9)]),
    ColeCole(2.0, 1.0, 1e9, 0.3),
])
def test_gradients_are_finite(material, wideband_freq):
    def loss(m):
        return jnp.sum(jnp.abs(m.properties(wideband_freq).ep_r))

    grads = jax.grad(loss)(material)
    leaves = jax.tree_util.tree_leaves(prf.unwrap(grads))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_dielectrics_are_non_magnetic_by_default(freq):
    """Permeability lives on the medium, and defaults to free space."""
    for material in (
        ConstantDielectric(4.3, 0.02),
        DjordjevicSarkar(4.3, 0.02),
        MultipoleDebye(ep_inf=2.0, poles=[(1.0, 1e9)]),
        ColeCole(2.0, 1.0, 1e9, 0.3),
    ):
        mu_r = material.properties(freq).mu_r
        assert mu_r.shape == (freq.npoints,)
        assert jnp.allclose(mu_r, 1.0)


def test_constant_dielectric_carries_permeability(freq):
    assert jnp.allclose(ConstantDielectric(4.3, mu_r=4.0).properties(freq).mu_r, 4.0)
