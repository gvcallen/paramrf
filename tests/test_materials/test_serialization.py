import jax.numpy as jnp
import pytest

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.materials import (
    Bulk,
    ColeCole,
    ConstantDielectric,
    DjordjevicSarkar,
    MultipoleDebye,
    RoughConductor,
    TabulatedDielectric,
)


MATERIALS = [
    ConstantDielectric(4.3, 0.02, 0.01),
    DjordjevicSarkar(4.3, 0.02),
    MultipoleDebye(eps_inf=2.0, poles=[(1.0, 1e9), (0.5, 1e10)]),
    ColeCole(2.0, 1.0, 1e9, 0.3),
    TabulatedDielectric(f=jnp.array([1e9, 2e9]), eps_r_values=jnp.array([4.0 - 0.1j, 3.8 - 0.2j])),
    Bulk(1.68e-8),
    RoughConductor(1.68e-8, 1e-6),
]


@pytest.mark.parametrize("material", MATERIALS)
def test_material_round_trips(material, tmp_path):
    path = tmp_path / "material.json"
    prf.save(path, material)
    loaded = prf.load(path)

    assert type(loaded) is type(material)

    freq = Frequency(start=1.0, stop=10.0, npoints=11, unit='GHz')
    if hasattr(material, "epsilon_r"):
        assert jnp.allclose(loaded.epsilon_r(freq), material.epsilon_r(freq))
    else:
        assert jnp.allclose(loaded.surface_impedance(freq), material.surface_impedance(freq))
