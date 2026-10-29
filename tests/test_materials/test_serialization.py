import jax.numpy as jnp
import pytest

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.materials import (
    BulkConductor,
    ColeColeDielectric,
    ConstantDielectric,
    DjordjevicSarkarDielectric,
    LorentzPole,
    PoleDielectric,
    RoughConductor,
    TabulatedDielectric,
)


MATERIALS = [
    ConstantDielectric(4.3, 0.02, 0.01),
    DjordjevicSarkarDielectric(4.3, 0.02),
    PoleDielectric(
        ConstantDielectric(ep_r=2.0, sigma=0.01),
        poles=[(1.0, 1e9), LorentzPole(dep_r=0.5, f_res=50e9, zeta=0.1)],
        mu_poles=[LorentzPole(dep_r=0.2, f_res=100e9)],
    ),
    ColeColeDielectric(2.0, 1.0, 1e9, 0.3),
    TabulatedDielectric(f=jnp.array([1e9, 2e9]), ep_r=jnp.array([4.0 - 0.1j, 3.8 - 0.2j])),
    BulkConductor(5.8e7),
    RoughConductor(5.8e7, roughness=1e-6),
]


@pytest.mark.parametrize("material", MATERIALS)
def test_material_round_trips(material, tmp_path):
    path = tmp_path / "material.json"
    prf.save(path, material)
    loaded = prf.load(path)

    assert type(loaded) is type(material)

    freq = Frequency(start=1.0, stop=10.0, npoints=11, unit='GHz')
    if isinstance(
        material,
        (ConstantDielectric, DjordjevicSarkarDielectric, PoleDielectric, ColeColeDielectric, TabulatedDielectric),
    ):
        assert jnp.allclose(loaded.properties(freq).ep_r, material.properties(freq).ep_r)
    else:
        assert jnp.allclose(loaded.properties(freq).zs, material.properties(freq).zs)
