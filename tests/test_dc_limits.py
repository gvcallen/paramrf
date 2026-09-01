"""
DC and low-frequency numerical guards.

Every model must give a finite ``s`` and a finite gradient on a sweep whose
first point sits at 0 Hz. The singularities are removable -- ``sqrt(w)`` has a
finite value but an infinite derivative there, and ``sigma/(w eps_0)`` and
``Im(Z)/w`` are both 0/0 -- so each is guarded with the double-``where``
pattern. These tests are parametrised over every concrete class so that models
added later inherit the check.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from pmrf.frequency import Frequency
from pmrf.parameters import Param
from pmrf.materials import (
    ColeCole,
    ConstantDielectric,
    DjordjevicSarkar,
    HammerstadRoughness,
    MultipoleDebye,
    BulkConductor,
    DebyePole,
    RoughConductor,
    TabulatedDielectric,
)
from pmrf.materials.conductor import AbstractConductor
from pmrf.materials.dielectric import AbstractDielectric
from pmrf.models.components.lines.base import TransmissionLine
from pmrf.models import (
    CoaxialLine,
    DatasheetLine,
    FloatingLine,
    MicrostripLine,
    PhaseLine,
    PhysicalLine,
    RLGCLine,
    StriplineLine,
)
from pmrf.models.components.lines.formulations import (
    HammerstadJensenMicrostripFormulation,
    KirschningJansenMicrostripDispersion,
    WheelerMicrostripFormulation,
)


@pytest.fixture
def dc_freq():
    """A sweep whose first point is exactly DC."""
    return Frequency(start=0.0, stop=10e9, npoints=11, unit='Hz')


LINES = {
    "PhaseLine": PhaseLine(z0=50.0, theta=90.0, f0=5e9),
    "RLGCLine": RLGCLine(R=0.1, L=250e-9, G=1e-6, C=100e-12, length=0.1),
    "RLGCLine (lossless)": RLGCLine(R=0.0, L=250e-9, G=0.0, C=100e-12, length=0.1),
    "PhysicalLine": PhysicalLine(
        zn=50.0, ep_r=2.2, A=0.01, f_A=1e9, tand=0.001, length=1.0
    ),
    "DatasheetLine": DatasheetLine(zn=50.0, vf=0.69, k1=0.2, k2=0.01, length=1.0),
    "CoaxialLine": CoaxialLine(
        d_in=0.9e-3,
        d_out=2.95e-3,
        dielectric=ConstantDielectric(ep_r=1.5, tand=4e-4),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=0.5,
    ),
    "MicrostripLine": MicrostripLine(
        w=3e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=0.1,
    ),
    "MicrostripLine (no dispersion)": MicrostripLine(
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02), dispersion=None, length=0.1
    ),
    "MicrostripLine (explicit KirschningJansenMicrostripDispersion)": MicrostripLine(
        formulation=WheelerMicrostripFormulation(),
        dispersion=KirschningJansenMicrostripDispersion(),
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        length=0.1,
    ),
    "MicrostripLine (finite thickness)": MicrostripLine(
        formulation=HammerstadJensenMicrostripFormulation(),
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        t=35e-6,
        length=0.1,
    ),
    "StriplineLine": StriplineLine(
        w=2.655e-3,
        b=3.2e-3,
        t=35e-6,
        dielectric=ConstantDielectric(ep_r=2.2, tand=0.001),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=0.1,
    ),
    "StriplineLine (zero thickness)": StriplineLine(
        dielectric=ConstantDielectric(ep_r=2.2, tand=0.001), t=None, length=0.1
    ),
    "FloatingLine": FloatingLine(floating=PhaseLine(z0=50.0, theta=90.0, f0=5e9)),
}

MATERIALS = {
    "ConstantDielectric": ConstantDielectric(ep_r=4.3, tand=0.02, sigma=1e-3),
    "DjordjevicSarkar": DjordjevicSarkar(sigma=1e-3),
    "MultipoleDebye": MultipoleDebye(
        poles=(DebyePole(dep_r=0.5, f_relax=1e9),), sigma=1e-3
    ),
    "ColeCole": ColeCole(dep_r=0.5, f_relax=1e9, alpha=0.3, sigma=1e-3),
    "TabulatedDielectric": TabulatedDielectric(
        f=jnp.array([0.0, 5e9, 10e9]),
        ep_r=jnp.array([4.3 - 0.1j, 4.2 - 0.1j, 4.1 - 0.1j]),
        sigma=1e-3,
    ),
    "BulkConductor": BulkConductor(sigma=1 / 1.72e-8),
    "RoughConductor": RoughConductor(sigma=1 / 1.72e-8, roughness=HammerstadRoughness(1e-6)),
}


def _freed(model):
    """Release every parameter so that the gradient actually reaches them."""
    return jax.tree.map(
        lambda x: x.as_free() if isinstance(x, Param) else x,
        model,
        is_leaf=lambda x: isinstance(x, Param),
    )


def _assert_finite_value_and_grad(model, fn):
    """`fn(model)` and its gradient with respect to every leaf must be finite."""
    model = _freed(model)
    value = fn(model)
    assert jnp.all(jnp.isfinite(value)), "non-finite value at DC"

    def loss(m):
        out = fn(m)
        return jnp.sum(jnp.abs(out) ** 2)

    grads = eqx.filter_grad(loss)(model)
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert leaves, "model exposed no differentiable leaves"
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), "non-finite gradient at DC"


@pytest.mark.parametrize("line", LINES.values(), ids=LINES.keys())
def test_line_is_finite_at_dc(line, dc_freq):
    _assert_finite_value_and_grad(line, lambda m: m.s(dc_freq))


@pytest.mark.parametrize("material", MATERIALS.values(), ids=MATERIALS.keys())
def test_material_is_finite_at_dc(material, dc_freq):
    if isinstance(material, AbstractDielectric):
        _assert_finite_value_and_grad(material, lambda m: m.properties(dc_freq).ep_r)
    else:
        _assert_finite_value_and_grad(material, lambda m: m.properties(dc_freq).zs)
        _assert_finite_value_and_grad(material, lambda m: m.properties(dc_freq).sigma)


def test_immittance_l_and_c_carry_the_lowest_frequency(dc_freq):
    """The existing `_per_w` guard: DC inherits the lowest non-zero point."""
    line = RLGCLine(R=0.1, L=250e-9, G=1e-6, C=100e-12, length=0.1)
    immittance = line.immittance(dc_freq)

    assert jnp.allclose(immittance.L, 250e-9)
    assert jnp.allclose(immittance.C, 100e-12)


def _concrete_subclasses(base):
    """Every instantiable class under `base`, so a new model cannot slip the net.

    A concrete class that is itself subclassed still counts -- `BulkConductor`
    is directly usable as well as being the base of `RoughConductor`.
    """
    found = set()
    pending = [base]
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if (
            cls is not base
            and cls.__module__.startswith("pmrf.")
            and not getattr(cls, "__abstractmethods__", None)
        ):
            found.add(cls)
    return found


@pytest.mark.parametrize(
    "base, covered",
    [
        (TransmissionLine, {type(line) for line in LINES.values()}),
        (AbstractDielectric, {type(m) for m in MATERIALS.values()}),
        (AbstractConductor, {type(m) for m in MATERIALS.values()}),
    ],
    ids=["lines", "dielectrics", "conductors"],
)
def test_dc_coverage_is_exhaustive(base, covered):
    """A model added later must be added to `LINES` or `MATERIALS` as well.

    The DC checks above are parametrised over hand-written instances, because
    each model needs a physically sensible geometry. This test is what makes a
    future model inherit them: it fails until the new class is listed.
    """
    missing = _concrete_subclasses(base) - covered
    assert not missing, f"not covered by the DC checks: {sorted(c.__name__ for c in missing)}"
