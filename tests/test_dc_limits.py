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

import pmrf as prf
from pmrf.frequency import Frequency
from pmrf.parameters import Param
from pmrf.materials.conductor import AbstractConductor
from pmrf.materials.dielectric import AbstractDielectric
from pmrf.models import AbstractCurrentDistribution, RLGCLine
from pmrf.models.components.lines.base import TransmissionLine
from tests._catalogue import LINES, MATERIALS, concrete_subclasses


@pytest.fixture
def dc_freq():
    """A sweep whose first point is exactly DC."""
    return Frequency(start=0.0, stop=10e9, npoints=11, unit='Hz')


def _freed(model):
    """Release every parameter so that the gradient actually reaches them."""
    return jax.tree.map(
        lambda x: prf.update(x, fixed=False) if isinstance(x, Param) else x,
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
        _assert_finite_value_and_grad(material, lambda m: m.properties(dc_freq).mu_r)
    else:
        _assert_finite_value_and_grad(material, lambda m: m.properties(dc_freq).zs)
        _assert_finite_value_and_grad(material, lambda m: m.properties(dc_freq).sigma)


def test_immittance_l_and_c_carry_the_lowest_frequency(dc_freq):
    """The existing `_per_omega` guard: DC inherits the lowest non-zero point."""
    line = RLGCLine(R=0.1, L=250e-9, G=1e-6, C=100e-12, length=0.1)
    immittance = line.immittance(dc_freq)

    assert jnp.allclose(immittance.L, 250e-9)
    assert jnp.allclose(immittance.C, 100e-12)


@pytest.mark.parametrize(
    "base, covered",
    [
        (TransmissionLine, {type(line) for line in LINES.values()}),
        (AbstractDielectric, {type(m) for m in MATERIALS.values()}),
        (AbstractConductor, {type(m) for m in MATERIALS.values()}),
        (
            AbstractCurrentDistribution,
            {
                type(line.current_distribution)
                for line in LINES.values()
                if hasattr(line, "current_distribution")
            },
        ),
    ],
    ids=["lines", "dielectrics", "conductors", "current distributions"],
)
def test_dc_coverage_is_exhaustive(base, covered):
    """A model added later must be added to the shared catalogue as well.

    The DC and serialization checks are parametrised over hand-written
    instances, because each model needs a physically sensible geometry. This
    test is what makes a future model inherit them: it fails until the new
    class is listed.
    """
    missing = concrete_subclasses(base) - covered
    assert not missing, f"not in the shared catalogue: {sorted(c.__name__ for c in missing)}"
