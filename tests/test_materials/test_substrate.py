import doctest

import jax
import jax.numpy as jnp
import pytest

import pmrf.materials.substrate as substrate_module
from pmrf.frequency import Frequency
from pmrf.materials import BulkConductor, ConstantDielectric, Substrate
from pmrf.models import AbstractBuilder, MicrostripLine
from pmrf.parameters import Param


@pytest.fixture
def freq():
    return Frequency(start=1.0, stop=10.0, npoints=11, unit="GHz")


def test_substrate_coerces_loose_materials():
    """Scalars are coerced by the same converters the lines use."""
    substrate = Substrate(h=1.6e-3, dielectric=4.3, conductor=5.8e7)

    assert isinstance(substrate.dielectric, ConstantDielectric)
    assert isinstance(substrate.conductor, BulkConductor)
    assert substrate.t is None


def test_substrate_docstring_doctests():
    """The shared-substrate example in the docstring is executable."""
    results = doctest.testmod(substrate_module, verbose=False)
    assert results.failed == 0
    assert results.attempted > 0


def test_shared_substrate_exposes_one_permittivity():
    """A builder injecting one substrate into two traces dedupes its parameters."""
    class Board(AbstractBuilder):
        substrate: Substrate
        w1: Param
        w2: Param

        def build(self):
            return (
                MicrostripLine(w=self.w1, substrate=self.substrate, length=0.1)
                ** MicrostripLine(w=self.w2, substrate=self.substrate, length=0.2)
            )

    board = Board(substrate=Substrate(h=1.6e-3, dielectric=4.3), w1=1e-3, w2=2e-3)
    names = board.named_params()

    permittivities = [name for name in names if name.endswith("ep_r")]
    assert len(permittivities) == 1


def test_same_param_object_in_two_lines_does_not_dedupe():
    """The contrast: PyTree flattening gives one leaf per line, not one shared leaf."""
    ep_r = Param(value=4.3)
    lines = (
        MicrostripLine(w=1e-3, h=1.6e-3, dielectric=ConstantDielectric(ep_r=ep_r), length=0.1)
        ** MicrostripLine(w=2e-3, h=1.6e-3, dielectric=ConstantDielectric(ep_r=ep_r), length=0.2)
    )

    names = lines.named_params()
    permittivities = [name for name in names if name.endswith("ep_r")]
    assert len(permittivities) == 2


def test_microstrip_substrate_and_loose_forms_are_identical(freq):
    """Both idioms build the same canonical substrate, so both PyTrees match."""
    loose = MicrostripLine(
        w=3e-3,
        h=1.6e-3,
        dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
        conductor=BulkConductor(sigma=1 / 1.72e-8),
        length=0.1,
    )
    grouped = MicrostripLine(
        w=3e-3,
        substrate=Substrate(
            h=1.6e-3,
            dielectric=ConstantDielectric(ep_r=4.3, tand=0.02),
            conductor=BulkConductor(sigma=1 / 1.72e-8),
        ),
        length=0.1,
    )

    assert jax.tree_util.tree_structure(loose) == jax.tree_util.tree_structure(grouped)
    assert jnp.array_equal(
        jnp.asarray(jax.tree_util.tree_leaves(loose)),
        jnp.asarray(jax.tree_util.tree_leaves(grouped)),
    )
    assert jnp.allclose(loose.s(freq), grouped.s(freq))


def test_microstrip_rejects_substrate_with_loose_fields():
    """The two idioms are alternatives, not a mixture."""
    with pytest.raises(ValueError, match="substrate"):
        MicrostripLine(
            w=3e-3,
            h=1.6e-3,
            substrate=Substrate(h=1.6e-3, dielectric=4.3),
            length=0.1,
        )
