import equinox as eqx
import jax.numpy as jnp
import numpy as np

import pmrf as prf
from pmrf.utils import ByIdentity, field


class _ElementwiseEq:
    """An object whose ``==`` is elementwise, like ``skrf.Network``."""

    def __init__(self, n):
        self.data = np.zeros(n)

    def __eq__(self, other):
        return self.data == other.data


def test_by_identity_compares_by_identity():
    value = _ElementwiseEq(3)
    assert ByIdentity(value) == ByIdentity(value)
    assert hash(ByIdentity(value)) == hash(ByIdentity(value))
    assert ByIdentity(value) != ByIdentity(_ElementwiseEq(3))
    assert ByIdentity(value) != value


def test_by_identity_static_fields_of_different_shapes_share_a_jit():
    class Holder(eqx.Module):
        held: ByIdentity = field(static=True)
        x: jnp.ndarray

    traces = []

    @eqx.filter_jit
    def fn(holder):
        traces.append(None)
        return holder.x * 2

    first = Holder(ByIdentity(_ElementwiseEq(2)), jnp.array(1.0))
    fn(first)
    fn(Holder(ByIdentity(_ElementwiseEq(5)), jnp.array(1.0)))
    fn(prf.utils.replace(first, x=jnp.array(3.0)))
    assert len(traces) == 2
