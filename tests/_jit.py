"""Test helpers for checking that a change to a model does not recompile it."""
import jax
import numpy as np


def _leaf_key(leaf):
    """What `eqx.filter_jit` keys a leaf on: dtype, shape and weak_type for arrays,
    the value itself for anything else, which it treats as static."""
    if isinstance(leaf, (jax.Array, np.ndarray)):
        return ("array", str(leaf.dtype), leaf.shape, getattr(leaf, "weak_type", False))
    return ("static", leaf)


def assert_same_jit_key(a, b):
    """Asserts that `a` and `b` share an `eqx.filter_jit` cache key.

    They must have the same treedef, and every array leaf the same dtype, shape and
    `weak_type`.
    """
    leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
    assert treedef_a == treedef_b, f"treedefs differ:\n{treedef_a}\n{treedef_b}"
    for i, (x, y) in enumerate(zip(leaves_a, leaves_b)):
        key_x, key_y = _leaf_key(x), _leaf_key(y)
        if key_x[0] == "static" and key_y[0] == "static":
            continue  # static leaves are part of the treedef comparison above
        assert key_x == key_y, f"leaf {i} differs: {key_x} != {key_y}"
