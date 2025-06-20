from datetime import datetime
from typing import Union, Sequence
from numbers import Number

import pmrf.numpy as np
from pmrf.numpy import USE_JAX
if USE_JAX:
    from equinox import field
else:
    from dataclasses import field

import jax
from jaxtyping import Array, ArrayLike, Bool, Float, PyTree, PyTreeDef

def time_string(format="%H:%M:%S"):
    return datetime.now().strftime(format)

NumberLike = Union[Number, Sequence[Number], np.ndarray]

def tree_flatten_one_level_with_path(
    pytree: PyTree,
) -> tuple[list[PyTree], PyTreeDef]:
    # See eqx.tree_flatten_one_level
    seen_pytree = False

    def is_leaf(node):
        nonlocal seen_pytree
        if node is pytree:
            if seen_pytree:
                try:
                    type_string = type(pytree).__name__
                except AttributeError:
                    type_string = "<unknown>"
                raise ValueError(
                    f"PyTree node of type `{type_string}` is immediately "
                    "self-referential; that is to say it appears within its own PyTree "
                    "structure as an immediate subnode. (For example "
                    "`x = []; x.append(x)`.) This is not allowed."
                )
            else:
                seen_pytree = True
            return False
        else:
            return True

    return jax.tree.flatten_with_path(pytree, is_leaf=is_leaf)