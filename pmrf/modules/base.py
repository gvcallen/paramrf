"""Base class for parameter-aware ParamRF modules."""

from __future__ import annotations

import dataclasses
from typing import Any, TypeGuard

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import parax as prx
from jaxtyping import PyTree

from pmrf.parameters import is_param
from pmrf.utils import field, unwrap

class _PhysicalArray(np.ndarray):
    """A parameter's physical value that remembers its declared value, for ``repr``.

    Arithmetic returns a plain array, so a value a tie derives is shown as computed.
    """

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        return np.asarray(obj).view(np.ndarray)


class _ReprParam(prx.AbstractUnwrappable):
    """Stands in for a parameter while formatting: unwraps to its physical value, as
    ties expect, tagged with the declared value to show."""

    physical: np.ndarray
    declared: np.ndarray

    def unwrap(self):
        value = self.physical.view(_PhysicalArray)
        value.declared = self.declared
        return value


class Module(eqx.Module):
    """Base class for parameter-aware objects in ParamRF.

    A module is an immutable JAX PyTree that may contain ParamRF parameters,
    RF models, and other modules. Subclassing it says the object takes part in
    three things:

    - **Naming.** A module's :attr:`name` collapses the path to its left into a
      namespace, which parameter names (:func:`pmrf.params`) depend on.
    - **Field validation.** :func:`pmrf.modules.validate` rejects raw float JAX
      arrays in its fields, which are ambiguous between free and fixed parameters.
    - **A shared base class** for :class:`pmrf.Model`, losses, likelihoods,
      kernels, materials and evaluators, with :attr:`metadata` and a readable
      ``repr`` that shows parameters in declared space.

    It has no public methods: operations that make equal sense on any collection of
    models and parameters are free functions, such as :func:`pmrf.update` and
    :func:`pmrf.tie` (ADR-0002).

    Passing the *same* instance to two sibling fields does not share it. A module
    is a JAX PyTree, and each path holds its own copy of the leaves, so
    ``prf.params(Two(a=Resistor(R=p), b=Resistor(R=p)))`` gives two
    independent parameters, ``a.R`` and ``b.R``. Object identity is not tracked,
    because JAX transformations rebuild objects. To share a parameter, inject it
    once into a builder (see :class:`pmrf.models.AbstractBuilder` and
    :class:`pmrf.materials.Substrate`), or tie the copies together with
    :func:`pmrf.tie`.
    """

    name: str | None = field(default=None, kw_only=True, static=True)
    """A name for the module."""

    metadata: Any = field(default=None, kw_only=True, static=True)
    """Arbitrary metadata stored alongside the module."""

    def __repr__(self) -> str:
        try:
            tree_to_format = unwrap(jax.tree.map(
                lambda p: _ReprParam(np.asarray(p.physical_value), np.asarray(p.value)) if is_param(p) else p,
                self,
                is_leaf=is_param,
            ))
        except Exception:
            return eqx.tree_pformat(self, short_arrays=False)

        class _RawFormatter:
            def __init__(self, val):
                self.val = np.asarray(getattr(val, "declared", val))

            def __repr__(self):
                return np.array2string(self.val, separator=", ", precision=4)

        is_array = lambda x: isinstance(x, (jax.Array, np.ndarray))
        tree_clean = jax.tree.map(
            lambda x: _RawFormatter(x) if is_array(x) else x,
            tree_to_format,
            is_leaf=is_array,
        )
        return eqx.tree_pformat(tree_clean, short_arrays=False)

    def __str__(self) -> str:
        return repr(self)


def is_module(x: Any) -> TypeGuard[Module]:
    """Return whether ``x`` is a :class:`Module`."""
    return isinstance(x, Module)


def validate(tree: PyTree):
    """Validate a parameter PyTree.

    Any PyTree structure is accepted. ParamRF modules nested anywhere within it
    additionally retain their stricter field validation, which prevents raw JAX
    arrays from ambiguously representing either fixed or free parameters.
    """
    def _is_leaf(x):
        return isinstance(x, Module) or prx.constraints.is_leaf(x)

    nodes, _ = jax.tree.flatten(tree, is_leaf=_is_leaf)
    for node in nodes:
        if isinstance(node, Module):
            for f in dataclasses.fields(node):
                val = getattr(node, f.name)
                is_array = isinstance(val, jnp.ndarray)
                is_static = f.metadata.get("static", False)
                if is_array and not is_static and jnp.issubdtype(val.dtype, jnp.inexact):
                    raise TypeError(
                        f"Field '{f.name}' in '{node.__class__.__name__}' is a raw JAX array, "
                        "meaning it is unclear whether this is a free or fixed parameter.\n\n"
                        "Use a `pmrf.parameters` factory for free variables, a NumPy array "
                        "for fixed values, or an explicit ParamRF field converter."
                    )
                validate(val)
