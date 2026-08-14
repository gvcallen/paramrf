"""Base class for parameter-aware ParamRF modules."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Self, TypeGuard, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import parax as prx

from pmrf.parameters import Param, tree_named_params, tree_param_names_to_path
from pmrf.utils import field, unwrap
from pmrf.utils.optix import Lens, focus
from pmrf.utils.tree import resolve_target

T = TypeVar("T")


class Module(eqx.Module):
    """Base class for parameter-aware objects in ParamRF.

    A module is an immutable JAX PyTree that may contain ParamRF parameters,
    RF models, and other modules. Unlike :class:`pmrf.Model`, it does not imply
    an RF response or a number of ports.
    """

    name: str | None = field(default=None, kw_only=True, static=True)
    """A name for the module."""

    metadata: Any = field(default=None, kw_only=True, static=True)
    """Arbitrary metadata stored alongside the module."""

    def named_params(
        self,
        full_params: bool = False,
        free_only: bool = False,
        namespace_separator: str = "_",
    ) -> dict[str, float | jnp.ndarray | Param]:
        """Return a named dictionary of parameters in the module.

        Parameters and modules can be given names upon construction. If no custom
        names are present, standard Python attribute paths are used. Named modules
        collapse the path to their left into a namespace, whilst a named parameter
        collapses its path to the nearest named module or the root. This supports
        flat parameter names, flat module names, or fully nested namespaces.

        Parameters
        ----------
        full_params : bool, default=False
            Return full parameter objects instead of their resulting values.
        free_only : bool, default=False
            Return only free parameters.
        namespace_separator : str, default="_"
            Separator used to join named module namespaces.

        Returns
        -------
        dict[str, Any]
            Parameter names mapped to values or parameter objects.
        """
        module = self if free_only else prx.unwrap(
            self, only_if=lambda x: isinstance(x, prx.Tie), cascade=False
        )
        return tree_named_params(
            module,
            full_params=full_params,
            free_only=free_only,
            namespace_separator=namespace_separator,
        )

    def __repr__(self) -> str:
        try:
            tree_to_format = unwrap(self)
        except Exception:
            return eqx.tree_pformat(self, short_arrays=False)

        class _RawFormatter:
            def __init__(self, val):
                self.val = np.asarray(val)

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

    def at(
        self: Self,
        target: Union[Callable[[Self], T], str, tuple[str, ...], list[str]],
    ) -> Lens[Self, T]:
        """A functional interface for module manipulation.

        This wraps :func:`equinox.tree_at` using an optic. Pass a callable, a
        string parameter name, or a tuple of names selecting the values to inspect
        or replace, then call methods such as ``.get()``, ``.set()``, or ``.apply()``.

        Updates are surgical: replacement values bypass dataclass converters and
        validation. Callers must therefore preserve field invariants and pass fully
        constructed parameters where required.

        Examples
        --------
        >>> import pmrf as prf
        >>> from pmrf.models import Resistor
        >>> module = Resistor(R=50.0, name="res")
        >>> module.at("res.R").get()
        50.0
        >>> updated = module.at("res.R").set(prf.Unconstrained(100.0))

        Returns
        -------
        Lens
            An optic focused on the root of this module.
        """
        try:
            name_to_path = tree_param_names_to_path(self)
            resolved_where = resolve_target(target, name_to_path)
        except Exception as e:
            raise ValueError(f"Could not resolve parameter name: {e}")
        return focus(self).at(resolved_where)

    def map(
        self: Self, fn: Callable[[Any], Any], is_target: Callable | None = None
    ) -> Self:
        """A functional interface for mapping over a module.

        This wraps :func:`jax.tree.map`. To map parameters, pass
        ``is_target=pmrf.is_param``.

        Examples
        --------
        >>> import pmrf as prf
        >>> from pmrf.models import Resistor, Capacitor
        >>> module = Resistor(R=50.0) ** Capacitor(C=1e-12)
        >>> scaled = module.map(lambda p: p * 2.0, is_target=prf.is_param)

        Returns
        -------
        Module
            The mapped module.
        """
        if is_target is None:
            raise TypeError("`is_target` must be provided when mapping a module")

        def _wrapped_fn(node):
            return fn(node) if is_target(node) else node

        return jax.tree.map(_wrapped_fn, self, is_leaf=is_target)

    def tied(
        self,
        target: Union[Callable[[Any], Any], str, tuple[str, ...], list[str]],
        source: Union[Callable[[Any], Any], str, tuple[str, ...], list[str]],
        tie_fn: Callable[[Any], Any] = lambda x: x,
    ) -> Module:
        """Tie parameters or sub-modules within this module together.

        The target is hidden from optimizers and reconstructed from the source when
        the module is unwrapped. Targets and sources may be structural callables or
        resolved parameter names.

        Parameters
        ----------
        target
            Callable or parameter name selecting the value to replace.
        source
            Callable or parameter name selecting the value it is derived from.
        tie_fn
            Transformation applied to the source value. Defaults to identity.

        Returns
        -------
        Module
            A wrapped module with the relationship applied during unwrapping.
        """
        from pmrf.modules import Tied

        module = self.module if isinstance(self, Tied) else self
        name_to_path = tree_param_names_to_path(module)
        resolved_target = resolve_target(target, name_to_path)
        resolved_source = resolve_target(source, name_to_path)
        return Tied(
            self,
            target=resolved_target,
            source=resolved_source,
            tie_fn=tie_fn,
        )


def is_module(x: Any) -> TypeGuard[Module]:
    """Return whether ``x`` is a :class:`Module`."""
    return isinstance(x, Module)


def validate(tree):
    """Ensure modules contain no ambiguous raw, trainable-looking arrays."""
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
