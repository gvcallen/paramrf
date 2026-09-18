"""Models and values derived from a base with new parameters."""

import copy
import functools
from typing import Any, Callable

import equinox as eqx
import parax as prx

from pmrf.models.base import Model
from pmrf.models.adapters.delegated import AbstractBuilder
from pmrf.utils import field


class _NoBase(eqx.Module):
    """Sentinel for a derived node built from new parameters alone."""


NO_BASE = _NoBase()
"""The base of a derived node that has none. A pytree rebuild makes a new one, so it
is recognised by its type rather than by identity."""


def _call(fn, base, new):
    """Calls `fn` on a base (unless there is none) and the new parameters."""
    if isinstance(base, _NoBase):
        return fn(**new)
    return fn(base, **new)


class DerivedValue(prx.AbstractUnwrappable):
    """A value computed by a function from a base and new named parameters.

    Built by :func:`derived` when the base is not a :class:`pmrf.Model`. The node
    stands in for the value wherever it is stored, most often a parameter field of a
    model, and unwraps to ``fn(base, **new)`` on the unwrapped base and parameters.

    It is not a parameter: it has no declared value, scale, prior or fixed state, and
    the field it is stored in applies neither its constraint nor its scale to it.
    Parameter names see through it: the base keeps the name its position gives it and
    each new parameter is named by its keyword at the same level.
    """

    #: The base the value is derived from, or ``NO_BASE`` if it has none.
    base: Any

    #: The new parameters, keyed by keyword.
    new: dict[str, Any]

    #: The function building the value from the base and the new parameters.
    fn: Callable[..., Any] = field(static=True, kw_only=True)

    def unwrap(self) -> Any:
        return _call(self.fn, self.base, self.new)


class Derived(AbstractBuilder):
    """A model computed by a function from a base model and new named parameters.

    Built by :func:`derived` when the base is a :class:`pmrf.Model`, which documents
    the contract. The base and the new parameters are held once and ``fn(base, **new)``
    is called on their unwrapped values whenever the model is used. Parameter names
    see through this wrapper: the base's names are unchanged and each new parameter
    is named by its keyword.
    """

    #: The base model, or ``NO_BASE`` if it has none.
    base: Any

    #: The new parameters, keyed by keyword.
    new: dict[str, Any]

    #: The function building the model from the base and the new parameters.
    fn: Callable[..., Model] = field(static=True, kw_only=True)

    def build(self) -> Model:
        model = _call(self.fn, self.base, self.new)
        if not isinstance(model, Model):
            raise TypeError(
                f"The derived function {getattr(self.fn, '__qualname__', self.fn)!r} must "
                f"return a pmrf.Model; got {type(model).__name__}."
            )
        return model


def _is_model_base(base: Any) -> bool:
    """Returns whether a base makes the derived node a model.

    A model, or a wrapper that unwraps to one, such as a joint prior over a model
    (:class:`pmrf.modules.Probabilistic`). A collection of models is not one.
    """
    from pmrf.parameters import is_param

    if isinstance(base, Model):
        return True
    if is_param(base) or not isinstance(base, prx.AbstractUnwrappable):
        return False
    return isinstance(prx.unwrap(base), Model)


def is_derived(x: Any) -> bool:
    """Returns whether `x` is a derived node, a model or a value."""
    return isinstance(x, (Derived, DerivedValue))


def _unnamed(node: Any) -> Any:
    """Returns `node` without its own name, so its parameters keep their names below it."""
    if getattr(node, 'name', None) is None:
        return node
    node = copy.copy(node)
    object.__setattr__(node, 'name', None)
    return node


def derived(fn: Callable[..., Any]) -> Callable[..., Derived | DerivedValue]:
    """
    Turns a function of a base and new parameters into a derived constructor.

    A derived node starts from something nominal and derives a more complete version
    of it, with new parameters. The decorated function is called as ``f(base, **new)``:
    at most one positional base and keyword arguments that are the new parameters (a
    :class:`pmrf.Param`, or anything :func:`pmrf.as_param` accepts). The result holds
    the base and the new parameters once and calls `fn` on them whenever it is used.

    **What the result is.** When the base is a :class:`pmrf.Model`, the result is a
    model too (a :class:`Derived`), whose RF interface is `fn`'s output: a nominal
    cable derives a wet one. Any other base — a :class:`pmrf.Param`, an array, a
    pytree — or no base at all gives a plain derived value (a :class:`DerivedValue`)
    that can be stored in any parameter field: a nominal permittivity derives a
    drifted one. A bare collection of models is not a model, so derive at the model
    that contains them instead.

    **No base.** Calling with keywords only derives a value from new parameters
    alone, as when the prior is on a velocity factor but the line takes permittivity. At
    least one new parameter is always required.

    **A derived value is not a parameter.** It has no declared value, scale, prior or
    fixed state, and it does not appear in :func:`pmrf.params`. Any constraint on it
    follows from its inputs' priors: the field's own constraint and scale do not apply,
    and nothing is bounds-checked or clamped. Read one back by unwrapping the model
    it sits in (``prf.unwrap(line).dielectric.ep_r``).

    `fn` receives the base and the new parameters as the node is unwrapped, so
    parameters arrive as physical values, as for :func:`pmrf.tie`. It must be pure,
    and (for a model) its output's structure and port count must not depend on
    parameter values. Inside `fn`, use :func:`pmrf.replace` for fields of the object
    in hand and :func:`pmrf.update` for parts reached by name.

    **Names.** The base's parameters keep the names they have on the base, and each
    new parameter is named by its keyword: at the top level for a derived model, and
    beside the field for a derived value (``dielectric.tc`` for a value stored in
    ``dielectric.ep_r``). A keyword that clashes with a name in the base raises here;
    one that clashes with a sibling parameter or with another derived field's keyword
    is a name collision, raised when names are resolved. Nothing produced inside `fn`
    is named. A derived model takes
    the base's name unless ``name=`` is passed, so a container prefixes both as usual.
    Derived nodes can be nested and names accumulate flat; a parameter shared by
    several parts is expressed by deriving at the level that owns it.

    `fn` is a static part of the tree: two nodes derived with the same function share
    a jit cache entry. Define it once at module level; a lambda created on every call
    recompiles.

    Parameters
    ----------
    fn : Callable
        ``fn(base, **new) -> Model``, or ``fn(base, **new) -> Any`` for a value.

    Returns
    -------
    Callable
        ``constructor(base=..., *, name=None, **new) -> Derived | DerivedValue``.

    Raises
    ------
    TypeError
        On a call with more than one positional base or with no new parameters, on
        ``name=`` for a node that is not a model, or (when the model is used) if `fn`
        does not return a model for a model base.
    ValueError
        If a keyword clashes with a parameter name of the base. A keyword that clashes
        with a sibling parameter or with another derived field's keyword is a name
        collision, raised by :func:`pmrf.params` and the functions that resolve names.

    Examples
    --------
    A coaxial cable of total length ``L`` that is wet for its first ``wet_length``:

    .. code-block:: python

        @prf.derived
        def wet(cable, wet_length, wet_ep_r):
            wet = prf.replace(cable, length=wet_length,
                              dielectric=prf.replace(cable.dielectric, ep_r=wet_ep_r))
            dry = prf.replace(cable, length=cable.length - wet_length)
            return wet ** dry

        coax = wet(coax, wet_length=prf.Random(Uniform(0, 20), scale=1e-3),
                   wet_ep_r=prf.Random(Uniform(1, 80)))
        prf.params(coax)   # the cable's names, plus 'wet_length' and 'wet_ep_r'

    Permittivity that drifts with temperature, keeping the nominal one and its prior:

    .. code-block:: python

        @prf.derived
        def drift(ep_r, tc):
            return ep_r * (1 + tc * DELTA_T)

        line = prf.update(line, 'dielectric.ep_r', drift(line.dielectric.ep_r, tc=tc))
        prf.params(line)   # ..., 'dielectric.ep_r' (the base), 'dielectric.tc'

    Permittivity parametrised by the velocity factor instead, with no base:

    .. code-block:: python

        @prf.derived
        def from_vf(vf):
            return 1 / vf**2

        line = prf.update(line, 'dielectric.ep_r', from_vf(vf=prf.Random(...)))
        prf.params(line)   # ..., 'dielectric.vf'; 'dielectric.ep_r' is gone
    """
    from pmrf.parameters import as_param, params

    @functools.wraps(fn)
    def constructor(*args, name: str | None = None, **new) -> Derived | DerivedValue:
        if len(args) > 1:
            raise TypeError(
                f"{fn.__name__}() takes at most one positional base and new parameters as "
                f"keywords, e.g. {fn.__name__}(model, wet_length=...); got {len(args)} "
                "positional arguments."
            )
        if not new:
            raise TypeError(f"{fn.__name__}() needs at least one new parameter as a keyword.")
        base = args[0] if args else NO_BASE
        clashes = sorted(set(new) & set(params(base)))
        if clashes:
            raise ValueError(
                f"{fn.__name__}(): new parameter names {clashes} clash with parameters of "
                "the base. Choose different keywords."
            )
        # A derived value passed as a new parameter is kept as it is; anything else
        # becomes a parameter, named by its keyword rather than by itself.
        coerced = {key: _unnamed(as_param(value)) for key, value in new.items()}
        if not _is_model_base(base):
            if name is not None:
                raise TypeError(
                    f"{fn.__name__}(): 'name=' applies to a derived model, but the base is "
                    f"{'absent' if isinstance(base, _NoBase) else type(base).__name__}. A derived value "
                    "is named by where it is stored."
                )
            return DerivedValue(base, coerced, fn=fn)
        if name is None:
            name = getattr(base, 'name', None)
        return Derived(_unnamed(base), coerced, fn=fn, name=name)

    return constructor


__all__ = ["Derived", "DerivedValue", "derived", "is_derived"]
