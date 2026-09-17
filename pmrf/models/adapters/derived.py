"""Models derived from an existing model with new parameters."""

import copy
import functools
from typing import Any, Callable

from pmrf.models.base import Model
from pmrf.models.adapters.delegated import AbstractBuilder
from pmrf.utils import field


class Derived(AbstractBuilder):
    """A model computed by a function from a base and new named parameters.

    Built by :func:`derived`, which documents the contract. The base and the new
    parameters are held once, as ``operands = (base, new)``, and ``fn(base, **new)``
    is called on their unwrapped values whenever the model is used. Parameter names
    see through this wrapper: the base's names are unchanged and each new parameter
    is named by its keyword.
    """

    #: The base, and the new parameters keyed by keyword.
    operands: tuple[Any, dict[str, Any]]

    #: The function building the model from the base and the new parameters.
    fn: Callable[..., Model] = field(static=True, kw_only=True)

    def build(self) -> Model:
        base, new = self.operands
        model = self.fn(base, **new)
        if not isinstance(model, Model):
            raise TypeError(
                f"The derived function {getattr(self.fn, '__qualname__', self.fn)!r} must "
                f"return a pmrf.Model; got {type(model).__name__}."
            )
        return model


def _unnamed(node: Any) -> Any:
    """Returns `node` without its own name, so its parameters keep their names below it."""
    if getattr(node, 'name', None) is None:
        return node
    node = copy.copy(node)
    object.__setattr__(node, 'name', None)
    return node


def derived(fn: Callable[..., Model]) -> Callable[..., Derived]:
    """
    Turns a function of a base model and new parameters into a derived model constructor.

    A derived model starts from a nominal model and derives a more complete one that
    captures some artifact (a wet section, degraded copper, a cut). The decorated
    function is called as ``f(base, **new)``: exactly one positional base (a model,
    or any collection of models and parameters) and keyword arguments that are the
    new parameters (a :class:`pmrf.Param`, or anything :func:`pmrf.as_param`
    accepts). It returns a :class:`pmrf.Model` that holds the base and the new
    parameters once, and calls `fn` on them whenever the model is used.

    `fn` receives the base and the new parameters as the model is unwrapped, so
    parameters arrive as physical values, as for :func:`pmrf.tie`. It must return a
    :class:`pmrf.Model`, must be pure, and its output's structure and port count must
    not depend on parameter values. Inside `fn`, use :func:`pmrf.replace` for fields
    of the object in hand and :func:`pmrf.update` for parts reached by name.

    **Names.** The base's parameters keep the names they have on the base, and each
    new parameter is named by its keyword. Nothing produced inside `fn` is named. The
    derived model takes the base's name unless ``name=`` is passed, so a container
    prefixes both as usual. A derived model can be the base of another, and names
    accumulate flat; a parameter shared by several parts is expressed by deriving at
    the level that owns it.

    `fn` is a static part of the model: two models derived with the same function
    share a jit cache entry. Define it once at module level; a lambda created on every
    call recompiles.

    Parameters
    ----------
    fn : Callable
        ``fn(base, **new) -> Model``.

    Returns
    -------
    Callable
        ``constructor(base, *, name=None, **new) -> Derived``.

    Raises
    ------
    TypeError
        On a call with other than one positional base or with no new parameters, or
        (when the model is used) if `fn` does not return a model.
    ValueError
        If a keyword clashes with a parameter name of the base.

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
    """
    from pmrf.parameters import as_param, params

    @functools.wraps(fn)
    def constructor(*args, name: str | None = None, **new) -> Derived:
        if len(args) != 1:
            raise TypeError(
                f"{fn.__name__}() takes exactly one positional base and new parameters as "
                f"keywords, e.g. {fn.__name__}(model, wet_length=...); got {len(args)} "
                "positional arguments."
            )
        if not new:
            raise TypeError(f"{fn.__name__}() needs at least one new parameter as a keyword.")
        base = args[0]
        clashes = sorted(set(new) & set(params(base)))
        if clashes:
            raise ValueError(
                f"{fn.__name__}(): new parameter names {clashes} clash with parameters of "
                "the base. Choose different keywords."
            )
        coerced = {}
        for key, value in new.items():
            value = as_param(value)
            if value.name is not None:
                value = _unnamed(value)
            coerced[key] = value
        if name is None:
            name = getattr(base, 'name', None)
        return Derived((_unnamed(base), coerced), fn=fn, name=name)

    return constructor


__all__ = ["Derived", "derived"]
