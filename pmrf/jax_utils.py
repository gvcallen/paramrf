from typing import Any, Callable
import parax as prx

from equinox import (
    Partial as Partial,
    field as field,
    combine as combine,
)
from parax import (
    unwrap as unwrap,
    unwrap_self as unwrap_self,
    is_constant as is_constant,
    is_param as is_param,
)

from dataclasses import (
    InitVar as InitVar,
    replace as replace,
)

def tie(
    tree: Any, 
    target: Callable[[Any], Any], 
    source: Callable[[Any], Any], 
    tie_fn: Callable[[Any], Any] = lambda x: x
):
    """(experimental) Allows tieing different nodes in a PyTree together.

    To evaluate the tie, use :func:`prf.unwrap`.

    Args:
        tree: The root PyTree or Equinox module to wrap.
        target: A callable (lens) that extracts the parameter to be replaced 
            (e.g., `lambda m: m.layer.weight`).
        source: A callable (lens) that extracts the parameter to draw values 
            from (e.g., `lambda m: m.layer.bias`).
        tie_fn: An optional transformation function applied to the source 
            parameter before injecting it into the target. Defaults to the 
            identity function.
    """
    return prx.Tie(tree, target=target, source=source, tie_fn=tie_fn)

def freeze(model: Any):
    """
    Freezes a model (or any JAX PyTree) and returns the frozen model.

    This can be used to freeze models to make them non-optimizable,
    but should also be used as a field converter (using `prf.field(converter=prf.freeze)`)
    when storing raw arrays within in a model.
    """
    return prx.as_opaque(model)

def unfreeze(model: Any):
    """
    Unfreezes a potentially frozen model and returns the unfrozen model.
    """
    model = prx.as_free(model)
    if isinstance(model, prx.Static):
        model = model.unwrap()
    return model

def is_model(x: Any):
    """
    Returns if `x` is an instance of :class:`pmrf.Model`.
    """
    from pmrf.models import Model
    return isinstance(x, Model)