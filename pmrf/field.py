import dataclasses
from typing import Any, Callable
import equinox as eqx

def field(
    *,
    save: bool = True,
    transparent: bool = False,
    **kwargs: Any,
) -> Any:
    """Custom field specifier for ParamRF."""

    # Handle ParamRF-specific metadata
    metadata = dict(kwargs.pop("metadata", {}))
    if not save:
        metadata["save"] = False
    if transparent:
        metadata["transparent"] = True
        
    return eqx.field(
        **kwargs
    )