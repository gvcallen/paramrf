from typing import Any
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

    kwargs['metadata'] = metadata
        
    return eqx.field(
        **kwargs
    )