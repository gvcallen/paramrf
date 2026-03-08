from typing import Any, Callable
import equinox as eqx

def field(
    *,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
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
        converter=converter,
        static=static,
        **kwargs
    )