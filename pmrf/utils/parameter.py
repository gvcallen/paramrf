from typing import Any

from pmrf.core import Parameter

def is_param(x) -> bool:
    r"""
    Check if an object is an instance of a `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter)

def is_valid_param(x) -> bool:
    r"""
    Check if an object is an instance of a `Parameter` and if its value is not None.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a valid Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and x.value is not None

def is_free_param(x) -> bool:
    r"""
    Check if an object is a non-fixed `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a non-fixed Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and not x.fixed

def is_fixed_param(x) -> bool:
    r"""
    Check if an object is a fixed `Parameter`.

    Parameters
    ----------
    x
        The object to check.

    Returns
    -------
    bool
        `True` if the object is a fixed Parameter, `False` otherwise.
    """
    return isinstance(x, Parameter) and x.fixed

def as_param(x: Any | list[Any] | dict[str, Any], **kwargs) -> Parameter:
    r"""
    Ensure an object is a `Parameter` or container over parameters.

    If the object is already a `Parameter`, it is returned unchanged.
    Otherwise, the underlying objects are converted into new `Parameter` objects.

    Parameters
    ----------
    x
        The object to convert.
    **kwargs
        Additional keyword arguments passed to the `Parameter` constructor (e.g. `name`).

    Returns
    -------
    Parameter
        The object wrapped as a `Parameter`.
    """
    from pmrf.parameters import Free, Fixed
    
    if isinstance(x, Parameter):
        return x
    elif isinstance(x, list):
        return [as_param(xi, **kwargs) for xi in x]
    elif isinstance(x, dict):
        return {k: as_param(xi, **kwargs) for k, xi in x.items()}
    else:
        is_fixed = kwargs.pop('fixed', False)
        if is_fixed:
            return Fixed(value=x, **kwargs)
        else:
            return Free(value=x, **kwargs)