from typing import Union, get_origin
import inspect
from types import GenericAlias, UnionType

def is_overridden(cls, baseclass, method_name):
    result = False
    for cls in inspect.getmro(cls):
        if method_name in cls.__dict__:
            result = cls is not baseclass
            break
    return result

def get_first_underlying_type(tp: type) -> type | None:
    # The annotations could be unions - in this case we just take the first one TODO upgrade this to do more in-depth inspection?
    if isinstance(tp, UnionType):
        return get_first_underlying_type(tp.__args__[0])
    if isinstance(tp, (type,)) and not isinstance(tp, (GenericAlias, UnionType)):
        return tp

    origin = get_origin(tp)
    if origin is None:
        return None
    if origin is Union:
        return None
    return get_first_underlying_type(origin)