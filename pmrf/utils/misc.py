import pkgutil
import importlib
from datetime import datetime
from typing import Union, get_origin
import inspect
from types import GenericAlias, UnionType

class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(cls)

def is_convertible_to_float(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False         
           
def load_class_from_string(dotted_path):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
    
def iter_submodules(package_name: str):
    """Yield all submodules and subpackages of a given package."""
    package = importlib.import_module(package_name)
    if not hasattr(package, '__path__'):
        raise ValueError(f"{package_name} is not a package")

    for _finder, name, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        yield name, ispkg

def time_string(format="%H:%M:%S"):
    return datetime.now().strftime(format)

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