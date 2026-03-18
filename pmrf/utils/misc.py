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

# @classmethod
# def from_alias(cls, alias: str) -> 'Evaluator':
#     fields = alias.split('.')
#     if len(fields) > 1:
#         # FIX: Join back into a dot-separated string for attrgetter
#         subattrs = ".".join(fields[:-1]) 
#         alias = fields[-1]
#     else:
#         subattrs = None

#     match = re.match(r'^([a-zA-Z]+)(\d)?(\d)?(.*)$', alias)
#     if not match:
#         raise ValueError(f"Invalid feature alias format: '{alias}'")

#     prop_prefix = match.group(1)
#     port1 = match.group(2)
#     port2 = match.group(3)
#     prop_suffix = match.group(4)
    
#     prop = prop_prefix + prop_suffix

#     # Map 1-indexed string alias (e.g., S11) to 0-indexed port array slices
#     if port1 is not None and port2 is not None:
#         ports = (int(port1) - 1, int(port2) - 1)
#     else:
#         ports = None
    
#     # FIX: Changed `property` to `prop`
#     return cls(prop=prop, ports=ports, subattrs=subattrs)