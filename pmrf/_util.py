import pkgutil
import importlib
from datetime import datetime
from typing import Union, get_origin, get_args, Union
import sys
import logging
import inspect
from types import GenericAlias, UnionType
from equinox import field as base_field
           
# Temporarily not supporting derived fields
field = base_field
# def field(
#     *,
#     derived: bool = False,
#     **kwargs,
# ):    
#     metadata = dict(kwargs.pop('metadata', {}))
#     if 'derived' in metadata:
#         raise Exception("Cannot use metadata with `derived` already set.")
#     metadata['derived'] = derived
    
#     if derived:
#         kwargs['init'] = False
    
#     return base_field(metadata=metadata, **kwargs)

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

def update_dict_with_alias(original: dict, updates: dict, alias_map: dict) -> None:
    # Build prefix lookup trie (flattened since prefixes are strings)
    # Sort prefixes by length (longest first) to match the most specific prefix first
    sorted_aliases = sorted(alias_map.items(), key=lambda x: -len(x[0]))

    for key in original:
        for orig_prefix, update_prefix in sorted_aliases:
            if key.startswith(orig_prefix):
                aliased_key = update_prefix + key[len(orig_prefix):]
                if aliased_key in updates:
                    original[key] = updates[aliased_key]
                break
        # if no prefix matched, keep the original value
        
class classproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        return self.func(cls)

class LevelFilteredLogger:
    def __init__(self, null_level=logging.WARNING):
        self.null_level = null_level

    def _should_suppress(self, level):
        return level < self.null_level

    def debug(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.DEBUG):
            print(f"[DEBUG] {msg}", file=sys.stderr)

    def info(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.INFO):
            print(f"[INFO] {msg}", file=sys.stderr)

    def warning(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.WARNING):
            print(f"[WARNING] {msg}", file=sys.stderr)

    def error(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.ERROR):
            print(f"[ERROR] {msg}", file=sys.stderr)

    def critical(self, msg, *args, **kwargs):
        if not self._should_suppress(logging.CRITICAL):
            print(f"[CRITICAL] {msg}", file=sys.stderr)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        if not self._should_suppress(logging.ERROR):
            print(f"[EXCEPTION] {msg}", file=sys.stderr)

def is_overridden(cls, baseclass, method_name):
    result = False
    for cls in inspect.getmro(cls):
        if method_name in cls.__dict__:
            result = cls is not baseclass
            break
    return result

def is_instance_of_annotated_type(instance, annotated_type) -> bool:
    origin = get_origin(annotated_type)
    args = get_args(annotated_type)

    if origin is UnionType:
        # Union or Optional
        return any(is_instance_of_annotated_type(instance, arg) for arg in args)

    elif origin is not None:
        # Handles e.g. Annotated[T, ...], Literal[T], etc.
        return is_instance_of_annotated_type(instance, args[0])

    else:
        return isinstance(instance, annotated_type)

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