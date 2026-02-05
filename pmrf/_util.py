import jax.numpy as jnp
import pkgutil
import importlib
from datetime import datetime
from typing import Union, get_origin, get_args, Union
import sys
import logging
import inspect
from types import GenericAlias, UnionType
from equinox import field

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
except:
    RANK = 0
    MPI_AVAILABLE = False

def is_convertible_to_float(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

def wait_for_all_ranks():
    if not MPI_AVAILABLE:
        return
    COMM.Barrier()

def sync_across_all_ranks(x, root=0):
    if not MPI_AVAILABLE:
        return
    return COMM.bcast(x, root=root)
        
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

def is_overridden(cls, baseclass, method_name):
    result = False
    for cls in inspect.getmro(cls):
        if method_name in cls.__dict__:
            result = cls is not baseclass
            break
    return result

def defining_class(obj, attr):
    for cls in obj.__class__.__mro__:
        if attr in cls.__dict__:
            return cls
    return None

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

def explicit_kwargs():
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    func_obj = frame.f_globals[func_name]
    sig = inspect.signature(func_obj)

    bound = sig.bind_partial(**frame.f_locals)

    result = {}
    for name, value in bound.arguments.items():
        param = sig.parameters[name]
        # Only keep keyword-only parameters
        # or positional-or-keyword parameters that were passed via keyword
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            result[name] = value
        elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            # Passed by keyword if it's in locals and not before var-positional
            if name in frame.f_locals:
                # Check if it was supplied via keyword
                if name in frame.f_locals and name not in bound.args:
                    result[name] = value
    return result

def interp_tree(x_old, x_new, tree):
    """Recursively interpolate any pytree of JAX arrays."""
    if isinstance(tree, jnp.ndarray):
        return jnp.interp(x_old, x_new, tree)
    elif isinstance(tree, (float, int)):
        return tree
    elif isinstance(tree, dict):
        return {k: interp_tree(x_old, x_new, v) for k, v in tree.items()}
    elif isinstance(tree, (tuple, list)):
        return type(tree)(interp_tree(x_old, x_new, v) for v in tree)
    else:
        raise TypeError(f"Cannot interpolate object of type {type(tree)}")


def interp_distribution(x_old, x_new, d):
    """
    Interpolates a numpyro Distribution `d` by tree-interpolating
    its internal parameter dictionary _params, then reconstructing
    the same distribution class with the new parameters.
    """
    # Extract internal parameters (e.g., mean, scale, covariance, logits, etc.)
    params = d._params

    # Interpolate each parameter
    new_params = {
        k: interp_tree(x_old, x_new, v) for k, v in params.items()
    }

    # Rebuild distribution of same class with new parameters
    return type(d)(**new_params)