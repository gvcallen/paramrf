import numpy as np
from typing import Any, Sequence
import jax
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

def remove_constant_params(list_of_params: list[dict[str, Any]]):
    # 1. Identify all unique parameter keys present in the networks
    all_keys = set()
    for params in list_of_params:
        all_keys.update(params.keys())

    # 2. Find keys that have the same value across all networks
    keys_to_remove = []
    for key in all_keys:
        # Get values for this key from every network (None if missing)
        values = [params.get(key) for params in list_of_params]
        
        # Check if all values are identical
        # Using set(values) works for hashable types (ints, floats, strings)
        if len(set(values)) == 1:
            keys_to_remove.append(key)

    # 3. Purge the static keys from every network
    for params in list_of_params:
        for key in keys_to_remove:
            if key in params:
                del params[key]
                
        
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

def lhs_sample(N: int, d: int, key=None) -> jnp.ndarray:
    key_perm, key_noise = jax.random.split(key)
    keys_perm = jax.random.split(key_perm, d)
    perms = jax.vmap(lambda k: jax.random.permutation(k, N))(keys_perm)
    noise = jax.random.uniform(key_noise, shape=(d, N))
    lhs_unit = (perms + noise) / N
    lhs_unit = lhs_unit.T
    return lhs_unit    

def no_recent_improvement(values, patience):
    values = list(values)
    best_idx = min(range(len(values)), key=lambda i: values[i])
    return len(values) - 1 - best_idx >= patience

def has_converged(
    y_history: Sequence[float],
    *,
    window: int = 5,
    rtol: float = 1e-6,
    eps: float = 1e-12,
) -> bool:
    """
    Robust convergence check using relative tolerance.

    Converged if the maximum oscillation in the last `window`
    iterations is small relative to the magnitude of the function.

    max(|Δy|) / max(|y|) < rtol
    """
    if len(y_history) < window + 1:
        return False

    recent = jnp.asarray(y_history[-(window + 1):])
    deltas = jnp.abs(jnp.diff(recent))

    scale = jnp.max(jnp.abs(recent))
    scale = max(scale, eps)  # avoid division by zero

    return jnp.max(deltas) / scale < rtol

import matplotlib.pyplot as plt
import numpy as np

class LivePlotter:
    def __init__(self, title="Live Plot", xlabel="X", ylabel="Y"):
        plt.ion()  # interactive mode ON

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Dictionary to store data and line objects: 
        # { "label_name": { "x": [], "y": [], "line": line_object } }
        self.lines = {} 
        
        self.fig.show()

    def _get_or_create_line(self, label, color=None):
        """Helper to create a new line if the label doesn't exist."""
        if label not in self.lines:
            line, = self.ax.plot([], [], label=label, lw=1.0, color=color)
            self.lines[label] = {
                "x": [], 
                "y": [], 
                "line": line
            }
            # self.ax.legend(loc='upper left')
        return self.lines[label]

    def _redraw(self):
        """Handles the canvas refresh and scaling."""
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # MODE 1: Growing Axis (Stream)
    def add_point(self, label, value, x_value=None):
        """
        Appends a single value to the plot. 
        If x_value is None, it increments automatically based on list length.
        """
        data = self._get_or_create_line(label)
        
        # Append Y
        data["y"].append(value)
        
        # Determine X
        if x_value is not None:
            data["x"].append(x_value)
        else:
            # If no X provided, use the current step index
            data["x"].append(len(data["y"]) - 1)

        # Update the specific line object
        data["line"].set_data(data["x"], np.array(data["y"]))
        
        self._redraw()

    # MODE 2: Full Curve (Snapshot)
    def add_curve(self, label, y_values, x_values=None):
        """
        Replaces the entire curve for a specific label.
        Useful for plotting functions or distributions that change over time.
        """
        data = self._get_or_create_line(label)
        
        # Generate X if not provided
        if x_values is None:
            x_values = np.arange(len(y_values))
            
        # Replace data
        data["x"] = x_values
        data["y"] = y_values
        
        # Update line
        data["line"].set_data(data["x"], data["y"])
        
        self._redraw()