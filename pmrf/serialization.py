"""
Model loading and saving.
"""

import os
from pathlib import Path
from typing import BinaryIO, Any
import sys

import importlib
import json
import dataclasses

import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from jaxtyping import PyTree

# Define the public namespaces where your users typically import from.
# We use this to mask internal deep paths.
PUBLIC_NAMESPACES = ['pmrf.models', 'pmrf.parameters', 'pmrf.simulate', 'pmrf']

def _get_public_module(cls: type) -> str:
    """
    Finds the highest-level public namespace for a class to prevent 
    serialization files from relying on fragile internal directory structures.
    """
    for ns in PUBLIC_NAMESPACES:
        if ns in sys.modules:
            if getattr(sys.modules[ns], cls.__name__, None) is cls:
                return ns
    return cls.__module__

def _get_jsonpickle(for_serialization: bool = True):
    """
    Centralized loader for jsonpickle.
    Attempts to import jsonpickle and register useful extensions (like pandas/numpy)
    to ensure robust fallback serialization for complex internal objects.
    """
    try:
        import jsonpickle
    except ImportError:
        if for_serialization:
            raise TypeError(
                "Cannot serialize the current object.\n"
                "Ensure all custom classes are wrapped in `eqx.Module`, or install "
                "`jsonpickle` (`pip install jsonpickle`) to enable automatic fallback serialization."
            )
        else:
            raise ImportError(
                "This model contains a node serialized with `jsonpickle`, but the library "
                "is not currently installed. Please run `pip install jsonpickle` to load this file."
            )
            
    # Register pandas and numpy handlers safely if the libraries exist
    try:
        import jsonpickle.ext.numpy as jsonpickle_np
        jsonpickle_np.register_handlers()
    except ImportError:
        pass

    try:
        import jsonpickle.ext.pandas as jsonpickle_pd
        jsonpickle_pd.register_handlers()
    except ImportError:
        pass

    return jsonpickle

def _serialize_generic(node: Any) -> Any:
    """Recursively converts any PyTree/Equinox node into a JSON-serializable dict.
    Falls back to jsonpickle for unregistered or non-standard objects if installed.
    """
    # Base Case: Standard Python primitives
    if isinstance(node, (int, float, str, bool, type(None))):
        return node
        
    # Base Case: Native Python complex numbers
    if isinstance(node, complex):
        return {
            "__type__": "__complex__",
            "__real__": node.real,
            "__imag__": node.imag
        }
        
    # Intercept PyTreeDef objects before jsonpickle to prevent JAX C++ pickling errors
    if isinstance(node, jtu.PyTreeDef):
        # Create a dummy tree with `None` replacing all active arrays/leaves
        dummy_leaves = [None] * node.num_leaves
        dummy_tree = jtu.tree_unflatten(node, dummy_leaves)
        
        return {
            "__type__": "__pytreedef__",
            "__dummy_tree__": _serialize_generic(dummy_tree)
        }
        
    # 1. Dynamic Nodes: Equinox Modules (Dataclasses)
    # Always processed first so Equinox modules are serialized by state, not reference.
    if hasattr(node, "__dataclass_fields__"):
        state = {}
        for field_name, field in node.__dataclass_fields__.items():
            val = getattr(node, field_name)
            
            # Skip saving if the value exactly matches the field's explicit default
            if field.default is not dataclasses.MISSING:
                try:
                    if val == field.default:
                        continue 
                except Exception:
                    pass
            
            # Skip saving if the value matches the default_factory output
            if field.default_factory is not dataclasses.MISSING:
                try:
                    if val == field.default_factory():
                        continue
                except Exception:
                    pass
                    
            state[field_name] = _serialize_generic(val)
                
        cls = type(node)
        return {
            "__type__": "__dynamic_node__",
            "__module__": _get_public_module(cls),
            "__class__": cls.__name__,
            "__state__": state
        }
        
    # 2. Base Case: Callables and Function Wrappers
    # Functions, classes, and JAX trace objects (e.g., custom_jvp) should be serialized by reference.
    # To ensure it's a valid global reference (and not a stateful instance), we dynamically verify it resolves.
    if callable(node) and hasattr(node, "__module__") and hasattr(node, "__name__"):
        node_name = getattr(node, "__qualname__", node.__name__)
        if "<lambda>" not in node_name:
            try:
                # Verify the object is globally resolvable from its module
                mod = sys.modules.get(node.__module__)
                if mod is None:
                    mod = importlib.import_module(node.__module__)
                
                resolved_obj = mod
                for part in node_name.split("."):
                    resolved_obj = getattr(resolved_obj, part)
                
                # If it successfully resolves without error, it is safe to serialize by reference
                return {
                    "__type__": "__callable__",
                    "__module__": node.__module__,
                    "__name__": node_name
                }
            except Exception:
                pass  # Dynamic, local, or unresolvable callable; safely fall through
        
    # Base Case: Pure Arrays (jax.Array, np.ndarray, etc.)
    if eqx.is_array_like(node):
        arr = jnp.asarray(node)
        
        # Handle complex arrays
        if jnp.iscomplexobj(arr):
            return {
                "__type__": "__complex_array__",
                "__dtype__": str(getattr(arr, "dtype", "complex64")),
                "__real__": arr.real.tolist(),
                "__imag__": arr.imag.tolist()
            }
        # Handle standard real arrays
        else:
            return {
                "__type__": "__array__",
                "__dtype__": str(getattr(arr, "dtype", "float32")),
                "__data__": arr.tolist()
            }
        
    # Standard Containers
    if isinstance(node, dict):
        return {k: _serialize_generic(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_serialize_generic(x) for x in node]
        
    # Fallback: jsonpickle for arbitrary objects (Centralized)
    jsonpickle = _get_jsonpickle(for_serialization=True)
    
    try:
        pickled_string = jsonpickle.encode(node)
        return {
            "__type__": "__jsonpickle__",
            "__payload__": json.loads(pickled_string) 
        }
    except Exception as e:
        raise TypeError(
            f"Cannot serialize {type(node)}. Custom traversal failed, "
            f"and jsonpickle fallback also raised an error: {e}"
        )

def _deserialize_generic(data: Any) -> Any:
    """Recursively reconstructs PyTrees/Equinox modules from a JSON dict."""
    
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
        
    if isinstance(data, list):
        return [_deserialize_generic(x) for x in data]
        
    if isinstance(data, dict):
        node_type = data.get("__type__")
        
        # Rebuild Callables and Functions
        if node_type == "__callable__":
            module_name = data["__module__"] or "builtins"
            name = data["__name__"]
            
            def _get_obj(mod_name):
                mod = importlib.import_module(mod_name)
                obj = mod
                for part in name.split("."):
                    obj = getattr(obj, part)
                return obj

            try:
                return _get_obj(module_name)
            except Exception:
                # Resiliency Fallback
                for ns in PUBLIC_NAMESPACES:
                    try:
                        return _get_obj(ns)
                    except Exception:
                        continue
                raise ImportError(f"Could not reconstruct callable '{module_name}.{name}'.")
        
        # Rebuild the PyTreeDef from the dummy tree
        if node_type == "__pytreedef__":
            dummy_tree = _deserialize_generic(data["__dummy_tree__"])
            return jtu.tree_structure(dummy_tree)
        
        # Rebuild native complex number
        if node_type == "__complex__":
            return complex(data["__real__"], data["__imag__"])
            
        # Rebuild complex array
        if node_type == "__complex_array__":
            real_part = jnp.array(data["__real__"])
            imag_part = jnp.array(data["__imag__"])
            return jnp.array(real_part + 1j * imag_part, dtype=data["__dtype__"])

        # Rebuild standard Array
        if node_type == "__array__":
            return jnp.array(data["__data__"], dtype=data["__dtype__"])
            
        # Rebuild Dynamic Equinox Module
        if node_type == "__dynamic_node__":
            module_name = data["__module__"]
            class_name = data["__class__"]
            
            cls = None
            
            # 1. Try to load from the exact module saved in the file
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
            except ImportError:
                pass
                
            # Resiliency Fallback: If internal files moved, check public APIs
            if cls is None:
                for ns in PUBLIC_NAMESPACES:
                    try:
                        fallback_mod = importlib.import_module(ns)
                        if hasattr(fallback_mod, class_name):
                            cls = getattr(fallback_mod, class_name)
                            break
                    except ImportError:
                        continue
                        
            if cls is None:
                raise ImportError(f"Could not reconstruct '{class_name}'. Tried loading from '{module_name}' and public 'pmrf' namespaces.")
            
            instance = object.__new__(cls) 
            
            for field_name, field_def in getattr(cls, "__dataclass_fields__", {}).items():
                if field_name in data["__state__"]:
                    # The field was saved in the file; deserialize it
                    val = _deserialize_generic(data["__state__"][field_name])
                    object.__setattr__(instance, field_name, val)
                else:
                    # The field was skipped during save; restore its default
                    if field_def.default is not dataclasses.MISSING:
                        object.__setattr__(instance, field_name, field_def.default)
                    elif field_def.default_factory is not dataclasses.MISSING:
                        object.__setattr__(instance, field_name, field_def.default_factory())
                
            return instance

        # Rebuild jsonpickle fallback (Centralized)
        if node_type == "__jsonpickle__":
            jsonpickle = _get_jsonpickle(for_serialization=False)
            pickled_string = json.dumps(data["__payload__"])
            return jsonpickle.decode(pickled_string)
            
        # Standard dictionary
        return {k: _deserialize_generic(v) for k, v in data.items()}

def tree_save_json(filepath: str, model: PyTree) -> None:
    """Serializes an Equinox model or PyTree and saves it to a JSON file."""
    serialized_tree = _serialize_generic(model)
    with open(filepath, "w") as f:
        json.dump(serialized_tree, f, indent=4)

def tree_load_json(filepath: str) -> PyTree:
    """Loads a serialized Equinox model or PyTree from a JSON file."""
    with open(filepath, "r") as f:
        serialized_tree = json.load(f)
    return _deserialize_generic(serialized_tree)

def save(target: str | os.PathLike | BinaryIO, tree: Any):
    """
    (experimental) Save a ParamRF Model (or any Parax PyTree) to a file.
    ...
    """
    if isinstance(target, (str, os.PathLike)):
        target_path = Path(target)
        if not target_path.suffix:
            target_path = target_path.with_suffix('.prf')
        target = target_path

    tree_save_json(target, tree)


def load(source: str | os.PathLike | BinaryIO) -> Any:
    """
    (experimental) Load a ParamRF Model (or any Parax PyTree) from a file.
    ...
    """
    if isinstance(source, (str, os.PathLike)):
        source_path = Path(source)
        if not source_path.exists() and not source_path.suffix:
            prf_path = source_path.with_suffix('.prf')
            if prf_path.exists():
                source = prf_path
        else:
            source = source_path

    return tree_load_json(source)


__all__ = ['save', 'load']