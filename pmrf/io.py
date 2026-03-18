"""
IO helpers e.g. for model loading and saving.
"""
import os
import jsonpickle
from typing import BinaryIO

from pmrf.core.model import Model

def load(source: str | BinaryIO) -> Model:
    """
    Load a ParamRF model from a file or file-like object.

    Parameters
    ----------
    source : str or BinaryIO
        The path to the saved model file or an open file-like object 
        containing the model data.

    Returns
    -------
    Model
        The deserialized ParamRF model instance.
        
    Raises
    ------
    TypeError
        If the root model or any nested submodels fail to load and 
        silently degrade into dictionaries (usually due to moved classes).
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf8") as f:
            data = f.read()
    else:
        data = source.read()

    decoded = jsonpickle.decode(data)    
    
    # 1. Check if the root object degraded
    if not isinstance(decoded, Model):
        raise TypeError(
            f"Failed to load model. Expected a ParamRF Model, but got '{type(decoded).__name__}'. "
            "This almost always happens because the model's class was moved to a different module, "
            "renamed, or deleted. jsonpickle cannot find the import path and silently degraded it to a dict."
        )

    # 2. Recursively check for nested degraded models
    def _verify_no_degraded_models(obj, current_path="root"):
        if isinstance(obj, dict):
            # ParamRF models have tell-tale internal fields. If a dict has these, it's a dead model.
            if '_param_groups' in obj or '_separator' in obj or 'z0' in obj:
                raise TypeError(
                    f"Degraded submodel found at path '{current_path}'. "
                    "A nested model failed to instantiate and became a dictionary. "
                    "Did you move or rename a component class in your codebase?"
                )
            for k, v in obj.items():
                _verify_no_degraded_models(v, f"{current_path}.{k}")
                
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _verify_no_degraded_models(v, f"{current_path}[{i}]")
                
        elif hasattr(obj, '__dataclass_fields__'):
            # Safely traverse Equinox modules and dataclasses
            for f in obj.__dataclass_fields__:
                _verify_no_degraded_models(getattr(obj, f), f"{current_path}.{f}")

    # _verify_no_degraded_models(decoded)

    return decoded

def save(target: str | BinaryIO, model: Model):
    """
    Save a ParamRF model to a file or file-like object.
    ...
    """
    model_save = model._saveable()
    data = jsonpickle.encode(model_save)
    
    if isinstance(target, (str, os.PathLike)):
        with open(target, "w", encoding="utf8") as f:
            f.write(data)
    else:
        target.write(data)

__all__ = ['save', 'load']