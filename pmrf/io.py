"""
IO helpers e.g. for model loading and saving.
"""
import os
import jsonpickle
from typing import BinaryIO

from pmrf.models.model import Model

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
        
    Notes
    -----
    This function uses ``jsonpickle`` to reconstruct the model hierarchy 
    from its serialized state.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf8") as f:
            data = f.read()
    else:
        # Note: If the source is BinaryIO, it might need .decode() 
        # depending on if jsonpickle expects a string or bytes.
        data = source.read()

    return jsonpickle.decode(data)    

def save(target: str | BinaryIO, model: Model):
    """
    Save a ParamRF model to a file or file-like object.

    Parameters
    ----------
    target : str or BinaryIO
        The path where the model should be saved, or an open 
        file-like object.
    model : Model
        The ParamRF model instance to serialize.

    Notes
    -----
    The model is first converted to a saveable state via its internal 
    ``_saveable()`` method before being encoded into JSON format 
    using ``jsonpickle``.
    """
    model_save = model._saveable()
    data = jsonpickle.encode(model_save)
    
    if isinstance(target, (str, os.PathLike)):
        with open(target, "w", encoding="utf8") as f:
            f.write(data)
    else:
        target.write(data)