"""
IO helpers e.g. for model loading and saving.
"""
import os
import jsonpickle
from typing import BinaryIO

from pmrf.models.model import Model

def load(source: str | BinaryIO) -> Model:
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf8") as f:
            data = f.read()
    else:
        data = source.read()

    return jsonpickle.decode(data)    

def save(target: str | BinaryIO, model: Model):
    model_save = model._saveable()
    data = jsonpickle.encode(model_save)
    
    if isinstance(target, (str, os.PathLike)):
        with open(target, "w", encoding="utf8") as f:
            f.write(data)
    else:
        target.write(data)