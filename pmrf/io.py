from pmrf.models.model import Model
from typing import BinaryIO

def load(source: str | BinaryIO) -> Model:
    return Model.load(source)

def save(target: str | BinaryIO, model: Model):
    return model.save(target)