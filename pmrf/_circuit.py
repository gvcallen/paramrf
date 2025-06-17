from abc import abstractmethod

import pmrf.numpy as np
from pmrf._frequency import Frequency
from pmrf._model import Model
from pmrf._compound import CompoundModel

# from skrf import Circuit
# connections: list[list[tuple[Network, int]]],

# class CircuitLayout:
#     _connections: list[list[tuple[Model, int]]] | list[Model]
#     _model: CompoundModel # the model that calculates the actual computation for the connections

    # def __init__(self, )


class CircuitModel(Model):
    # _layout: CircuitLayout | None = None 
    _built_model: CompoundModel | None = None

    def __post_init__(self):
        Model.__post_init__(self)
        self._built_model = self.build()

    @abstractmethod
    def build(self) -> CompoundModel:
        raise NotImplementedError("Error: circuit model sub-classes *have* to implement the build() function to build their circuit layout")
    
    def a(self, freq: Frequency) -> np.ndarray:
        return self._built_model.a(freq)
    
    def s(self, freq: Frequency) -> np.ndarray:
        self._built_model.s(freq)