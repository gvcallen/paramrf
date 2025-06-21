from abc import abstractmethod
from typing import final

import pmrf.numpy as np
from pmrf._misc import field
from pmrf._frequency import Frequency
from pmrf._model import Model
# from pmrf._compound import CompoundModel

# from skrf import Circuit
# connections: list[list[tuple[Network, int]]],

# class CircuitLayout:
#     _connections: list[list[tuple[Model, int]]] | list[Model]
#     _model: CompoundModel # the model that calculates the actual computation for the connections

    # def __init__(self, )


class CircuitModel(Model):
    # _layout: CircuitLayout | None = None 
    # _built_model: Model | None = field(default=None, init=False, repr=False)

    def build(self) -> Model:
        raise Exception("Sub-classes must implemented 'combine' to combine their sub-models into a single model")

    # @final
    # def post(self):
    #     self._built_model = self.combine()

    def a(self, freq: Frequency) -> np.ndarray:
        # TODO call combine once only
        return self.build().a(freq)
    
    def s(self, freq: Frequency) -> np.ndarray:
        return self.build().s(freq)