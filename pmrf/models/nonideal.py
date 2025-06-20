from abc import abstractmethod

from pmrf.models.lumped import Resistor
from pmrf.models.topological import PiCLC

from pmrf._model import Model
from pmrf._circuit import CircuitModel
from pmrf._misc import field

class NonIdealResistor(CircuitModel):
    ideal: Resistor = field(default_factory=lambda: Resistor())

    def build(self) -> Model:
        return self.parasitics ** self.ideal

    @property
    @abstractmethod
    def parasitics(self) -> Model:
        raise Exception("Base classes must specify the form of the parasitics")

class CLCResistor(NonIdealResistor):
    clc: PiCLC = field(default_factory=lambda: PiCLC())

    @property
    def parasitics(self) -> Model:
        return self.clc