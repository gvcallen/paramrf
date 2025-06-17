from abc import abstractmethod

from pmrf.models.lumped import Resistor
from pmrf.models.topological import PiCLC

from pmrf._model import Model
from pmrf._circuit import CircuitModel

class NonIdealResistor(CircuitModel):
    ideal: Resistor = Resistor()

    def build(self) -> Model:
        return self.parasitics ** self.ideal

    @property
    @abstractmethod
    def parasitics(self) -> Model:
        raise Exception("Base classes must specify the form of the parasitics")

class CLCResistor(NonIdealResistor):
    clc: PiCLC = PiCLC()

    @property
    def parasitics(self) -> Model:
        return self.clc