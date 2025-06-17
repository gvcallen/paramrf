from abc import abstractmethod

import pmrf.numpy as np
from pmrf.models.lumped import Resistor
from pmrf.models.topological import PiCLC

from pmrf._frequency import Frequency
from pmrf._model import Model

class NonIdealResistor(Model):
    ideal: Resistor

    def __init__(self):
        self.ideal = Resistor()

    def combined(self) -> Model:
        return self.parasitics ** self.ideal

    @property
    @abstractmethod
    def parasitics(self) -> Model:
        raise Exception("Base classes must specify the form of the parasitics")

class CLCResistor(NonIdealResistor):
    clc: PiCLC

    @property
    def parasitics(self) -> Model:
        return self.clc