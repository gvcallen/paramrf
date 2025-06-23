from abc import abstractmethod

from pmrf.models.lumped import Resistor
from pmrf.models.topological import PiCLC

from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._misc import field

class NonIdealResistor(Model):
    cascaded: Model = field(derived=True)
    
    ideal: Resistor = Resistor()

    def __post_init__(self):
        self.cascaded = self.parasitics ** self.ideal
        
    def a(self, freq: Frequency):
        return self.cascaded.a(freq)

    @property
    @abstractmethod
    def parasitics(self) -> Model:
        raise Exception("Base classes must specify the form of the parasitics")

class CLCResistor(NonIdealResistor):
    clc: PiCLC = PiCLC()

    @property
    def parasitics(self) -> Model:
        return self.clc