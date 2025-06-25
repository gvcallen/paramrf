from abc import abstractmethod

from pmrf.models.lumped import Resistor
from pmrf.models.topological import PiCLC

from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._misc import field

class NonIdealResistor(Model):
    @property
    @abstractmethod
    def ideal(self) -> Model:
        raise Exception("Base classes must specify the form of the parasitics")

    @property
    @abstractmethod
    def parasitics(self) -> Model:
        raise Exception("Base classes must specify the form of the parasitics")

class CLCResistor(NonIdealResistor):
    cascaded: Model = field(derived=True)
    res: Resistor = Resistor()
    clc: PiCLC = PiCLC()

    @property
    def ideal(self) -> Model:
        return self.clc
    
    @property
    def parasitics(self) -> Model:
        return self.clc
    
    def __post_init__(self):
        self.cascaded = self.clc ** self.res
        
    def a(self, freq: Frequency):
        return self.cascaded.a(freq)