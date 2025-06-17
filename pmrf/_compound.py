from abc import abstractmethod

import pmrf.numpy as np
from pmrf.models import Short
from pmrf._model import Model
from pmrf._frequency import Frequency

class CompoundModel(Model):
    @abstractmethod
    @property
    def models(self) -> list[Model]:
        # TODO implement this automagically
        raise NotImplementedError("'models' property must be implemented sub-classes for a CompoundModel")
    
    @property
    def num_submodels(self):
        return len(self.models)
    
    @property
    def n_submodels(self):
        return len(self.models)