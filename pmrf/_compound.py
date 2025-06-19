from abc import abstractmethod

from pmrf._model import Model

class CompoundModel(Model):
    @property
    @abstractmethod
    def models(self) -> list[Model]:
        # TODO implement this automagically
        raise NotImplementedError("'models' property must be implemented sub-classes for a CompoundModel")
    
    @property
    def num_submodels(self):
        return len(self.models)