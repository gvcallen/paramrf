from abc import abstractmethod
from typing import Sequence, Union

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

class CascadedModel(CompoundModel):
    models: list[Model]
    
    def __init__(self, models: list[Model], **kwargs):
        if models[0].n_ports != 2:
            raise Exception('First network must be a two port when cascaded')
        for model in models[1:-1]:
            if model.n_ports != 2:
                raise Exception('Inner networks must be two ports when cascaded')
        if models[-1].n_ports != 2 or models[-1].n_ports != 2:
            raise Exception('First network must be a two port when cascaded')

        Model.__init__(self, models, **kwargs)

    @property
    def models(self) -> list[Model]:
        return self.models        

    def a(self, freq: Frequency):
        a = self.models[0].a(freq)
        for model in self.models:
            a = a @ model.a(freq)

    def s(self, freq: Frequency):
        if len(self.n_submodels) != 2 or self.models[-1].n_ports != 1:
            return CompoundModel.s(self, freq)

        # Optimization for when we only have two models and the second is a one-port
        a0 = self.models[0].a(freq)
        s1 = self.models[1].s(freq)
        z0 = self.models[0]._z0
        
        A, B, C, D = a0[:,0,0], a0[:,0,1], a0[:,1,0], a0[:,1,1]
        num = z0 * (1 + s1) * (A - z0*C) + (B - D*z0)*(1-s1)
        den = z0 * (1 + s1) * (A + z0*C) + (B + D*z0)*(1-s1)
        s11 = num / den        
        
        return s11
    
class RenumberedModel(CompoundModel):
    model: Model
    from_ports: np.ndarray
    to_ports: np.ndarray

    def __init__(self, ntwk: Model, from_ports: Sequence[int], to_ports: Sequence[int]):
        self.model = ntwk
        self.from_ports = np.array(from_ports)
        self.to_ports = np.array(to_ports)

        if len(np.unique(from_ports)) != len(from_ports):
            raise ValueError('an index can appear at most once in from_ports or to_ports')
        if any(np.unique(from_ports) != np.unique(to_ports)):
            raise ValueError('from_ports and to_ports must have the same set of indices')
        if ntwk.primary_function(return_str=True)[1] == 'a' and len(from_ports) != 1 and len(to_ports) != 1:
            raise ValueError("(from_ports, to_ports) must be either (0, 1) or (1, 0) for 'a' primary networks")

        self.z0[:, to_ports] = self.z0[:, from_ports]

    @property
    def models(self) -> list[Model]:
        return self.models                

    def renumber(self, p):
        p[:, self.to_ports, :] = p[:, self.from_ports, :]
        p[:, :, self.to_ports] = p[:, :, self.from_ports]
        return p
    
    def a(self, x):
        return self.renumber(self.model.a(x))

    def s(self, x):
        return self.renumber(self.model.s(x)) 

class FlippedModel(RenumberedModel):
    def __init__(self, model: Model):
        if self.number_of_ports % 2 != 0:
            raise ValueError('you can only flip multiple-of-two-port Networks')
        n = int(self.number_of_ports / 2)
        old = list(range(0, 2*n))
        new = list(range(n, 2*n)) + list(range(0, n))
        RenumberedModel.__init__(self, model, old, new)

class TerminatedModel(CompoundModel):
    model: Model
    short: Short

    @property
    def models(self) -> list[Model]:
        return [self.short, self.model]

    def __init__(self, model: Model, termination_port=1):
        if self.number_of_ports != 2:
            raise ValueError('you can only terminated two-port Networks')

        self.model = model

    # TODO choose primary property based on sub-networks primary property
    def a(self, freq: Frequency) -> np.ndarray:
        return self.model.a(freq) @ self.short.a(freq)