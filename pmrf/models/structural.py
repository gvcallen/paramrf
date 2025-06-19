from typing import Sequence

import pmrf.numpy as np
from pmrf.models.lumped import Short
from pmrf._frequency import Frequency
from pmrf._model import Model

class CascadedModel(Model):
    _models: tuple[Model]
    
    def __init__(self, models: tuple[Model], **kwargs):
        if models[0].n_ports != 2:
            raise Exception('First network must be a two port when cascaded')
        for model in models[1:-1]:
            if model.n_ports != 2:
                raise Exception('Inner networks must be two ports when cascaded')
        if models[-1].n_ports not in (1, 2):
            raise Exception('Last network must either be a one port or a two port when cascaded')
        
        self._models = models
        Model.__init__(self, **kwargs)

    @property
    def first_model(self) -> Model:
        return self._models[0]
    
    @property
    def inner_models(self) -> tuple['Model']:
        return self._models[1:-1]
    
    @property
    def last_model(self) -> Model:
        return self._models[-1]

    def a(self, freq: Frequency):
        a = self.first_model.a(freq)
        for model in self.inner_models:
            a = a @ model.a(freq)
        if self.last_model.n_ports == 1:
            raise Exception('Cannot get abcd-matrix for a cascade of models terminated in a one-port')
        
        return a @ self.last_model.a(freq)

    def s(self, freq: Frequency):
        # We only implement s when we are terminating in a one-port.
        # Otherwise, we call the parent s, which will ultimatlely call the 'a' implementation above
        if self.last_model.n_ports != 1:
            return Model.s(self, freq)
        
        # Get abcd matrix of inners
        a = self.first_model.a(freq)
        for model in self.inner_models:
            a = a @ model.a(freq)
        
        # Terminated last in s11
        a = self._models[0].a(freq)
        s11 = self._models[1].s(freq)
        z0 = self._models[0]._z0
        
        A, B, C, D = a[:,0,0], a[:,0,1], a[:,1,0], a[:,1,1]
        num = z0 * (1 + s11) * (A - z0*C) + (B - D*z0)*(1-s11)
        den = z0 * (1 + s11) * (A + z0*C) + (B + D*z0)*(1-s11)
        s11_out = num / den        
        return s11_out
        
    
class RenumberedModel(Model):
    _model: Model
    _from_ports: np.ndarray
    _to_ports: np.ndarray

    def __init__(self, ntwk: Model, from_ports: Sequence[int], to_ports: Sequence[int]):
        self._model = ntwk
        self._from_ports = np.array(from_ports)
        self._to_ports = np.array(to_ports)

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
        p[:, self._to_ports, :] = p[:, self._from_ports, :]
        p[:, :, self._to_ports] = p[:, :, self._from_ports]
        return p
    
    def a(self, x):
        return self.renumber(self._model.a(x))

    def s(self, x):
        return self.renumber(self._model.s(x)) 

class FlippedModel(RenumberedModel):
    def __init__(self, model: Model):
        if self.number_of_ports % 2 != 0:
            raise ValueError("You can only flip multiple-of-two-port Networks")
        n = int(self.number_of_ports / 2)
        old = list(range(0, 2*n))
        new = list(range(n, 2*n)) + list(range(0, n))
        RenumberedModel.__init__(self, model, old, new)