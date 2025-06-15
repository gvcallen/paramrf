import jax
import equinox as eqx
import skrf as skrf
import inspect
import dataclasses

from pmrf._parameter import Parameter
from pmrf._numpy import numpy as np

from numbers import Number
from typing import Literal, Sequence, Union, get_args

PRIMARY_PROPERTIES = ['s', 'a']
unit = ['w', 'f']

NumberLike = Union[Number, Sequence[Number], np.ndarray]

jax.config.update("jax_enable_x64", True)

class Model(eqx.Module):
    _z0: jax.Array = eqx.field(default=50.0, init=False, static=True)
    _unit: str = eqx.field(default='w', init=False, static=True)
    s_def: str | None = eqx.field(default='power', init=False, static=True)
    name: str | None = eqx.field(default=None, static=True)

    def __init_subclass__(cls, unit = 'w', name = None, **kwargs):
        super().__init_subclass__(**kwargs)

        # Set the "asarray" converter for fields that are marked as `Parameter` with a default value but haven't been created using "field"
        for field_name, field_type in cls.__annotations__.items():
            if field_type is Parameter:
                try:
                    current_value = getattr(cls, field_name, None)
                    if current_value.__class__.__hash__ is None:
                        default_factory = lambda: current_value
                        field = eqx.field(default_factory=default_factory, converter=jax.numpy.asarray)
                    else:
                        field = eqx.field(default=current_value, converter=jax.numpy.asarray)

                    setattr(cls, field_name, field)
                except:
                    pass

        if unit is not None and unit not in unit:
            raise Exception("Only 'f' and 'w' unit are supported")
        
        cls._unit = unit
        cls.name = name

    def __new__(cls, *args, **kwargs):                  
        return eqx.Module.__new__(cls)
    
    def __pow__(self, other: 'Model') -> 'Model':
        return _Cascaded(self, other)

    @property    
    def primary_function(self):
        for property in PRIMARY_PROPERTIES:
            if _is_overridden(self, Model, property):
                attr = getattr(self, property)
                return attr
        
        raise Exception(f"No primary properties in {PRIMARY_PROPERTIES} are overrided, which are the only ones supported currently")
            
    @property    
    def primary_property(self):
        for property in PRIMARY_PROPERTIES:
            if _is_overridden(self, Model, property):
                return property
        raise Exception(f"No primary properties in {PRIMARY_PROPERTIES} are overrided, which are the only ones supported currently")

    @property
    def number_of_ports(self):
        return jax.eval_shape(self.s, (1))[1]
    
    @property
    def n_ports(self):
        return self.number_of_ports
    
    @property
    def z0(self):
        return self._z0
    
    def a(self, x) -> jax.Array:
        """ABCD Matrix as a function of frequency.

        Args:
            x: Frequency array. Units are specified by the sub-class.

        Returns:
            jax.Array: _description_
        """
        return s2a(self.s(x), self._z0)
    
    def s(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError("Error: sub-classes must implemented the 's' function")
    
    def flipped(self) -> 'Model':
        return _Flipped(self)
    
    def to_skrf(self, frequency: skrf.Frequency, **kwargs) -> skrf.Network:
        # Get the frequency argument
        if self._unit == 'f':
            x = frequency.f
        elif self._unit == 'w':
            x = frequency.w
        else:
            raise Exception("Unknown unit")
        
        # Evaluate the model
        f, fname = self.primary_function, self.primary_property
        kwargs = kwargs or {}
        kwargs.update({
            fname: f(x),
            'frequency': frequency,
            'name': kwargs.get('name', self.name),
            'z0': self._z0,
        })

        # Return network
        return skrf.Network(**kwargs)

class _Cascaded(Model):
    model1: Model
    model2: Model

    def __init__(self, model1: Model, model2: Model):
        if model1.n_ports != 2:
            raise Exception('First network must be a two port when cascaded')
        if model2.n_ports != 1 and model2.n_ports != 2:
            raise Exception('Second network must be a one port or two port when cascaded')
        self.model1 = model1
        self.model2 = model2

    def a(self, x):
        return self.model1.a(x) @ self.model2.a(x)

    def s(self, x):
        # Only works if ntwk 1 is two port and ntwk 2 is one port
        a1 = self.model1.a(x)
        s2 = self.model2.s(x)
        A, B, C, D = a1[:,0,0], a1[:,0,1], a1[:,1,0], a1[:,1,1]
        z0 = self.model1._z0
        num = z0 * (1 + s2) * (A - z0*C) + (B - D*z0)*(1-s2)
        den = z0 * (1 + s2) * (A + z0*C) + (B + D*z0)*(1-s2)
        s11 = num / den        
        return s11
    
class _Renumbered(Model):
    model: Model
    from_ports: jax.Array
    to_ports: jax.Array

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

    def renumber(self, p):
        p[:, self.to_ports, :] = p[:, self.from_ports, :]
        p[:, :, self.to_ports] = p[:, :, self.from_ports]
        return p
    
    def a(self, x):
        return self.renumber(self.model.a(x))

    def s(self, x):
        return self.renumber(self.model.s(x)) 

class _Flipped(_Renumbered):
    def __init__(self, model: Model):
        if self.number_of_ports % 2 != 0:
            raise ValueError('you can only flip multiple-of-two-port Networks')
        n = int(self.number_of_ports / 2)
        old = list(range(0, 2*n))
        new = list(range(n, 2*n)) + list(range(0, n))
        _Renumbered.__init__(self, model, old, new)        
    

def a2s(a: np.ndarray, z0: NumberLike = 50) -> np.ndarray:
    nfreqs, nports, nports = a.shape

    if nports != 2:
        raise IndexError('abcd parameters are defined for 2-ports networks only')

    z0 = _fix_z0_shape(z0, nfreqs, nports)
    z01 = z0[:,0]
    z02 = z0[:,1]
    A = a[:,0,0]
    B = a[:,0,1]
    C = a[:,1,0]
    D = a[:,1,1]
    denom = A*z02 + B + C*z01*z02 + D*z01

    s = np.array([
        [
            (A*z02 + B - C*z01.conj()*z02 - D*z01.conj() ) / denom,
            (2*np.sqrt(z01.real * z02.real)) / denom,
        ],
        [
            (2*(A*D - B*C)*np.sqrt(z01.real * z02.real)) / denom,
            (-A*z02.conj() + B - C*z01*z02.conj() + D*z01) / denom,
        ],
    ]).transpose()
    return s

def s2a(s: np.ndarray, z0: NumberLike = 50) -> np.ndarray:
    nfreqs, nports, nports = s.shape

    if nports != 2:
        raise IndexError('abcd parameters are defined for 2-ports networks only')

    z0 = _fix_z0_shape(z0, nfreqs, nports)
    z01 = z0[:,0]
    z02 = z0[:,1]
    denom = (2*s[:,1,0]*np.sqrt(z01.real * z02.real))
    a = np.array([
        [
            ((z01.conj() + s[:,0,0]*z01)*(1 - s[:,1,1]) + s[:,0,1]*s[:,1,0]*z01) / denom,
            ((1 - s[:,0,0])*(1 - s[:,1,1]) - s[:,0,1]*s[:,1,0]) / denom,
        ],
        [
            ((z01.conj() + s[:,0,0]*z01)*(z02.conj() + s[:,1,1]*z02) - s[:,0,1]*s[:,1,0]*z01*z02) / denom,
            ((1 - s[:,0,0])*(z02.conj() + s[:,1,1]*z02) + s[:,0,1]*s[:,1,0]*z02) / denom,
        ],
    ]).transpose()
    return a


def _is_overridden(self, baseclass, method_name):
    for cls in inspect.getmro(self.__class__):
        if method_name in cls.__dict__:
            return cls is not baseclass
    return False

def _fix_z0_shape(z0: NumberLike, nfreqs: int, nports: int) -> np.ndarray:
    if np.shape(z0) == (nfreqs, nports):
        return z0.copy()
    elif np.ndim(z0) == 0:
        return np.array(nfreqs * [nports * [z0]])
    elif len(z0) == nports:
        return np.array(nfreqs * [z0])
    elif len(z0) == nfreqs:
        return np.array(nports * [z0]).T
    else:
        raise IndexError('z0 is not an acceptable shape')
    