import equinox as eqx
import skrf as skrf
import inspect
import dataclasses
from dataclasses import dataclass
from typing import Callable, Any, Optional, Dict

from pmrf._numpy import USE_JAX
if USE_JAX:
    import jax
from pmrf._typing import Scalar, Vector

from jaxtyping import Float, Array
from numbers import Number
from typing import Sequence, Union

from pmrf.frequency import Frequency
from pmrf._numpy import numpy as np
from pmrf._pytree import tree_with_params, tree_params
from pmrf._math import dB20


PRIMARY_PROPERTIES = ['s', 'a']

NumberLike = Union[Number, Sequence[Number], np.ndarray]

jax.config.update("jax_enable_x64", True)


class Model(eqx.Module):
    """Class for a single network model.

    This is an abstract class and should not be instantiated directly.

    Models accept their parameters as input arguments into their intitializers, as well as general keyword arguments.
    Then, they can be used to calculate their properties as function of frequency (S-matrix, ABCD-matrix etc.)
    as well as a configurable "feature" matrix, with is their output when called as a function.
    
    Since all models derived from `dataclass`, arguments propagate to dervied classes.
    Therefore, the following arguments apply to sub-classes by default:

    Args:
        name (str, optional): A name associated with the model instance.
    """
    _z0: np.ndarray = eqx.field(default=50.0, init=False, static=True)
    # s_def: str | None = eqx.field(default='power', init=False, static=True)
    name: str | None = eqx.field(default=None, kw_only=True, static=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Set the "asarray" converter for fields that are marked as `Scalar` or `Vector` with a default value but haven't been created using "field"
        for field_name, field_type in cls.__annotations__.items():
            # TODO type check that a vector value was not used for `Scalar` and vica versa
            if field_type is Scalar or field_type is Vector:
                try:
                    current_value = getattr(cls, field_name, None)

                    if field_type is Vector:
                        dtype = type(current_value[0])
                    else:
                        dtype = type(current_value)

                    converter = lambda val: jax.numpy.asarray(val, dtype=dtype)
                    if current_value.__class__.__hash__ is None:
                        default_factory = lambda: current_value
                        field = eqx.field(default_factory=default_factory, converter=converter)
                    else:
                        field = eqx.field(default=current_value, converter=converter)

                    setattr(cls, field_name, field)
                except:
                    pass

    def __new__(cls, *args, **kwargs):                  
        return eqx.Module.__new__(cls)
    
    def __pow__(self, other: 'Model') -> 'Model':
        return _Cascaded(self, other)
    
    def __call__(self, freq: Frequency):
        n_frequencies = len(freq)
        n_features = len(self.features)

        X = np.zeros((n_frequencies, n_features), dtype=np.complex128)
        for d, feature in enumerate(self.features):
            if USE_JAX:
                X = X.at[:, d].set(feature.extract_from_network(self.measured) - feature.extract_from_model(model, freq))
            else:
                X[:, d] = feature.extract_from_network(self.measured) - feature.extract_from_model(model, freq)        

    
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
       
    def a(self, freq: Frequency) -> np.ndarray:
        """Calculates the abcd parameter matrix as a function of frequency.

        This is one of the primary property functions that derived classes may implemented.
        If not implemented, and at least one other primary function has been implemented,
        then conversion formulae or used dynamically to calculate the resultant matrix.

        Args:
            freq (Frequency): Specifies the frequency to calculate the abcd-parameters at.

        Returns:
            np.ndarray: The resultant abcd matrix.
        """
        if self.primary_property != 's':
            raise NotImplementedError("Error: sub-classes currently *have* to implement the 's' or the 'a' function")
        
        s = self.s(freq)
        return s2a(s, self.z0)
    
    def s(self, freq: Frequency) -> np.ndarray:
        """Calculates the S parameter matrix as a function of frequency.

        This is one of the primary property functions that derived classes may implemented.
        If not implemented, and at least one other primary function has been implemented,
        then conversion formulae or used dynamically to calculate the resultant matrix.

        Args:
            freq (Frequency): Specifies the frequency to calculate the S-parameters at.

        Returns:
            np.ndarray: The resultant S matrix.
        """
        if self.primary_property != 'a':
            raise NotImplementedError("Error: sub-classes currently *have* to implement the 's' or the 'a' function")
        
        a = self.a(freq)
        return a2s(a, self.z0)
           
    def flipped(self) -> 'Model':
        return _Flipped(self)
    
    def with_params(
        self,
        flat_params: jax.Array | None = None,
        separator: str | None = '_',
        submodel_separator: str | None = None,
        array_separator: str | None = None,
        index_separator: str | None = None,
        param_filter: Callable[[Any], bool] | None = None,
        **params: Any
    ) -> "Model":
        """
        Returns a model with the specified parameter values.

        This method supports two calling styles:
        1. By keyword: `model.with_params(R=50.0, C=1e-9)`
        2. By flat array: `model.with_params(np.array([50.0, 1e-9]))`

        For the keyword style, separators are specified using the relavant arguments.
        In this case, the expected keys are those returned by `self.parameter_paths(..)` used with the same arguments.

        Args:
            flat_params: A 1D array containing all dynamic parameter values in their flattened tree order.
            separator (str | None, optional): The separator to use for all dividers that are not passed. Defaults to '_'.
            submodel_separator (str | None, optional): The separator before submodels. Defaults to `None`, in which case `separator` is used.
            array_separator (str | None, optional): The separator before array-like parameters. Defaults to `None`, in which case `separator` is used.
            index_separator (str | None, optional): The separator between array sub-indices. Defaults to `None`, in which case `separator` is used.
            param_filter (Callable[[Any], bool], optional): A filter to determine which fields are considered parameters. Defaults to `None`, in which case only the default `Scalar` and `Vector` types are considered.            
            **params: Keyword arguments, where keys are the names of the parameters to update and values are their new values.

        Returns:
            A new `Model` instance with the specified parameters updated.
        """
        return tree_with_params(self, flat_params=flat_params, separator=separator, subtree_separator=submodel_separator, array_separator=array_separator, index_separator=index_separator, param_filter=param_filter, **params)
    
    def params(
        self,
        flat: bool = False,
        separator: str | None = '_',
        submodel_separator: str | None = None,
        array_separator: str | None = None,
        index_separator: str | None = None,
        param_filter: Callable[[Any], bool] | None = None,
    ) -> Union[Dict[str, Any], jax.Array]:
        """Returns an dictionary of human-readable string paths and values for every
        scalar value in the flattened parameters.

        This is useful for mapping parameter names to values for external
        solvers, setting bounds, or interpreting results.

        Args:
            separator (str | None, optional): The separator to use for all dividers that are not passed. Defaults to '_'.
            submodel_separator (str | None, optional): The separate before submodels. Defaults to `None`, in which case `separator` is used.
            array_separator (str | None, optional): The separate before array-like parameter. Defaults to `None`, in which case `separator` is used.
            index_separator (str | None, optional): The separator between array sub-indices_. Defaults to `None`, in which case `separator` is used.
            param_filter (Callable[[Any], bool], optional): A filter to determine which fields are considered parameters. Defaults to `None`, in which case only the default `Scalar` and `Vector` types are considered.

        Returns:
            A dictionary of parameter names/paths and values e.g. {'R': 0.0, 'sub_L': 1.0, 'sub.C[0,0]': 2.0, 'sub.C[0,1]': 3.0, ...].
        """
        return tree_params(self, flat=flat, separator=separator, subtree_separator=submodel_separator, array_separator=array_separator, index_separator=index_separator, param_filter=param_filter)    
    
    def to_skrf(self, freq: skrf.Frequency, **kwargs) -> skrf.Network:
        f, fname = self.primary_function, self.primary_property
        kwargs = kwargs or {}
        kwargs.update({
            fname: f(Frequency(freq)),
            'frequency': freq,
            'name': kwargs.get('name', self.name),
            'z0': self._z0,
        })

        return skrf.Network(**kwargs)

class _Cascaded(Model):
    model1: Model
    model2: Model

    def __init__(self, model1: Model, model2: Model, **kwargs):
        if model1.n_ports != 2:
            raise Exception('First network must be a two port when cascaded')
        if model2.n_ports != 1 and model2.n_ports != 2:
            raise Exception('Second network must be a one port or two port when cascaded')
        self.model1 = model1
        self.model2 = model2

        Model.__init__(self, **kwargs)

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
    