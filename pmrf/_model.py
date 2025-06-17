from abc import abstractmethod, ABC

import skrf as skrf
import inspect
from typing import Callable, Any, Dict

from pmrf.numpy import USE_JAX
if USE_JAX:
    import jax
import equinox as eqx

import pmrf.numpy as np
from pmrf._math import a2s, s2a
from pmrf._frequency import Frequency
from pmrf._pytree import tree_with_params, tree_params


PRIMARY_PROPERTIES = ['s', 'a']

jax.config.update("jax_enable_x64", True)

class Model(eqx.Module, ABC):
    """Base class representing an RF network that is computable, referred to in param-rf as a `Model`.

    This is an abstract class and should not be instantiated directly.

    Model initializers accept their parameters and sub-networks as input arguments, as well as general keyword arguments.
    Then, they can be used to calculate their properties as function of frequency (S-matrix, ABCD-matrix etc.)
    as well as a configurable "feature" matrix, with is their output when called as a function.
    
    Since all models derived from `dataclass`, arguments propagate to dervied classes.
    Therefore, the following arguments apply to sub-classes by default:

    Args:
        name (str, optional): A name associated with the model instance.
    """
    _z0: np.ndarray = eqx.field(default=50.0, init=False, static=True)
    s_def: str | None = eqx.field(default='power', init=False, static=True)
    name: str | None = eqx.field(default=None, kw_only=True, static=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Set the "asarray" converter for fields that are marked as `float` or `np.ndarray` with a default value but haven't been created using "field"
        for field_name, field_type in cls.__annotations__.items():
            # TODO type check that a vector value was not used for `Scalar` and vica versa
            if field_type is float or field_type is np.ndarray:
                try:
                    current_value = getattr(cls, field_name, None)

                    if field_type is np.ndarray:
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
        from pmrf._compound import CascadedModel
        return CascadedModel(self, other)
        
    @property    
    def primary_function(self):
        for property in PRIMARY_PROPERTIES:
            if is_overridden(self, Model, property):
                attr = getattr(self, property)
                return attr        
        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overrided, which are the only ones supported currently")
            
    @property    
    def primary_property(self):
        for property in PRIMARY_PROPERTIES:
            if is_overridden(self, Model, property):
                return property
        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overrided, which are the only ones supported currently")

    @property
    def number_of_ports(self):
        return jax.eval_shape(self.s, (1))[1]
    
    @property
    def n_ports(self):
        return self.number_of_ports
    
    @property
    def z0(self):
        return self._z0
       
    @abstractmethod
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
            raise NotImplementedError("Error: model sub-classes currently *have* to implement the 's' or the 'a' function")
        
        s = self.s(freq)
        return s2a(s, self.z0)
    
    @abstractmethod
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
            raise NotImplementedError("Error: model sub-classes currently *have* to implement the 's' or the 'a' function")
        
        a = self.a(freq)
        return a2s(a, self.z0)
           
    def flipped(self) -> 'Model':
        from pmrf.models.structural import FlippedModel
        return FlippedModel(self)
    
    def terminated(self) -> 'Model':
        from pmrf.models.structural import TerminatedModel
        return TerminatedModel(self)
    
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
            param_filter (Callable[[Any], bool], optional): A filter to determine which fields are considered parameters. Defaults to `None`, in which case only the model `float` and `np.ndarray` types are considered.
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
    ) -> Dict[str, Any] | np.ndarray:
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

def is_overridden(self, baseclass, method_name):
    for cls in inspect.getmro(self.__class__):
        if method_name in cls.__dict__:
            return cls is not baseclass
    return False