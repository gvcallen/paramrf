from abc import abstractmethod, ABC

import skrf as skrf
import inspect
from typing import Callable, Any, Dict, get_args, get_origin, Union, Optional
from types import UnionType

from pmrf.numpy import USE_JAX
if USE_JAX:
    import jax
import equinox as eqx

import pmrf.numpy as np
from pmrf._misc import field
from pmrf._math import a2s, s2a
from pmrf._frequency import Frequency
from pmrf._pytree import tree_with_params, tree_params

PRIMARY_PROPERTIES = ['s', 'a']

jax.config.update("jax_enable_x64", True)

class Model(eqx.Module):
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
    _z0: np.ndarray = field(default=50.0+0j, init=False, static=True)
    s_def: str | None = field(default='power', init=False, static=True)
    name: str | None = field(default=None, kw_only=True, static=True)
    dynamic: tuple = field(default=(float, np.ndarray), kw_only=True, static=True)
    priority: tuple = field(default=(), kw_only=True, static=True)

    def __init_subclass__(cls, dynamic: tuple | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        dynamic = dynamic or cls.dynamic

        for dynamic_type in dynamic:
            if issubclass(dynamic_type, Model):
                raise Exception("Error: do not set `Model` types as dynamic")

        # Add metadata and field properties to certain sub-class fields since we have certains constraints for the API.
        # Currently, we add default, default_factory, converter, and kw_only where necessary
        for field_name, field_types in cls.__annotations__.items():
            # The annotations could be unions - in this case we just take the first one TODO upgrade this to do more in-depth inspection?
            origin = get_origin(field_types)
            if origin in (Union, UnionType):
                field_type = get_args(field_types)[0]
            else:
                field_type = field_types
            
            # We populate the field kwargs dynamically
            field_kwargs = {}

            # First, populate the default
            default = getattr(cls, field_name, None)
            if not default is None:
                # We use `default.__class__.__hash__` to guess if the type is mutable
                if default.__class__.__hash__ is None:
                    field_kwargs['default_factory'] = lambda: default
                else:
                    field_kwargs['default'] = default

                # If the type is static and has a default, we only allow it to be passed by key-word argument
                if not field_type in dynamic:
                    field_kwargs['kw_only'] = True

            # Next, populate the jax.array converter for types considered dynamic (even those without defaults).
            if field_type in dynamic:
                field_kwargs['converter'] = lambda val: jax.numpy.asarray(val, dtype=float)
                # try:
                #     _ = jax.numpy.asarray(default)
                # except:
                #     raise Exception(f"Could not convert field '{field_name}' with default '{default}' specified as type {field_type} in class {cls} to a dynamic field")
                

            # if issubclass(field_type, Model):
            #     print(f'Found model subclass! Name = {field_name}')
                    
            # Finally, create the field and replace the class's value (but only if we need to - no need if kwargs is ultimately empty)
            if len(field_kwargs) != 0:
                setattr(cls, field_name, field(**field_kwargs))

    def __new__(cls, *args, **kwargs):                  
        return eqx.Module.__new__(cls)
    
    def __pow__(self, other: 'Model') -> 'Model':
        from pmrf.models.structural import CascadedModel
        return CascadedModel([self, other])
        
    @property    
    def primary_function(self) -> Callable[[Frequency], np.ndarray]:
        return getattr(self, self.primary_property)
            
    @property    
    def primary_property(self) -> str:
        prioritized = self.priority
        unprioritized = set(PRIMARY_PROPERTIES).difference(set(self.priority))

        for property in prioritized:
            if is_overridden(self, Model, property):
                return property
        for property in unprioritized:
            if is_overridden(self, Model, property):
                return property
        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overrided, which are the only ones supported currently")

    @property
    def number_of_ports(self):
        freq = Frequency(skrf.Frequency(1, 1, 1))
        sf = lambda: self.s(freq)
        return sf().shape[1]
        # return jax.eval_shape(sf).shape[1]
    
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
        if not is_overridden(self, Model, 's'):
            raise NotImplementedError(f"Error: model sub-classes currently *have* to implement the 's' or the 'a' function, but class {type(self)} has neither")
        
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
        if not is_overridden(self, Model, 'a'):
            raise NotImplementedError(f"Error: model sub-classes currently *have* to implement the 's' or the 'a' function, but class {type(self)} has neither")
        
        a = self.a(freq)
        return a2s(a, self.z0)
           
    def flipped(self) -> 'Model':
        from pmrf.models.structural import FlippedModel
        return FlippedModel(self)
    
    def terminated(self, load: 'Model' = None) -> 'Model':
        from pmrf.models.lumped import Short
        from pmrf.models.structural import CascadedModel
        
        load = load or Short()
        terminated_model = CascadedModel((self, load))
        return terminated_model
    
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