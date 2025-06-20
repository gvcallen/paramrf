import skrf as skrf
import inspect
from typing import Callable, Any, Dict, get_args, get_origin, Union, Tuple, List, final
from types import GenericAlias, UnionType
import dataclasses

from pmrf.numpy import USE_JAX
if USE_JAX:
    import jax
import equinox as eqx

import pmrf.numpy as np
from pmrf._misc import field, tree_flatten_one_level_with_path
from pmrf._math import a2s, s2a
from pmrf._frequency import Frequency
from pmrf._pytree import tree_with_params, tree_params

PRIMARY_PROPERTIES = ['s', 'a']

jax.config.update("jax_enable_x64", True)

class Model(eqx.Module):
    """Base class representing an RF network that is computable, referred to in **paramrf** as a `Model`.

    This is an abstract class and should not be instantiated directly.

    Model initializers accept their parameters and sub-networks as input arguments, as well as general keyword arguments.
    Then, they can be used to calculate their properties as function of frequency (S-matrix, ABCD-matrix etc.)
    as well as a configurable "feature" matrix, with is their output when called as a function.
    
    Since all models derived from `dataclass`, arguments propagate to dervied classes.
    Therefore, the following arguments apply to sub-classes by default:

    Args:
        name (str, optional): A name associated with the model instance.
    """
    # Instance fields
    _z0: np.ndarray = field(default=50.0+0j, init=False, static=True)
    name: str | None = field(default=None, kw_only=True, static=True)

    # Class fields
    s_def: str | None = field(default='power', init=False, static=True, repr=False)
    priority: tuple = field(default=(), init=False, static=True, repr=False)
    dynamic: tuple = field(init=False, static=True, repr=False)

    def __init_subclass__(cls, dynamic: tuple | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.dynamic = dynamic = dynamic or (float, np.ndarray)
        for dynamic_type in dynamic:
            if issubclass(dynamic_type, Model):
                raise Exception("Error: do not set `Model` types as dynamic")

        # Add metadata and field properties to certain sub-class fields since we have certains constraints for the API.
        # Currently, we add default, default_factory, converter, and kw_only where necessary
        for field_name, field_types in cls.__annotations__.items():
            field_type = get_underlying_types(field_types)
            if field_type is None:
                return
            
            # We populate the field kwargs dynamically
            field_kwargs = {}

            # First, populate the default.
            default = getattr(cls, field_name, None)
            if not default is None:
                # We don't automatically assigned a field if the user has already
                if isinstance(default, dataclasses.Field):
                    continue

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

            # Finally, create the field and replace the class's value (but only if we need to - no need if kwargs is ultimately empty)
            if len(field_kwargs) != 0:
                setattr(cls, field_name, field(**field_kwargs))

    @final
    def __post_init__(self):
        self.build()
        self.post()

    def build(self) -> Any:
        pass

    def post(self):
        pass

    def __new__(cls, *args, **kwargs):
        return eqx.Module.__new__(cls)
    
    def __pow__(self, other: 'Model') -> 'Model':
        from pmrf.models.structural import CascadedModel
        return CascadedModel([self, other])
    
    @property
    def dynamic_filter(self) -> Callable[[Any], bool]:
        return lambda element: any([isinstance(element, dynamic) for dynamic in self.dynamic])
    
    @property
    def static_filter(self) -> Callable[[Any], bool]:
        return lambda element: not any([isinstance(element, dynamic) for dynamic in self.dynamic])
    
    @property
    def nested_submodels(self) -> list['Model']:
        return [node for node in jax.tree.flatten(self,)[0] if isinstance(node, Model)]
    
    @property
    def num_nested_submodels(self) -> int:
        return len(self.nested_submodels)    
    
    @property
    def submodels(self) -> list['Model']:
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Model)]
    
    @property
    def submodels_with_paths(self) -> list['Model']:
        return [path_val for path_val in tree_flatten_one_level_with_path(self)[0] if isinstance(path_val[1], Model)]
    
    @property
    def num_submodels(self):
        return len(self.submodels)    
        
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
        freq = Frequency(1, 1, 1)
        sf = lambda: self.s(freq)
        # return sf().shape[1]
        return jax.eval_shape(sf).shape[1]
    
    @property
    def nports(self):
        return self.number_of_ports
    
    @property
    def port_tuples(self) -> list[tuple[int, int]]:
        """
        Returns a list of tuples, for each port index pair.

        A convenience function for the common task for iterating over
        all s-parameters index pairs.

        This just calls::

            [(y,x) for x in range(self.nports) for y in range(self.nports)]


        Returns
        -------
        ports_ind : list of tuples
            list of all port index tuples.

        Examples
        --------
        >>> ntwk = skrf.data.ring_slot
        >>> for (idx_i, idx_j) in ntwk.port_tuples: print(idx_i, idx_j)

        """
        return [(y, x) for x in range(self.nports) for y in range(self.nports)]    
    
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

def get_underlying_types(tp: type) -> type | None:
    """
    Recursively gets the origin of a type annotation until a type that
    can be used in issubclass is found.

    This function handles generic aliases (like list[int]), unions,
    and type aliases.

    Args:
        tp: The type annotation.

    Returns:
        The underlying, non-generic type that can be used with issubclass,
        or None if no such type can be determined (e.g., for TypeVar).
    """
    # The annotations could be unions - in this case we just take the first one TODO upgrade this to do more in-depth inspection?

    if isinstance(tp, (type,)) and not isinstance(tp, (GenericAlias, UnionType)):
        return tp

    if isinstance(tp, UnionType):
        return None

    origin = get_origin(tp)

    if origin is None:
        return None

    if origin is Union:
        return None

    # Recursively call to handle nested generics like list[list[int]]
    return get_underlying_types(origin)

def model_check(model: Model) -> None:
    all_nodes = {}
    _model_check(model, all_nodes, model.dynamic_filter)

_leaf_treedef = jax.tree.structure(0)
def _model_check(node, all_nodes: dict, is_dynamic: Callable = None):
    subnodes, treedef = eqx.tree_flatten_one_level(node)

    # We allow duplicate leaves, empty containers
    if treedef == _leaf_treedef or treedef.num_leaves == 0:
        return
    
    # We allow duplications for non-dynamic, non-model types
    dynamic = is_dynamic(node) if is_dynamic is not None else True
    if not isinstance(node, Model) and not dynamic:
        return

    try:
        self_referential, type_string = all_nodes[id(node)]
    except KeyError:
        pass
    else:
        if self_referential:
            raise ValueError(
                f"Model node with value {node} is self-referential; that is "
                "to say it appears somewhere within its own PyTree structure. This "
                "is not allowed."
            )
        else:
            model_type = list(all_nodes.values())[0][1]
            if isinstance(node, Model):
                raise ValueError(
                    f"Sub-model with name '{node.name}' appears in model '{model_type}' multiple times. "
                    "If you would like to use multiple instances of a sub-model type in your model, explicitly create it each time."
                    "Otherwise, if you do want to share a sub-model across your model, create it with `shared=True`, "
                    "or pass `sharing=True` as an inheritance parameter in your model class declaration."
                )
            else:
                raise ValueError(
                    f"Model field with value {node} appears in the Model '{model_type}'"
                    "multiple times. This is almost always an error, as these nodes "
                    "will turn into two duplicate copies after "
                    "flattening/unflattening, e.g. when crossing a JIT boundary."
                )
    try:
        type_string = type(node).__name__
    except AttributeError:
        # AttributeError: in case we cannot get __name__ for some weird reason.
        type_string = "<unknown type>"
    all_nodes[id(node)] = (True, type_string)
    for subnode in subnodes:
        _model_check(subnode, all_nodes, is_dynamic)
    all_nodes[id(node)] = (False, type_string)
