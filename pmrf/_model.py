from functools import cached_property
from copy import deepcopy
from typing import Callable, Literal, TypeVar

import skrf as skrf
import inspect
from typing import Callable, Any, Dict, get_origin, Union
from types import GenericAlias, UnionType
import dataclasses
from dataclasses import fields, is_dataclass
from jax.tree_util import GetAttrKey

import pmrf.numpy as np
from pmrf.numpy import USE_JAX
from pmrf.functions.math import complex_2_db
if USE_JAX:
    import jax
import equinox as eqx
from jaxtyping import PyTree

from pmrf._misc import field
from pmrf._frequency import Frequency
from pmrf._tree import with_params_from_dict, with_params_from_array, params_dict, params_array, flatten_one_level_with_path, nodes_by_type, nodes_by_type_with_path, partition, combine, param_names_tree, dealias, restore
import pmrf.functions.math as mf
from pmrf.functions.parameters import a2s, s2a

ComponentFuncT = Literal["re", "im", "mag", "db", "db10", "rad", "deg", "arcl", "rad_unwrap", "deg_unwrap",
                         "arcl_unwrap", "vswr", "time", "time_db", "time_mag", "time_impulse", "time_step"]

PRIMARY_PROPERTIES = ('s', 'a')
FUNC_LOOKUP: dict[ComponentFuncT, tuple[str, Callable | None]] = {
    're': ('Real Part', np.real),
    'im': ('Imag Part', np.imag),
    'mag': ('Magnitude', np.abs),
    'db': ('Magnitude (dB)', complex_2_db),
    # 'db10': ('Magnitude (dB)', complex_2_db10),
    'rad': ('Phase (rad)', np.angle),
    'deg': ('Phase (deg)', lambda x: np.angle(x, deg=True)),
    'arcl': ('Arc Length',lambda x: np.angle(x) * np.abs(x)),
    # 'rad_unwrap': ('Phase (rad)', lambda x: unwrap_rad(np.angle(x))),
    # 'deg_unwrap': ('Phase (deg)', lambda x: radian_2_degree(unwrap_rad(np.angle(x)))),
    # 'arcl_unwrap': ('Arc Length', lambda x: unwrap_rad(np.angle(x)) * np.abs(x)),
    'vswr': ('VSWR', lambda x: (1 + abs(x)) / (1 - abs(x))),
    # 'time': ('Time (real)', mf.ifft),
    # 'time_db': ('Magnitude (dB)',  lambda x: mf.complex_2_db(mf.ifft(x))),
    # 'time_mag': ('Magnitude', lambda x: mf.complex_2_magnitude(mf.ifft(x))),
    # 'time_impulse': ('Magnitude', None),
    # 'time_step': ('Magnitude', None),
}

ModelT = TypeVar('ModelT', bound='Model')

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
    name: str | None = field(default=None, kw_only=True, static=True)
    aliases: dict[str, str] | list[str] | None = field(default=None, kw_only=True, static=True)
    _z0: np.ndarray = field(default=50.0+0j, init=False, static=True)

    # Class fields
    _s_def: str = field(init=False, repr=False, static=True)
    _priority: tuple = field(init=False, repr=False, static=True)
    _dynamic_types: tuple = field(init=False, repr=False, static=True)
    _separator: str = field(init=False, repr=False, static=True)    

    def __init_subclass__(cls, dynamic_types: tuple = (float, np.ndarray), s_def: str = 'power', separator = '_', **kwargs):
        super().__init_subclass__(**kwargs)        

        for dynamic_type in dynamic_types:
            if issubclass(dynamic_type, Model):
                raise Exception("Error: do not set `Model` types as dynamic")

        # Add metadata and field properties to certain sub-class fields since we have certains constraints for the API.
        # Currently, we add default, default_factory, converter, and kw_only where necessary
        found_model, found_dynamic = False, False
        for field_name, field_types in cls.__annotations__.items():
            field_type = get_underlying_type(field_types)
            if field_type is None:
                continue

            if field_type in dynamic_types:
                found_dynamic = True
            if issubclass(field_type, Model):
                found_model = True
            
            # We populate the field kwargs dynamically
            field_kwargs = {}

            # First, populate the default.
            default = getattr(cls, field_name, None)
            if not default is None:
                # We don't automatically assigned a field if the user has already
                if isinstance(default, dataclasses.Field):
                    continue

                # We use `default.__class__.__hash__` to guess if the type is mutable, or if its a Model (because users should share models explicitly)
                if default.__class__.__hash__ is None or isinstance(default, Model):
                    default = deepcopy(default)
                field_kwargs['default'] = default

            # Next, populate the Parameter converter for types considered dynamic (even those without defaults).
            if field_type in dynamic_types:
                field_kwargs['converter'] = lambda val: jax.numpy.asarray(val, dtype=np.float64)

            # Finally, create the field and replace the class's value (but only if we need to - no need if kwargs is ultimately empty)
            if len(field_kwargs) != 0:
                setattr(cls, field_name, field(**field_kwargs))

        if found_model and found_dynamic:
            # TODO We cannot support this currently because the user expects to be able to use parameters in __post_init__
            # but the references will be messed up because float have not yet been converted etc.
            # We will have to enforce them creating np.ndarray 's but still that won't work right now because of mutability, I think
            raise Exception("Error: currently compound models with parameters are not supported. To build such model, first wrap your parameters in a sub-model.")
                            
        cls._s_def = s_def
        cls._priority = ()
        cls._dynamic_types = dynamic_types
        cls._separator = separator

    def __new__(cls, *args, **kwargs):
        return eqx.Module.__new__(cls)
    
    def __pow__(self, other: 'Model') -> 'Model':
        from pmrf.models.containers import Cascade
        return Cascade([self, other])
    
    def copy(self) -> 'Model':
        return deepcopy(self)
        
    @cached_property
    def structure(self) -> Any:
        return jax.tree.structure(self)
    
    @cached_property
    def filter_function(self) -> Callable[[Any], bool]:
        return eqx.is_inexact_array    
    
    @cached_property
    def shared_spec(self) -> PyTree:
        filter_fn = self.filter_function
        return jax.tree.map(lambda node: filter_fn(node), self)        
    
    @cached_property
    def param_spec(self) -> PyTree:
        # We create a spec that has False for derived and True for non-derived parameters.
        # A parameter is derived if any of its parent dataclasses have 'derived' set to True in its field metadata.
        path_is_derived = {}
        def is_leaf(path, node):
            # If a dataclass, populate path_is_derived for all children that are derived
            if is_dataclass(node):
                for field in fields(node):
                    if field.metadata.get('derived', False):
                        field_path = path + (GetAttrKey(field.name),)
                        path_is_derived[field_path] = True
            
            # Set base path as not being derived, and this path's derived as equal to the parents if not already set
            if len(path) == 0:
                path_is_derived[path] = False
            else:
                path_is_derived.setdefault(path, path_is_derived[path[0:-1]])
            
            if isinstance(node, bool):
                return True
            else:
                return False
        
        def is_core(path, node):
            return node and not path_is_derived[path]
        
        return jax.tree.map_with_path(is_core, self.shared_spec, is_leaf=is_leaf, is_leaf_takes_path=True)    
    
    @cached_property
    def nested_submodels(self) -> list['Model']:
        return nodes_by_type(self, Model)[1:]
    
    @cached_property
    def nested_submodels_with_paths(self) -> list[tuple[PyTree, 'Model']]:
        return nodes_by_type_with_path(self, Model)[1:]
    
    @cached_property
    def num_nested_submodels(self) -> int:
        return len(self.nested_submodels)    
    
    @cached_property
    def submodels(self) -> list['Model']:
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Model)]
    
    @cached_property
    def submodels_with_paths(self) -> list[tuple[PyTree, 'Model']]:
        return [path_val for path_val in flatten_one_level_with_path(self)[0] if isinstance(path_val[1], Model)]
    
    @cached_property
    def num_submodels(self):
        return len(self.submodels)
    
    @cached_property
    def param_names_tree(self):
        params, static = partition(self, self.param_spec, self.shared_spec)
        params = param_names_tree(params, self._separator)
        return combine(params, static)

    def submodel_param_names(self, submodel_name: str | list[str]):
        submodel_names = submodel_name if isinstance(submodel_name, list) else [submodel_name]
        param_names_tree = self.param_names_tree

        def none_if_not_in_submodel(path, node):
            in_submodel = len(path) > 0 and isinstance(path[0], GetAttrKey) and path[0].name in submodel_names
            return node if in_submodel else None

        names_unordered = list(dict.fromkeys(jax.tree.flatten(jax.tree.map_with_path(none_if_not_in_submodel, param_names_tree))[0]))
        return [name for name in self.param_names if name in names_unordered]

    @cached_property
    def params(self) -> Dict[str, Any]:
        param_tree = eqx.filter(self, self.param_spec)
        return params_dict(param_tree, separator=self._separator, param_aliases=self.aliases)
    
    @cached_property
    def params_array(self) -> np.ndarray:
        param_tree = eqx.filter(self, self.param_spec)
        return params_array(param_tree)
    
    @cached_property
    def param_names(self) -> list[str]:
        return list(self.params.keys())
    
    @cached_property
    def param_names_tree(self) -> PyTree:
        shared, _ = eqx.partition(self, self.shared_spec)
        core, ref = dealias(shared, self.param_spec)
        core_names = param_names_tree(core, self._separator)
        ref_names = restore(ref, core_names)
        is_leaf = lambda node: node is None or isinstance(node, str) or isinstance(node, list)
        return jax.tree.map(lambda x, y: x or y, ref_names, core_names, is_leaf=is_leaf)    
        
    @cached_property
    def primary_function(self) -> Callable[[Frequency], np.ndarray]:
        return getattr(self, self.primary_property)
            
    @cached_property
    def primary_property(self) -> str:
        prioritized = self._priority
        unprioritized = tuple(p for p in PRIMARY_PROPERTIES if p not in self._priority)
        
        for property in prioritized:
            if is_overridden(type(self), Model, property):
                return property
        for property in unprioritized:
            if is_overridden(type(self), Model, property):
                return property
        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overrided, which are the only ones supported currently")

    @cached_property
    def number_of_ports(self):
        freq = Frequency(1, 2, 2)
        eval = jax.eval_shape(lambda: self.s(freq))
        return eval.shape[1]
    
    @cached_property
    def _has_a(self) -> bool:
        return is_overridden(type(self), Model, 'a')
    
    @cached_property
    def _has_s(self) -> bool:
        return is_overridden(type(self), Model, 's')    
    
    @property
    def nports(self):
        return self.number_of_ports
    
    @property
    def port_tuples(self) -> list[tuple[int, int]]:
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
        if not self._has_s:
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
        if not self._has_a:
            raise NotImplementedError(f"Error: model sub-classes currently *have* to implement the 's' or the 'a' function, but class {type(self)} has neither")
        
        a = self.a(freq)
        return a2s(a, self.z0)
    
    def __getattr__(self, name: str) -> Callable[..., Any]:
        found = False
        for p in PRIMARY_PROPERTIES:
            if name.startswith(f'{p}_'):
                found = True
                break
            
        if not found:
            return super().__getattr__(name)

        param, suffix = name[0], name[2:]
        if suffix in FUNC_LOOKUP:
            cache_name = f"__cached_{name}"
            if cache_name in self.__dict__:
                return self.__dict__[cache_name]

            _description, processing_func = FUNC_LOOKUP[suffix]

            # Handle cases where the function is explicitly None (not implemented)
            if processing_func is None:
                def not_implemented_func(*args, **kwargs):
                    raise NotImplementedError(f"The function for '{name}' is not yet implemented.")
                return not_implemented_func

            def dynamic_method(*args, **kwargs):
                fn = getattr(self, param)
                matrix = fn(*args, **kwargs)
                return processing_func(matrix)
            
            self.__dict__[cache_name] = dynamic_method
            return dynamic_method
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"Unknown S-parameter format: '{suffix}'"
            )        
           
    def flipped(self) -> 'Model':
        from models.containers import Flipped
        return Flipped(self)
    
    def terminated(self, load: 'Model' = None) -> 'Model':
        from pmrf.models.lumped import Short
        from pmrf.models.containers import Cascade
        
        load = load or Short()
        terminated_model = Cascade((self, load))
        return terminated_model
    
    def with_params(
        self,
        flat_params: np.ndarray = None,
        **params: Any
    ) -> "Model":
        param_tree, static = partition(self, self.param_spec, self.shared_spec)
        if not flat_params is None:
            param_tree = with_params_from_array(param_tree, params=flat_params)
        else:
            param_tree = with_params_from_dict(param_tree, separator=self._separator, param_aliases=self.aliases, **params)
        return combine(param_tree, static)

    def to_skrf(self, freq: skrf.Frequency, **kwargs) -> skrf.Network:
        f, fname = self.primary_function, self.primary_property
        kwargs = kwargs or {}
        kwargs.update({
            fname: f(Frequency(frequency=freq)),
            'frequency': freq,
            'name': kwargs.get('name', self.name),
            'z0': self._z0,
        })

        return skrf.Network(**kwargs)
    
def is_overridden(cls, baseclass, method_name):
    result = False
    for cls in inspect.getmro(cls):
        if method_name in cls.__dict__:
            result = cls is not baseclass
            break
    return result

def get_underlying_type(tp: type) -> type | None:
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
    return get_underlying_type(origin)

# def model_check(model: Model) -> None:
#     all_nodes = {}
#     dynamic, _ = partition(model)
#     _model_check(model, all_nodes, dynamic)

# _leaf_treedef = jax.tree.structure(0)
# def _model_check(node, all_nodes: dict):
#     subnodes, treedef = eqx.tree_flatten_one_level(node)

#     # We allow duplicate leaves, empty containers
#     if treedef == _leaf_treedef or treedef.num_leaves == 0:
#         return

#     try:
#         self_referential, type_string = all_nodes[id(node)]
#     except KeyError:
#         pass
#     else:
#         if self_referential:
#             raise ValueError(
#                 f"Model node with value {node} is self-referential; that is "
#                 "to say it appears somewhere within its own PyTree structure. This "
#                 "is not allowed."
#             )
#         else:
#             model_type = list(all_nodes.values())[0][1]
#             if isinstance(node, Model):
#                 raise ValueError(
#                     f"Sub-model with name '{node.name}' appears in model '{model_type}' multiple times. "
#                     "If you would like to use multiple instances of a sub-model type in your model, explicitly create it each time."
#                     "Otherwise, if you do want to share a sub-model across your model, create it with `shared=True`, "
#                     "or pass `sharing=True` as an inheritance parameter in your model class declaration."
#                 )
#             else:
#                 raise ValueError(
#                     f"Model field with value {node} appears in the Model '{model_type}'"
#                     "multiple times. This is almost always an error, as these nodes "
#                     "will turn into two duplicate copies after "
#                     "flattening/unflattening, e.g. when crossing a JIT boundary."
#                 )
#     try:
#         type_string = type(node).__name__
#     except AttributeError:
#         # AttributeError: in case we cannot get __name__ for some weird reason.
#         type_string = "<unknown type>"
#     all_nodes[id(node)] = (True, type_string)
#     for subnode in subnodes:
#         _model_check(subnode, all_nodes)
#     all_nodes[id(node)] = (False, type_string)
