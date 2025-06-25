from functools import cached_property
from copy import deepcopy
from typing import Callable, Literal, TypeVar

import skrf as skrf
import inspect
from typing import Callable, Any, Dict, get_type_hints, get_origin, get_args, Union, TypeVar, Type
from types import GenericAlias, UnionType
import dataclasses
from dataclasses import fields, is_dataclass
from jax.tree_util import GetAttrKey

import pmrf.numpy as np
from numpy import ndindex
from pmrf.parameters import Parameter, ParameterSet, is_param, is_free_param, asparam
from pmrf.numpy import USE_JAX
from pmrf.functions.math import complex_2_db
if USE_JAX:
    import jax
import equinox as eqx
from jaxtyping import PyTree

from pmrf._misc import update_dict_with_alias
from pmrf._misc import field
from pmrf._frequency import Frequency
from pmrf._tree import flatten_one_level_with_path, nodes_by_type, nodes_by_type_with_path, partition, combine, value_at_path
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
    _z0: np.ndarray = field(default=50.0+0j, init=False, static=True)

    # Class fields
    _s_def: str = field(init=False, repr=False, static=True)
    _priority: tuple = field(init=False, repr=False, static=True)
    _separator: str = field(init=False, repr=False, static=True)    

    def __init_subclass__(cls, s_def: str = 'power', separator = '_', **kwargs):
        super().__init_subclass__(**kwargs)        

        for field_name, field_types in cls.__annotations__.items():
            # First, we clone any defaults that are either Models, Parameters, Python built-ins, or numpy arrays
            if hasattr(cls, field_name):
                default = getattr(cls, field_name)
                new_default = None
                if isinstance(default, list) or isinstance(default, dict) or isinstance(default, tuple):
                    new_default = deepcopy(default)    
                elif isinstance(default, Parameter) or isinstance(default, Model):
                    new_default = deepcopy(default)
                elif isinstance(default, np.ndarray):
                    new_default = default.copy()            
                if new_default is not None:
                    setattr(cls, field_name, new_default)
                    
            # Then, we allow auto-conversion of Parameter annotations structures
            field_type = get_first_underlying_type(field_types)
            if issubclass(field_type, Parameter):
                default = getattr(cls, field_name, None)
                if default is not None:
                    param = asparam(default, name=field_name)
                    setattr(cls, field_name, eqx.field(default=param, converter=asparam))                     

        # Then initialize our own parameters        
        cls._s_def = s_def
        cls._priority = ()
        cls._separator = separator

    def __pow__(self, other: ModelT) -> ModelT:
        from pmrf.models.containers import Cascade
        return Cascade([self, other])
    
    def copy(self) -> ModelT:
        return deepcopy(self)
        
    @property
    def structure(self) -> Any:
        return jax.tree.structure(self)
       
    @property
    def nested_submodels(self) -> list[ModelT]:
        return nodes_by_type(self, Model)[1:]
    
    @property
    def nested_submodels_with_paths(self) -> list[tuple[PyTree, ModelT]]:
        return nodes_by_type_with_path(self, Model)[1:]
    
    @property
    def num_nested_submodels(self) -> int:
        return len(self.nested_submodels)    
    
    @property
    def submodels(self) -> list[ModelT]:
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Model)]
    
    @property
    def submodels_with_paths(self) -> list[tuple[PyTree, ModelT]]:
        return [path_val for path_val in flatten_one_level_with_path(self)[0] if isinstance(path_val[1], Model)]
    
    @property
    def num_submodels(self):
        return len(self.submodels)
    
    @property
    def param_spec(self) -> PyTree:
        def is_param(path, node):
            if len(path) == 0:
                return False
            is_array = eqx.is_inexact_array(node)
            param = value_at_path(self, path[0:-1])
            if is_array and not isinstance(param, Parameter):
                raise Exception(f"Error: found jax/numpy array outside of a Parameter at path ({path})")
            return is_array and path[-1].name == 'value'
        
        return jax.tree.map_with_path(
            is_param,
            self,
            is_leaf=lambda node: eqx.is_inexact_array(node)
        )
    
    @property
    def core_param_spec(self) -> PyTree:
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
        
        return jax.tree.map_with_path(lambda path, node: node and not path_is_derived[path], self.param_spec, is_leaf=is_leaf, is_leaf_takes_path=True)    

    @property
    def fit_param_spec(self) -> PyTree:
        def is_varying_value(path, is_param, node):
            if not is_param or not eqx.is_inexact_array(node):
                return False
            
            # Here we check that the array is within a parameter and that the parameter is varying
            param = value_at_path(path[0:-1])
            if not isinstance(param, Parameter):
                raise Exception(f"Found an array in a model not within a parameter: this is not allowed (at path {path})")
            return not param.fixed and path[-1].name == 'value'
        
        return jax.tree.map_with_path(is_varying_value, self.core_param_spec, self)
    
    @property
    def core_params_tree(self) -> Any:
        return eqx.filter(self, self.core_param_spec)
    
    @property
    def params(self) -> Dict[str, Parameter]:
        paths_and_params = jax.tree.leaves_with_path(self.core_params_tree, is_leaf=lambda p: is_param(p) and not p.value is None)

        parameters = {}
        for path, param in paths_and_params:
            param_name = self._path_to_param_name(path, param)
            parameters[param_name] = param
        return parameters
    
    @property
    def param_set(self) -> ParameterSet:
        flat_params, _ = jax.tree.flatten(self.core_params_tree, is_leaf=lambda p: is_param(p) and not p.value is None)
        param_names = self.param_names
        for i, name in enumerate(param_names):
            flat_params[i] = dataclasses.replace(flat_params[i], long_name=name)
        return ParameterSet(flat_params)    
    
    @property
    def params_array(self) -> np.ndarray:
        flat_params, _ = jax.tree.flatten(self.param_tree)
        if not flat_params:
            return np.array([]) # Return empty array if no params
        
        return np.concatenate([p.ravel() for p in flat_params])
    
    @property
    def param_names(self) -> list[str]:
        return list(self.params.keys())    
        
    def submodel_param_names(self, submodel_name: str | list[str]):
        submodel_names = submodel_name if isinstance(submodel_name, list) else [submodel_name]
        param_names_tree = self.param_names_tree

        def none_if_not_in_submodel(path, node):
            in_submodel = len(path) > 0 and isinstance(path[0], GetAttrKey) and path[0].name in submodel_names
            return node if in_submodel else None

        names_unordered = list(dict.fromkeys(jax.tree.flatten(jax.tree.map_with_path(none_if_not_in_submodel, param_names_tree))[0]))
        return [name for name in self.param_names if name in names_unordered]    
    
    def _path_to_param_name(self, path, param: Parameter) -> str | list[str]:
        return self._separator.join(key.name for key in path if isinstance(key, GetAttrKey))
        
    @property
    def primary_function(self) -> Callable[[Frequency], np.ndarray]:
        return getattr(self, self.primary_property)
            
    @property
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
    
    @property
    def _has_a(self) -> bool:
        return is_overridden(type(self), Model, 'a')
    
    @property
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
            # try:
            return super().__getattr__(name)
            # except:
            #     raise Exception(f"Failed trying to get attribute {name} for class type {type(self)}")

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
           
    def flipped(self) -> ModelT:
        from models.containers import Flipped
        return Flipped(self)
    
    def terminated(self, load: ModelT = None) -> ModelT:
        from pmrf.models.lumped import Short
        from pmrf.models.containers import Cascade
        
        load = load or Short()
        terminated_model = Cascade((self, load))
        return terminated_model
    
    def partitioned(self) -> tuple[ModelT, ModelT]:
        return partition(self, self.core_param_spec, self.param_spec)
    
    def with_params(
        self,
        params: ParameterSet | dict[str, Parameter] | dict[str, float] | np.ndarray | None = None,
        **param_kwargs: dict[str, Parameter] | dict[str, float]
    ) -> ModelT:
        if isinstance(params, ParameterSet):
            return self._with_params_from_set(params=params)
        if not params is None and isinstance(params, np.ndarray):
            return self._with_params_from_array(params=params)
        else:
            params = params if params is not None else {}
            params.update(param_kwargs)
            return self._with_params_from_dict(params=params)

    def _with_params_from_set(
        self,
        params: ParameterSet,
    ) -> ModelT:
        param_tree, static = partition(self, self.core_param_spec, self.param_spec)
        flat_params, treedef = jax.tree.flatten(param_tree, is_leaf=lambda p: is_param(p) and not p.value is None)
        
        if len(params) != len(flat_params):
            raise Exception('Currently the full parameter set must be passed when initializing parameters')
        
        param_tree = jax.tree.unflatten(treedef, params)
        return combine(param_tree, static)
        
    def _with_params_from_dict(
        self,
        params: dict[str, Parameter] | dict[str, float],
    ) -> ModelT:
        # First, generate an ordered, input flat params array
        new_params = self.params
        
        if all(isinstance(v, float) for v in params.values()):
            for name, value in params.items():
                # TODO create specs for the full parameter objects such that we can get and use the built-in scales
                new_params[name] = dataclasses.replace(new_params[name], value=np.array(value), scale=1.0)
        new_flat_params = list(new_params.values())
        
        # Then, get the current flate parameters
        params_tree, static = partition(self, self.core_param_spec, self.param_spec)
        flat_params, treedef = jax.tree.flatten(params_tree, is_leaf=lambda p: isinstance(p, Parameter) and not p.value is None)
        
        # We allow the caller to pass None for name and then we update the name. Otherwise names should match
        for i, param in enumerate(flat_params):
            if new_flat_params[i].name == None:
                new_flat_params[i] = dataclasses.replace(new_flat_params[i], name=param.name)
        
        # Finally create the update tree and return
        new_params_tree = jax.tree.unflatten(treedef, new_flat_params)
        return combine(new_params_tree, static)        
    
    def _with_params_from_array(
        self,
        params: np.ndarray,
    ) -> ModelT:
        param_tree, static = partition(self, self.core_param_spec, self.param_spec)
        flat_leaves, treedef = jax.tree.flatten(param_tree)
        num_expected_params = sum(p.size for p in flat_leaves)
        
        # Ensure input is a JAX array for consistency
        params = np.asarray(params)

        if params.size != num_expected_params:
            raise ValueError(f"Input `flat_params` has size {params.size}, "
                                f"but model requires {num_expected_params}.")

        # Unflatten the leaves into a PyTree with the original structure.
        leaves = []
        offset = 0
        for leaf in flat_leaves:
            end = offset + leaf.size
            leaves.append(params[offset:end].reshape(leaf.shape))
            offset = end

        new_tree = jax.tree.unflatten(treedef, leaves)
        return combine(new_tree, static)

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

def is_instance_of_annotated_type(instance, annotated_type) -> bool:
    origin = get_origin(annotated_type)
    args = get_args(annotated_type)

    if origin is UnionType:
        # Union or Optional
        return any(is_instance_of_annotated_type(instance, arg) for arg in args)

    elif origin is not None:
        # Handles e.g. Annotated[T, ...], Literal[T], etc.
        return is_instance_of_annotated_type(instance, args[0])

    else:
        return isinstance(instance, annotated_type)

def get_first_underlying_type(tp: type) -> type | None:
    # The annotations could be unions - in this case we just take the first one TODO upgrade this to do more in-depth inspection?
    if isinstance(tp, UnionType):
        return get_first_underlying_type(tp.__args__[0])
    if isinstance(tp, (type,)) and not isinstance(tp, (GenericAlias, UnionType)):
        return tp

    origin = get_origin(tp)
    if origin is None:
        return None
    if origin is Union:
        return None
    return get_first_underlying_type(origin)