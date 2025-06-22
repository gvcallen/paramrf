from functools import cached_property
from copy import deepcopy
from typing import Callable, Literal

import skrf as skrf
import inspect
from typing import Callable, Any, Dict, get_origin, Union
from types import GenericAlias, UnionType
import dataclasses
from dataclasses import fields

import pmrf.numpy as np
from pmrf.numpy import USE_JAX
from pmrf.functions.math import complex_2_db
if USE_JAX:
    import jax
import equinox as eqx
from jaxtyping import PyTree

from pmrf._misc import field
from pmrf._frequency import Frequency
from pmrf._tree import with_params_from_dict, with_params_from_array, params_dict, params_array, flatten_one_level_with_path, nodes_by_type, nodes_by_type_with_path
import pmrf.functions.math as mf
from pmrf.functions.parameters import a2s, s2a

ComponentFuncT = Literal["re", "im", "mag", "db", "db10", "rad", "deg", "arcl", "rad_unwrap", "deg_unwrap",
                         "arcl_unwrap", "vswr", "time", "time_db", "time_mag", "time_impulse", "time_step"]

PRIMARY_PROPERTIES = ('s', 'a')
FUNC_LOOKUP: dict[ComponentFuncT, tuple[str, Callable | None]] = {
    're': ('Real Part', np.real),
    'im': ('Imag Part', np.imag),
    'mag': ('Magnitude', np.abs),
    # 'db': ('Magnitude (dB)', complex_2_db),
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
    _dynamic_types: tuple = field(init=False, repr=False, static=True)
    _separator: str = field(init=False, repr=False, static=True)

    def __init_subclass__(cls, dynamic_types: tuple = (float, np.ndarray), s_def: str = 'power', separator = '_', **kwargs):
        super().__init_subclass__(**kwargs)        

        for dynamic_type in dynamic_types:
            if issubclass(dynamic_type, Model):
                raise Exception("Error: do not set `Model` types as dynamic")

        # Add metadata and field properties to certain sub-class fields since we have certains constraints for the API.
        # Currently, we add default, default_factory, converter, and kw_only where necessary
        for field_name, field_types in cls.__annotations__.items():
            field_type = get_underlying_type(field_types)
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
                    field_kwargs['default_factory'] = lambda: deepcopy(default)
                else:
                    field_kwargs['default'] = default

            # Next, populate the Parameter converter for types considered dynamic (even those without defaults).
            if field_type in dynamic_types:
                field_kwargs['converter'] = lambda val: jax.numpy.asarray(val, dtype=float)

            # Finally, create the field and replace the class's value (but only if we need to - no need if kwargs is ultimately empty)
            if len(field_kwargs) != 0:
                setattr(cls, field_name, field(**field_kwargs))
                            
        cls._s_def = s_def
        cls._priority = ()
        cls._dynamic_types = dynamic_types
        cls._separator = separator

    def __new__(cls, *args, **kwargs):
        return eqx.Module.__new__(cls)
    
    def __pow__(self, other: 'Model') -> 'Model':
        from pmrf.models.containers import Cascaded
        return Cascaded([self, other])
    
    def copy(self) -> 'Model':
        return deepcopy(self)
        
    @cached_property
    def structure(self) -> Any:
        return jax.tree.structure(self)
    
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
    def param_filter(self) -> PyTree | Callable[[Any], bool]:
        dynamic = eqx.filter(self, eqx.is_inexact_array, replace=False)
        bool_tree = eqx.filter(dynamic, eqx.is_inexact_array, replace=True, inverse=True)
        
        is_core = []
        for field_info in fields(self):
            derived = field_info.metadata.get('derived', False)
            if derived:
                is_core.append(False)
            else:
                is_core.append(True)
        
        derived_fields = [field.name for field in fields(self) if self.__dataclass_fields__[field.name].metadata.get('derived', False)]
        bool_tree = eqx.tree_at(
            lambda m: [getattr(m, name) for name in derived_fields],
            bool_tree,
            [False] * len(derived_fields)
        )
        
        return bool_tree
    
    @cached_property
    def param_names(self) -> list[str]:
        return list(self.params.keys())
    
    @cached_property
    def params(self) -> Dict[str, Any] | np.ndarray:
        return params_dict(self, separator=self._separator, param_filter=self.param_filter)
    
    @cached_property
    def flat_params(self) -> np.ndarray:
        return params_array(self, self.param_filter)
        
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
    
    def __getattribute__ (self, name: str) -> Callable[..., Any]:
        # We are only interested in attributes that follow the pattern 's_<suffix>'
        found = False
        for p in PRIMARY_PROPERTIES:
            if name.startswith(f'{p}_'):
                found = True
                break
            
        if not found:
            return super().__getattribute__ (name)

        param, suffix = name[0], name[2:]
        if suffix in FUNC_LOOKUP:
            _description, processing_func = FUNC_LOOKUP[suffix]

            # Handle cases where the function is explicitly None (not implemented)
            if processing_func is None:
                def not_implemented_func(*args, **kwargs):
                    raise NotImplementedError(f"The function for '{name}' is not yet implemented.")
                return not_implemented_func

            def dynamic_method(*args, **kwargs):
                matrix = getattr(self, param, *args, **kwargs)
                return processing_func(matrix)
            return dynamic_method
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"Unknown S-parameter format: '{suffix}'"
            )    
    
    def dynamic(self) -> 'Model':
        return eqx.filter(self, self.param_filter)
    
    def static(self) -> 'Model':
        return eqx.filter(self, self.param_filter, inverse=True)
    
    def partitioned(self) -> 'Model':
        return eqx.partition(self, self.param_filter)    
           
    def flipped(self) -> 'Model':
        from models.containers import Flipped
        return Flipped(self)
    
    def terminated(self, load: 'Model' = None) -> 'Model':
        from pmrf.models.lumped import Short
        from pmrf.models.containers import Cascaded
        
        load = load or Short()
        terminated_model = Cascaded((self, load))
        return terminated_model
    
    def with_params(
        self,
        **params: Any
    ) -> "Model":
        return with_params_from_dict(self, separator=self._separator, subtree_separator=self._separator, array_separator=self._separator, index_separator=self._separator, param_filter=self.param_filter, **params)

    def with_flat_params(
        self,
        params: np.ndarray
    ) -> "Model":
        return with_params_from_array(self, params=params, param_filter=self.param_filter)
    
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

def model_check(model: Model) -> None:
    all_nodes = {}
    _model_check(model, all_nodes, model.dynamic)

_leaf_treedef = jax.tree.structure(0)
def _model_check(node, all_nodes: dict):
    subnodes, treedef = eqx.tree_flatten_one_level(node)

    # We allow duplicate leaves, empty containers
    if treedef == _leaf_treedef or treedef.num_leaves == 0:
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
        _model_check(subnode, all_nodes)
    all_nodes[id(node)] = (False, type_string)
