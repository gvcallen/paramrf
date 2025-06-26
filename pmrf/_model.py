from functools import cached_property, partial
from copy import deepcopy
from typing import Callable, get_origin, get_args, Union, TypeVar
import inspect
import dataclasses
from dataclasses import dataclass, fields, is_dataclass
from types import GenericAlias, UnionType

import skrf as skrf
# import pmrf.numpy as np
# from pmrf.numpy import USE_JAX
# if USE_JAX:
import jax.numpy as np
import jax
import equinox as eqx
from jaxtyping import PyTree
from jax.tree_util import GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey

# import pmrf.functions.math as mf
# from pmrf.functions.math import complex_2_db, complex_2_db10
from pmrf.functions.parameters import a2s, s2a
from pmrf._frequency import Frequency
from pmrf.parameters import Parameter, is_param, asparam
from pmrf._misc import field
from pmrf._tree import flatten_one_level_with_path, nodes_by_type, nodes_by_type_with_path, partition, combine, value_at_path

PRIMARY_PROPERTIES = ('s', 'a')
FUNC_LOOKUP: dict[str, tuple[str, Callable | None]] = {
    're': ('Real Part', np.real),
    'im': ('Imag Part', np.imag),
    'mag': ('Magnitude', np.abs),
    # 'db': ('Magnitude (dB)', complex_2_db),
    # 'db10': ('Magnitude (dB)', complex_2_db10),
    'rad': ('Phase (rad)', np.angle),
    'deg': ('Phase (deg)', lambda x: np.angle(x, deg=True)),
    # 'arcl': ('Arc Length',lambda x: np.angle(x) * np.abs(x)),
    # 'rad_unwrap': ('Phase (rad)', lambda x: mf.unwrap_rad(np.angle(x))),
    # 'deg_unwrap': ('Phase (deg)', lambda x: mf.radian_2_degree(mf.unwrap_rad(np.angle(x)))),
    # 'arcl_unwrap': ('Arc Length', lambda x: mf.unwrap_rad(np.angle(x)) * np.abs(x)),
    # 'vswr': ('VSWR', lambda x: (1 + abs(x)) / (1 - abs(x))),
    # 'time': ('Time (real)', mf.ifft),
    # 'time_db': ('Magnitude (dB)',  lambda x: mf.complex_2_db(mf.ifft(x))),
    # 'time_mag': ('Magnitude', lambda x: mf.complex_2_magnitude(mf.ifft(x))),
}

ModelT = TypeVar('ModelT', bound='Model')

jax.config.update("jax_enable_x64", True)

class Model(eqx.Module):
    """
    **Overview**
    
    Base class representing an RF network that is computable,
    referred to in **paramrf** as a `Model`.

    This is the main abstract class, and should not be instantiated directly.
    
    To create a model, you should derive from the Model class, and
    override one of the primary functions e.g. 's', 'a' etc.
    The `Model` itself is a frozen python dataclass, and parameter definitions are
    done using the same dataclass format and typing. Since models are dataclasses,
    initializers accept any immediate parameters and sub-networks as input arguments.
    To change the model's parameters in a more flat manner, one can use the `with_params(...)`
    function to specify a dictionary of model parameters. To inspect which parameters a model supports,
    use `model.params` or `model.param_names`. If updating via fully flat array is desired,
    use `model.to_array(..)` with the resultant unravel function.
    
    Note that Model's internally derive from the `Equinox` `Module` class,
    and are immutable. This allows for the models to be used with `jax`
    (if enabled), with the whole model being treated as a pytree.
    To partition the model into a part that represents its core parameters,
    use e.g. `params, static = model.partition(..)`. To re-combine the model,
    use `pmrf.combine(params, static)`.
    
    See the `fitting` module for details on model fitting.

    **Example: pi-network**
    
    The following demonstrates the definition of a simple CLC network.
    We refer to these as "foundational" models, since they only consist of parameters
    (and not other models). Note that you can initialize parameters with any value
    that can be converted into a *numpy* or *jax* `ndarray`, and they will be converted
    after your dataclass `__init__` or `__post_init__` is called.
    
    ```python
    import jax.numpy as jnp # we use jax in this example
    import pmrf as prf
    from pmrf import Parameter
    
    class PiCLC(prf.Model):
        # These floats will be readily converted to `Parameter` objects, which operate like (and eagerly cast to) numpy arrays
        C1: Parameter = 1.0e-12
        L: Parameter = 1.0e-9
        C2: Parameter = 1.0e-12

        def a(self, freq: prf.Frequency) -> jnp.ndarray:
            # "freq" being passed in is very similar to skrf.Frequency, but can contain jax arrays
            C1, C2, L, w = self.C1, self.C2, self.L, freq.w
            Y1 = 1j * w * C1
            Y2 = 1j * w * C2
            Y3 = 1 / (1j * w * L)

            return jnp.array([
                [1 + Y2 / Y3,           1 / Y3      ],
                [Y1 + Y2 + Y1*Y2/Y3,    1 + Y1 / Y3 ],
            ]).transpose(2, 0, 1)  
    ```

    **Example: Series RLC**
    
    Here, we demonstrate the building of a "circuit" model that consists of a cascade of other models.
    We utilize some of the built-in foundational models available in `pmrf.models`, and also a more
    complicated initialization with `__init__`. Note that if you do not need any input model settings,
    then `__post_init__` should be prefered (see the python documentation on dataclasses for more information).
    
    Note that we mark fields that are derived from our components as `derived`.
    This is our way of specifying that our "core" model parameters are those *within* `res`, `ind` and `cap`.
        
    ```python
    import pmrf as prf
    from pmrf.models import Resistor, Capacitor, Inductor, Cascade
    
    class RLC(prf.Model):
        cascade: Cascade = field(derived=True)
        res: Resistor = Resistor(1.0)                   # note that, unlike with usual dataclasses, models can be constructed without "default_factory=..."
        ind: Inductor = Inductor(1.0e-9)
        cap: Capacitor = Capacitor(1.0e-12)
        
        def __init__(self, terminated=False):
            res, ind, cap = self.res, self.ind, self.cap
            if terminated:
                self.ind = self.ind.terminated()        # terminate in a short
            
            self.cascade = res ** ind ** cap            # similar syntax to scikit-rf

        def a(self, freq: prf.Frequency) -> np.ndarray:
            return self.cascade.a(freq)                 # just return cascade's abcd implementation
    ```
    
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
            # Clone any defaults that are either Models, Parameters, Python built-ins, or numpy arrays
            if hasattr(cls, field_name):
                default = getattr(cls, field_name)
                new_default = None
                if isinstance(default, list) or isinstance(default, dict) or isinstance(default, tuple):
                    new_default = deepcopy(default)    
                elif isinstance(default, Parameter) or isinstance(default, Model):
                    new_default = deepcopy(default)
                    if isinstance(default, Model) and default.name is None:
                        new_default = dataclasses.replace(new_default, name=field_name)
                elif isinstance(default, np.ndarray):
                    new_default = default.copy()            
                if new_default is not None:
                    setattr(cls, field_name, new_default)
                    
            # Allow auto-conversion of Parameter-annotated structures
            field_type = _get_first_underlying_type(field_types)
            if field_type is not None and issubclass(field_type, Parameter):
                default = getattr(cls, field_name, None)
                if default is not None:
                    param = asparam(default, name=field_name)
                    setattr(cls, field_name, eqx.field(default=param, converter=partial(asparam, name=field_name)))
                    
        # Implement dynamic functions
        for prop in PRIMARY_PROPERTIES:
            for prefix, lookup in FUNC_LOOKUP.items():
                func_name = f"{prop}_{prefix}"
                func = lookup[1]

                def make_dynamic_method(prop, func):
                    def dynamic_method(self, *args, **kwargs):
                        prop_fn = getattr(self, prop)
                        matrix = prop_fn(*args, **kwargs)
                        return func(matrix)
                    return dynamic_method

                setattr(cls, func_name, make_dynamic_method(prop, func))

        # Initialize class parameters
        cls._s_def = s_def
        cls._priority = ()
        cls._separator = separator
        
    def _path_to_param_name(self, path) -> str | list[str]:
        fields = []
        for key in path:
            if isinstance(key, GetAttrKey) or isinstance(key, DictKey):
                fields.append(key.name)
            elif isinstance(key, SequenceKey) or isinstance(key, FlattenedIndexKey):
                fields.append(str(key.idx))
        return self._separator.join(fields)

    def __pow__(self, other: ModelT) -> ModelT:
        from pmrf.models.containers import Cascade
        return Cascade([self, other])
    
    @property
    def _has_a(self) -> bool:
        return _is_overridden(type(self), Model, 'a')
    
    @property
    def _has_s(self) -> bool:
        return _is_overridden(type(self), Model, 's')        
    
    @property
    def params(self) -> dict[str, Parameter]:
        """A dictionary of the core model parameters.
        
        Keys are returned as long parameter names, and values are
        `Parameter` structures as they currently are in the model.
        The dictionary is ordered represents the underlying order.

        Returns:
            dict[str, Parameter]: The parameter dictionary.
        """
        core_params_tree = jax.tree.map(lambda node, is_core: node if is_core else None, self, self.core_object_spec, is_leaf=is_param)
        path_and_params = jax.tree.flatten_with_path(core_params_tree, is_leaf=is_param)
        return {self._path_to_param_name(path): param for path, param in path_and_params[0]}
          
    @property
    def param_names(self) -> list[str]:
        """A list of the core model parameter names.
        
        Returns:
            dict[str, Parameter]: The parameter dictionary.
        """        
        return list(self.params.keys())
    
    def a(self, freq: Frequency) -> np.ndarray:
        """Calculates the abcd parameter matrix as a function of frequency.

        This is one of the primary property functions that derived classes may implemented.
        If not implemented, and at least one other primary function has been implemented,
        then conversion formulae are used dynamically to calculate the resultant matrix.

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
        then conversion formulae are used dynamically to calculate the resultant matrix.

        Args:
            freq (Frequency): Specifies the frequency to calculate the S-parameters at.

        Returns:
            np.ndarray: The resultant S matrix.
        """
        if not self._has_a:
            raise NotImplementedError(f"Error: model sub-classes currently *have* to implement the 's' or the 'a' function, but class {type(self)} has neither")
        
        a = self.a(freq)
        return a2s(a, self.z0)    
            
    @property
    def submodels(self) -> list[ModelT]:
        """Returns a list of immediate submodels.

        Returns:
            list[ModelT]: The submodels.
        """
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Model)]
    
    @property
    def submodels_with_paths(self) -> list[tuple[PyTree, ModelT]]:
        """Return a list of immedate submodels, as well as their jax paths.

        Returns:
            list[tuple[PyTree, ModelT]]: A list of path-submodels.
        """
        return [path_val for path_val in flatten_one_level_with_path(self)[0] if isinstance(path_val[1], Model)]
    
    @property
    def nested_submodels(self) -> list[ModelT]:
        """Returns a list of nested submodels.
        
        This contains all immediate sub-models, as well as their sub-models.

        Returns:
            list[ModelT]: A list of nested submodels.
        """
        return nodes_by_type(self, Model)[1:]
    
    @property
    def nested_submodels_with_paths(self) -> list[tuple[PyTree, ModelT]]:
        """Returns a list of nested submodels, as well as their jax paths.
        
        See `nested_submodels`.

        Returns:
            list[ModelT]: A list of nested path-submodels.
        """        
        return nodes_by_type_with_path(self, Model)[1:]
    
    @property
    def primary_function(self) -> Callable[[Frequency], np.ndarray]:
        """A callable of the primary function e.g. 's', 'a' etc.
        
        See `self.primary_property` for more details..

        Returns:
            Callable[[Frequency], np.ndarray]: The primary function.
        """
        return getattr(self, self.primary_property)
            
    @property
    def primary_property(self) -> str:
        """The primary property e.g. 's', 'a' etc.
        
        This property will be equal to the function that has been overriden
        by the derived class. If multiple functions are overriden,
        then the priority tuple specified upon model creation is used.

        Returns:
            str: A string for the primary property.
        """
        prioritized = self._priority
        unprioritized = tuple(p for p in PRIMARY_PROPERTIES if p not in self._priority)
        
        for property in prioritized:
            if _is_overridden(type(self), Model, property):
                return property
        for property in unprioritized:
            if _is_overridden(type(self), Model, property):
                return property
        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overriden, which are the only ones supported currently")

    @cached_property
    def number_of_ports(self) -> int:
        """The number of ports in the model

        Returns:
            int: The port count.
        """
        freq = Frequency(1, 2, 2)
        eval = jax.eval_shape(lambda: self.s(freq))
        return eval.shape[1]
    
    @property
    def nports(self) -> int:
        """The number of ports in the model

        Returns:
            int: The port count.
        """
        return self.number_of_ports
    
    @property
    def port_tuples(self) -> list[tuple[int, int]]:
        """Tuples for the ports
        
        This returns a list of tuple combinations for all ports.

        Returns:
            list[tuple[int, int]]: The port tuples.
        """
        return [(y, x) for x in range(self.nports) for y in range(self.nports)]    
    
    @property
    def z0(self) -> np.ndarray:
        """The internal characteristic impedance matrix.

        Returns:
            float: Z0.
        """
        return self._z0
           
    def flipped(self) -> ModelT:
        """Returns a version of the model with its ports flipped.

        Returns:
            Model: The resultant model.
        """
        from models.containers import Flipped
        return Flipped(self)
    
    def terminated(self, load: ModelT = None) -> ModelT:
        """Returns the model terminated in a one-port load.
        
        May only be called for two-port models.

        Args:
            load (Model, optional): The load to terminate in. Defaults to None, in which case a short is used.

        Returns:
            ModelT: The terminated model.
        """
        from pmrf.models.lumped import Short
        from pmrf.models.containers import Cascade
        load = load or Short()
        terminated_model = Cascade((self, load))
        return terminated_model    
    
    @property
    def param_value_spec(self) -> PyTree:
        """A Pytree filter for all Model parameter values.
        
        The resultant dataclass contains `True` for all parameter values in
        `Parameter` structures within the `Model`, and `False` otherwise.
        Useful for `jax` and `Equinox` tree and filtering operations.

        Returns:
            PyTree: The resultant filter tree.
        """
        def is_param_value(path, node):
            if len(path) == 0:
                return False
            is_array = eqx.is_inexact_array(node)
            is_in_parameter = path[-1].name == 'value'
            if is_array and not is_in_parameter:
                raise Exception(f"Error: found jax/numpy array outside of a Parameter at path ({path})")
            return is_array and path[-1].name == 'value'
        
        return jax.tree.map_with_path(is_param_value, self, is_leaf=lambda node: eqx.is_inexact_array(node))
        
    @property
    def param_object_spec(self) -> PyTree:
        """A Pytree filter for all Model `Parameter` objects.
        
        The resultant dataclass contains `True` for in place of all
        `Parameter` structures within the `Model`, and `False` otherwise.
        Useful for `jax` and `Equinox` tree and filtering operations.

        Returns:
            PyTree: The resultant filter tree.
        """        
        return jax.tree.map(is_param, self, is_leaf=lambda node: is_param(node))
    
    @property
    def core_value_spec(self) -> PyTree:
        """A Pytree filter for all core Model parameter values.
        
        This filter is the same as `self.param_value_spec`,
        except excludes derived parameters.
        
        Returns:
            PyTree: The resultant filter tree.
        """        
        # A parameter is derived if any of its parent dataclasses have 'derived' set to True in its field metadata.
        path_is_derived = {}
        def is_leaf(path, node):
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
            return isinstance(node, bool)
        return jax.tree.map_with_path(lambda path, node: node and not path_is_derived[path], self.param_value_spec, is_leaf=is_leaf, is_leaf_takes_path=True)    
    
    @property
    def core_object_spec(self) -> PyTree:
        """A Pytree filter for all core Model `Parameter` objects.
        
        This filter is the same as `self.param_object_spec`,
        except excludes derived parameters.
        
        Returns:
            PyTree: The resultant filter tree.
        """           
        return jax.tree.map(lambda param, core_spec: is_param(param) and core_spec.value, self, self.core_value_spec, is_leaf=lambda node: is_param(node))

    @property
    def fit_value_spec(self) -> PyTree:
        """A Pytree filter for all fit Model parameter values.
        
        This filter is the same as `self.core_value_spec`,
        except excludes fixed parameters.
        
        Returns:
            PyTree: The resultant filter tree.
        """              
        def is_not_fixed(path, is_core):
            if not is_core:
                return False
            # Here we check that the array is within a parameter and that the parameter is varying
            param = value_at_path(self, path[0:-1])
            if not isinstance(param, Parameter):
                raise Exception(f"Found an array in a model not within a parameter: this is not allowed (at path {path})")
            return not param.fixed
        return jax.tree.map_with_path(is_not_fixed, self.core_value_spec)

    @property
    def fit_object_spec(self) -> PyTree:
        """A Pytree filter for all fit Model `Parameter` objects.
        
        This filter is the same as `self.core_object_spec`,
        except excludes fixed parameters.
        
        Returns:
            PyTree: The resultant filter tree.
        """                 
        return jax.tree.map(lambda param, fit_spec: is_param(param) and fit_spec.value, self, self.fit_value_spec, is_leaf=lambda node: is_param(node))
        
    def submodel_param_names(self, submodel_name: str | list[str]):
        raise Exception("Not yet working")
        submodel_names = submodel_name if isinstance(submodel_name, list) else [submodel_name]
        
        # First, generate a tree with the parameter names in place of the parameters.
        # We do so for the core parameters (the spec itself since we only have to check bools) and then recombine
        core_true, static = partition(self.param_object_spec, self.core_object_spec, self.param_object_spec)
        core_param_names_tree = jax.tree.map_with_path(lambda path, param: self._path_to_param_name(path), core_true)
        param_names_tree = combine(core_param_names_tree, static), self.param_value_spec

        # Then we remove names not in the specified submodel
        def none_if_not_in_submodel(path, node):
            in_submodel = len(path) > 0 and isinstance(path[0], GetAttrKey) and path[0].name in submodel_names
            return node if in_submodel else None

        names_unordered = list(dict.fromkeys(jax.tree.flatten(jax.tree.map_with_path(none_if_not_in_submodel, param_names_tree))[0]))
        return [name for name in self.param_names if name in names_unordered]    
    
    def partitioned(self, include_fixed=False, param_objects=False) -> tuple[ModelT, ModelT]:
        """Returns the model partitioned into parameters and a static part.
        
        This is useful for use with `jax` or `Equinox`, or for inspecting the model
        and its parameters.
        
        Note that, to combine the model again (e.g. after changing the parameters),
        note that `pmrf.combine(..)` and not `eqx.combiner(...)` should be used,
        since `pmrf` implements and extention the partitioning step that allows
        parameters to be referenced more than once, and then de-referenced on combining.

        Args:
            include_fixed (bool, optional): Whether or not to include fixed parameters in the first part. Defaults to `False`.
            param_objects (bool, optional): Whether or not to keep the whole `Parameter` object in the first part,
                                            or to filter out all non-`value` fields. Defaults to `False`.

        Returns:
            tuple[ModelT, ModelT]: The partitioned model.
        """
        if param_objects:
            shared_spec = self.param_object_spec
            if include_fixed:
                filter_spec = self.core_object_spec
            else:
                filter_spec = self.fit_object_spec
        else:
            shared_spec = self.param_value_spec
            if include_fixed:
                filter_spec = self.core_value_spec
            else:
                filter_spec = self.fit_value_spec
        return partition(self, filter_spec, shared_spec)
    
    def with_params(self, params: dict[str, Parameter] | dict[str, float] | None = None, **param_kwargs: dict[str, Parameter] | dict[str, float]) -> ModelT:
        """Returns a model the same type as `self`, but with core parameters updated from a dictionary.
        
        This is the most common way to initialize the parameters of a model.
        However, if you would like to populate the model with a flat array instead,
        convert it to an array using `self.to_array(..)` and use the resultant unravel function.

        Args:
            params (dict[str, Parameter] | dict[str, float] | None, optional): The parameter dictionary to updated from.
                                                                               Parameters can also be specified with key-word arguments.
                                                                               Defaults to `None`.

        Returns:
            ModelT: The model with the specific parameter changes.
        """
        params = params if params is not None else {}
        params.update(param_kwargs)
        
        # First, generate an ordered, input flat params array
        new_params = self.params
        
        # Validate the callers's input
        unknown_params = set(params.keys() - new_params.keys())
        if len(unknown_params) != 0:
            raise Exception(f"Error: unknown parameters {unknown_params} passed in")
        
        # Convert to an array of parameters instead of floats
        if all(isinstance(v, float) for v in params.values()):
            for name, value in params.items():
                # TODO create specs for the full parameter objects such that we can get and use the built-in scales
                new_params[name] = dataclasses.replace(new_params[name], value=np.array(value), scale=1.0)
        else:
            new_params.update(params)
        new_flat_params = list(new_params.values())
        
        # Get the current flat parameter object
        params_tree, static = partition(self, self.core_object_spec, self.param_object_spec, is_leaf=is_param)
        flat_params, treedef = jax.tree.flatten(params_tree, is_leaf=is_param)
        
        # We allow the caller to pass None for name and then we update the name. Otherwise names should match
        for i, param in enumerate(flat_params):
            if new_flat_params[i].name == None:
                new_flat_params[i] = dataclasses.replace(new_flat_params[i], name=param.name)
        
        # Create the update tree and return
        new_params_tree = jax.tree.unflatten(treedef, new_flat_params)
        return combine(new_params_tree, static, is_leaf=is_param)

    def to_array(self, fit=False, return_unravel_fn=False) -> np.ndarray | tuple[np.ndarray, Callable]:
        """Returns a raveled array of the model parameters.
        
        By default, all core parameters (including fixed ones)
        are returned, but retrieving only the non-fixed parameters
        can be done using `fit=True`.
        
        Args:
            fit (bool): Whether or not to return only the non-fixed (fit) parameters. Defaults to False.
            return_ravel_fn (bool): Whether or not to also return an unravel function, which can be called to re-create the `Model` from new parameters.

        Returns:
            np.ndarray: The resultant parameters, raveled into a 1D array.
        """
        filter_spec = self.fit_value_spec if fit else self.core_value_spec
        if return_unravel_fn:
            params, static = partition(self, filter_spec, self.param_value_spec)
            array, internal_fn = jax.flatten_util.ravel_pytree(params)
            unravel_fn = lambda arr: combine(internal_fn(arr), static)
            return array, unravel_fn
    
        params = eqx.filter(self, filter_spec)
        return jax.flatten_util.ravel_pytree(params)[0]
    
    def to_skrf(self, freq: Frequency | skrf.Frequency, **kwargs) -> skrf.Network:
        """Converts the model to a numpy array at the specified frequency.
        
        The internal primary property in `self.primary_property` is used for the conversion.

        Args:
            freq (pmrf.Frequency | skrf.Frequency): The frequency object.

        Returns:
            skrf.Network: The resultant skrf Network.
        """
        if isinstance(freq, Frequency):
            freq = freq.to_skrf()
        
        f, fname = self.primary_function, self.primary_property
        kwargs = kwargs or {}
        kwargs.update({
            fname: f(Frequency(frequency=freq)),
            'frequency': freq,
            'name': kwargs.get('name', self.name),
            'z0': self._z0,
        })

        return skrf.Network(**kwargs)    
    
def _is_overridden(cls, baseclass, method_name):
    result = False
    for cls in inspect.getmro(cls):
        if method_name in cls.__dict__:
            result = cls is not baseclass
            break
    return result

def _is_instance_of_annotated_type(instance, annotated_type) -> bool:
    origin = get_origin(annotated_type)
    args = get_args(annotated_type)

    if origin is UnionType:
        # Union or Optional
        return any(_is_instance_of_annotated_type(instance, arg) for arg in args)

    elif origin is not None:
        # Handles e.g. Annotated[T, ...], Literal[T], etc.
        return _is_instance_of_annotated_type(instance, args[0])

    else:
        return isinstance(instance, annotated_type)

def _get_first_underlying_type(tp: type) -> type | None:
    # The annotations could be unions - in this case we just take the first one TODO upgrade this to do more in-depth inspection?
    if isinstance(tp, UnionType):
        return _get_first_underlying_type(tp.__args__[0])
    if isinstance(tp, (type,)) and not isinstance(tp, (GenericAlias, UnionType)):
        return tp

    origin = get_origin(tp)
    if origin is None:
        return None
    if origin is Union:
        return None
    return _get_first_underlying_type(origin)