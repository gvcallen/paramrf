from abc import abstractmethod
from functools import cached_property
from copy import deepcopy, copy
from typing import Callable, Sequence, TypeVar
import dataclasses
from dataclasses import fields, is_dataclass

from typing import Sequence, Callable

import skrf as skrf
import numpy as np
import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import PyTree
from jax import flatten_util
from jax.tree_util import GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey
import equinox as eqx
from numpyro.distributions import Distribution

from pmrf._constants import PRIMARY_PROPERTIES, IndexArray, FeatureInputT, FeatureFunctionT, ModelParametersT, ModelT
from pmrf.functions.conversions import a2s, s2a
from pmrf.functions.math import FUNC_LOOKUP
from pmrf.parameters import Parameter, ParameterGroup, is_valid_param, asparam
from pmrf._frequency import Frequency
from pmrf._util import field, classproperty, is_overridden, get_first_underlying_type
from pmrf._tree import nodes_by_type, partition, combine, value_at_path

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
    function to specify a dictionary of model parameters. 
    To get the current parameters of a model instance, or just the names,
    use model.params() or model.param_names(). To get the names or values for the default configuration
    of the model, use model.DEFAULT_PARAMS or model.DEFAULT_PARAM_NAMES. These may be different
    to the instance parameters, if the model depends on some hyperparemeters, for example.
    If a flattened version of the parameters is desired, use model.flat_params().
    
    Note that Model's internally derive from the `Equinox` `Module` class,
    and are immutable. This allows for the models to be used with `jax` JIT compilation,
    with the whole model being treated as a pytree. For lower-level use, to partition the model into a part
    that represents its core parameters for use with e.g. jax-compatible optimizers,
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
            # "freq" being passed in is very similar to skrf.Frequency, but contains jax arrays
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
    Also note the "property" syntax below, which makes it easy to define derived models that
    can both be used in forward property functions (e.g. a(..) below), or as a regular property e.g. after fitting.
    
    ```python
    import pmrf as prf
    from pmrf.models import Resistor, Capacitor, Inductor, Cascade
    
    class RLC(prf.Model):
        # Note that, unlike with usual dataclasses, Model instances can be constructed without "default_factory=..." as below
        res: Resistor =                 Resistor(1.0)
        ind: Inductor | Cascade =       Inductor(1.0e-9)
        cap: Capacitor =                Capacitor(1.0e-12)
        
        @property
        # We define a cascade derived (note the familiar scikit-rf syntax)
        def cas(self) -> Cascade:       return self.res ** self.ind ** self.cap
        
        # And a one-line version of the above, but explicitly calling the Cascade constructor:
        # cas =                         property(lambda self: Cascade((self.res, self.ind, self.cap)))
        
        def __init__(self, terminated=False):
            res, ind, cap = self.res, self.ind, self.cap
            if terminated:
                # Terminate in a short, which creates a Cascade instance internally
                self.ind = self.ind.terminated()

        def a(self, freq: prf.Frequency) -> jnp.ndarray:
            # Just return the cascade's abcd implementation (we could also define 's' depending on the use case)
            return self.cas.a(freq)
    ```
    
    """
    # Instance fields
    name: str | None = field(default=None, kw_only=True, static=True)
    separator: str = field(default='_', kw_only=True, static=True)
    _z0: complex = field(default=50.0+0j, kw_only=True, static=True)
    _param_groups: list[ParameterGroup] = field(default_factory=lambda: list(), kw_only=True, repr=False, static=True)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)        

        # Pre-process fields by modifying any defaults as needed, as well as automatically applying converters
        for field_name, field_types in cls.__annotations__.items():
            # Setup
            field_kwargs = {}
            default = getattr(cls, field_name, None)
            if default is None or isinstance(default, dataclasses.Field):
                continue
                
            # Replace default model and parameter names with the field name
            if isinstance(default, Model) and default.name is None:
                default = dataclasses.replace(default, name=field_name)
            if isinstance(default, Parameter) and default.name is None:
                default = dataclasses.replace(default, name=field_name)

            # Auto apply asparam converter, to allow auto-conversion of Parameter-annotated structures
            field_type = get_first_underlying_type(field_types)
            if field_type is not None and issubclass(field_type, Parameter):
                if default is not None and not isinstance(default, Parameter):                
                    # Common mistake
                    if isinstance(default, tuple):
                        raise Exception(f"Expected a parameter for default '{field_name}' in class {cls} but found a tuple instead")
                    field_kwargs['default'] = default
                
                field_kwargs['converter'] = lambda x, field_name=field_name: asparam(x, name=field_name)
                # field_kwargs['default_factory'] = lambda default=default: default
            
            # Apply default_factory to avoid Python default mutable trap
            if any(isinstance(default, mutable_type) for mutable_type in {list, dict, tuple, Parameter, Model, jnp.ndarray}):
                field_kwargs['default_factory'] = lambda default=default: deepcopy(default)
            
            # Set final field
            if len(field_kwargs) != 0:
                setattr(cls, field_name, eqx.field(**field_kwargs))                
        
                    
        # Implement dynamic functions
        for prop in PRIMARY_PROPERTIES:
            for suffix, lookup in FUNC_LOOKUP.items():
                def make_dynamic_method(prop, func):
                    def dynamic_method(self, *args, **kwargs):
                        prop_fn = getattr(self, prop)
                        matrix = prop_fn(*args, **kwargs)
                        return func(matrix)
                    return dynamic_method
                
                # First the regular (non-indexed) function e.g. s_mag
                func_name = f"{prop}_{suffix}"
                func = lookup[1]
                setattr(cls, func_name, make_dynamic_method(prop, func))                
                
                # Then the index function function e.g. s_mn_mag
                func_name = f"{prop}_mn_{suffix}"
                func = lookup[1]
                setattr(cls, func_name, make_dynamic_method(f"{prop}_mn", func))                

    @property
    def _param_value_spec(self) -> PyTree:
        # A Pytree filter for *all* Model parameter values.
        def is_param_value(path, node):
            if len(path) == 0 or not eqx.is_inexact_array(node):
                return False
            return hasattr(path[-1], 'name') and path[-1].name == 'value'
        
        return jax.tree.map_with_path(is_param_value, self, is_leaf=lambda node: eqx.is_inexact_array(node))
        
    @property
    def _param_object_spec(self) -> PyTree:
        # A Pytree filter for all Model `Parameter` objects.
        return jax.tree.map(is_valid_param, self, is_leaf=lambda node: is_valid_param(node))
    
    @property
    def _core_value_spec(self) -> PyTree:
        # A Pytree filter for all core Model parameter values.
        # Temporarily no longer supporting derived parameters i.e. core parameters = parameters
        # return self._param_value_spec
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
        return jax.tree.map_with_path(lambda path, node: node and not path_is_derived[path], self._param_value_spec, is_leaf=is_leaf, is_leaf_takes_path=True)    
    
    @property
    def _core_object_spec(self) -> PyTree:
        # A Pytree filter for all core Model `Parameter` objects.
        return self._param_object_spec
        # return jax.tree.map(lambda param, core_spec: is_valid_param(param) and core_spec.value, self, self.core_value_spec, is_leaf=lambda node: is_valid_param(node))

    @property
    def _free_value_spec(self) -> PyTree:
        # A Pytree filter for all free Model parameter values.
        def is_not_fixed(path, is_core):
            if not is_core:
                return False
            # Here we check that the array is within a parameter and that the parameter is varying
            param = value_at_path(self, path[0:-1])
            if not isinstance(param, Parameter):
                raise Exception(f"Found an array in a model not within a parameter: this is not allowed (at path {path})")
            return not param.fixed
        return jax.tree.map_with_path(is_not_fixed, self._core_value_spec)

    @property
    def _free_object_spec(self) -> PyTree:
        # A Pytree filter for all free Model `Parameter` objects.
        return jax.tree.map(lambda param, fit_spec: is_valid_param(param) and fit_spec.value, self, self._free_value_spec, is_leaf=lambda node: is_valid_param(node))               
    
    def _path_to_param_name(self, path) -> str | list[str]:
        # Converts a path to its vector parameter name
        fields = []
        for key in path:
            if isinstance(key, GetAttrKey) or isinstance(key, DictKey):
                fields.append(key.name)
            elif isinstance(key, SequenceKey) or isinstance(key, FlattenedIndexKey):
                fields.append(str(key.idx))
        return self.separator.join(fields)

    def __pow__(self, other: ModelT) -> ModelT:
        from pmrf.models.containers import Cascade
        return Cascade([self, other])
    
    @classproperty
    def DEFAULT_PARAMS(cls) -> dict[str, Parameter]:
        """The default parameters for the model.

        Returns:
            dict[str, Parameter]: The parameters.
        """
        instance = cls()
        return instance.params()
    
    @classproperty
    def DEFAULT_PARAM_NAMES(cls) -> list[str]:
        """The default parameter names for the model.

        Returns:
            list[str]: The parameter names.
        """
        instance = cls()
        return instance.param_names()
    
    @property
    def primary_function(self) -> Callable[[Frequency], jnp.ndarray]:
        """A callable of the primary function e.g. 's', 'a' etc.
        
        See `self.primary_property` for more details..

        Returns:
            Callable[[Frequency], jnp.ndarray]: The primary function.
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
        prioritized = () # for future expansion
        unprioritized = tuple(p for p in PRIMARY_PROPERTIES if p not in prioritized)

        if is_overridden(type(self), Model, '__call__'):
            return self().primary_property
        
        for property in prioritized:
            if is_overridden(type(self), Model, property):
                return property
        for property in unprioritized:
            if is_overridden(type(self), Model, property):
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
    def z0(self) -> jnp.ndarray:
        """The internal characteristic impedance matrix.

        Returns:
            float: Z0.
        """
        return self._z0
    
    def __call__(self) -> 'Model':
        """Builds a compositional circuit model the represents this model.

        Returns:
            Model: The resultant model.
        """        
        raise NotImplementedError(f"Error: cannot use the __call__ method to build a model in the Model class directly")
    
    def a(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the abcd parameter matrix, to be derived by sub-classes.

        Args:
            freq (Frequency): Specifies the frequency to calculate the abcd-parameters at.
        Returns:
            jnp.ndarray: The resultant abcd matrix.
        """
        if is_overridden(type(self), Model, '__call__'):
            return self().a(freq)

        if not is_overridden(type(self), Model, 's'):
            raise NotImplementedError(f"Error: model sub-classes currently *have* to implement the 's' or the 'a' function, but class {type(self)} has neither")
        
        s = self.s(freq)
        return s2a(s, self.z0)
    
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Calculates the S parameter matrix, to be derived by sub-classes.

        Args:
            freq (Frequency): Specifies the frequency to calculate the abcd-parameters at.
        Returns:
            jnp.ndarray: The resultant abcd matrix.
        """
        if is_overridden(type(self), Model, '__call__'):
            return self().s(freq)

        if not is_overridden(type(self), Model, 'a'):
            raise NotImplementedError(f"Error: model sub-classes currently *have* to implement the 's' or the 'a' function, but class {type(self)} has neither")
        
        a = self.a(freq)
        return a2s(a, self.z0)
    
    def s_mn(self, freq: Frequency, m: IndexArray = None, n: IndexArray = None) -> jnp.ndarray:
        """Calculates the ABCD parameter matrix at specific ports.
        
        This can be overriden for performance reasons.

        Args:
            freq (Frequency): Specifies the frequency to calculate the S-parameters at.
            m (IndexArray): Specifies the first port indices, just as would be retrieved using `self.s(freq)[:,m,:]`. Defaults to `None` to specify a slice.
            n (IndexArray): Specifies the second port indices, just as would be retrieved using `self.s(freq)[:,:,m]`.  Defaults to `None` to specify a slice.
        Returns:
            jnp.ndarray: The resultant S matrix.
        """
        if m is None:
            if n is None:
                return self.a(freq)    
            return self.a(freq)[:, :, n]
        elif n is None:
            return self.a(freq)[:, m, :]

        return self.a(freq)[:, m, n]
    
    def s_mn(self, freq: Frequency, m: IndexArray = None, n: IndexArray = None) -> jnp.ndarray:
        """Calculates the S parameter matrix at specific ports.
        
        This can be overriden for performance reasons.

        Args:
            freq (Frequency): Specifies the frequency to calculate the S-parameters at.
            m (IndexArray): Specifies the first port indices, just as would be retrieved using `self.s(freq)[:,m,:]`. Defaults to `None` to specify a slice.
            n (IndexArray): Specifies the second port indices, just as would be retrieved using `self.s(freq)[:,:,m]`.  Defaults to `None` to specify a slice.
        Returns:
            jnp.ndarray: The resultant S matrix.
        """
        if m is None:
            if n is None:
                return self.s(freq)    
            return self.s(freq)[:, :, n]
        elif n is None:
            return self.s(freq)[:, m, :]

        return self.s(freq)[:, m, n]
            
    def children(self) -> list[ModelT]:
        """Returns a list of immediate submodels (children).

        Returns:
            list[ModelT]: The submodels.
        """
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Model)]
    
    def submodels(self) -> list[ModelT]:
        """Returns a list of all submodels.
        
        This contains all immediate sub-models, as well as their sub-models.

        Returns:
            list[ModelT]: A list of all nested submodels.
        """
        return nodes_by_type(self, Model)[1:]

    def flipped(self) -> ModelT:
        """Returns a version of the model with its ports flipped.

        Returns:
            Model: The resultant model.
        """
        from pmrf.models.containers import Flipped
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
    
    def copy(self: ModelT) -> ModelT:
        return deepcopy(self)
    
    def partition(self: ModelT, include_fixed=False, param_objects=False) -> tuple[ModelT, ModelT]:
        """Returns the model partitioned into parameters and a static part.
        
        This is useful for use with `jax` or `Equinox`, or for inspecting the model
        and its parameters.
        
        Note that, to combine the model again (e.g. after changing the parameters),
        note that either `Model.from_combined(..)` or `pmrf.combine(..)` should be used,
        as opposed to `eqx.combiner(...)` . This is due to the fact that `pmrf` implements
        an extention the partitioning step that allows parameters to be referenced more than once.

        Args:
            include_fixed (bool, optional): Whether or not to include fixed parameters in the first part. Defaults to `False`.
            param_objects (bool, optional): Whether or not to keep the whole `Parameter` object in the first part,
                                            or to filter out all non-`value` fields. Defaults to `False`.

        Returns:
            tuple[ModelT, ModelT]: The partitioned model.
        """
        if param_objects:
            shared_spec = self._param_object_spec
            if include_fixed:
                filter_spec = self._core_object_spec
            else:
                filter_spec = self._free_object_spec
        else:
            shared_spec = self._param_value_spec
            if include_fixed:
                filter_spec = self._core_value_spec
            else:
                filter_spec = self._free_value_spec
        return partition(self, filter_spec, shared_spec)   
    
    def params(self, include_fixed=False, flat=False, submodels: 'Model' | Sequence['Model'] | str | Sequence[str] | None = None) -> dict[str, Parameter]:
        """A dictionary of the core model parameters.
        
        Keys are returned as long parameter names, and values are the
        `Parameter` structures as they currently are in the model.
        The dictionary order matches the underlying, flattened array order.
        
        Args:
            include_fixed (bool):   Whether or not to also return fixed (fit) parameters. Defaults to `False`.
            flat (bool):            Flattens internal parameters such that all parameters are 1D.
                                    Only possible if all parameters are separable. In this case, the return `Parameter`
                                    objects are not guaranteed to refer to the same internal `Parameter` objects.
            submodels:              Specifies a submodel for which parameters should be returned.

        Returns:
            dict[str, Parameter]: The parameter dictionary.
        """
        spec = self._core_object_spec if include_fixed else self._free_object_spec
        params_tree = eqx.filter(self, spec, is_leaf=is_valid_param)
        path_and_params = jax.tree.flatten_with_path(params_tree, is_leaf=is_valid_param)
        
        
        _params: dict[str, Parameter] = {self._path_to_param_name(path): param for path, param in path_and_params[0]}          

        if flat:
            flat_params = {}
            for name, param in _params.items():
                if param.ndim > 1:
                    param_flattened = param.flatten(separator=self.separator)
                    for i, subparam in enumerate(param_flattened):
                        flat_params[f'{name}{self.separator}{i}'] = subparam
                else:
                    flat_params[name] = param
            _params = flat_params

        if submodels is not None:
            if isinstance(submodels, Model) or isinstance(submodels, str):
                submodels = [submodels]
            if len(submodels) != 0 and isinstance(submodels[0], str):
                submodels = [getattr(self, name) for name in submodels]
            submodel_param_values = [param for submodel in submodels for param in submodel.params(include_fixed=include_fixed, flat=flat).values()]
            _params = {k: v for k, v in _params.items() if any(v is p for p in submodel_param_values)}
            
        return _params
        
    def param_groups(self, include_fixed=False, flat=False) -> list[ParameterGroup]:
        """A list of all parameter groups in the model.
        
        This function is useful for fitters that need to take into account
        e.g. any constraints or correlated priors. Only groups that were
        create with the same `flat` flag are returned.
        
        Currently, only groups for the current model are considered
        (i.e. groups in children models are ignored).
        
        Args:
            include_fixed (bool):   Whether or not to also return fixed (fit) parameters. Defaults to `False`.
            flat (bool):            Flattens internal parameters such that all parameters are 1D.
                                    Only possible if all parameters are separable. In this case, the return `Parameter`
                                    objects are not guaranteed to refer to the same internal `Parameter` objects.

        Returns:
            dict[str, Parameter]: The parameter dictionary.
        """
        
        params = self.params(include_fixed=include_fixed, flat=flat)
        
        groups = [group for group in self._param_groups if group.flat == flat]

        grouped_param_names = {name for group in groups for name in group.params.keys()}
        for name, param in params.items():
            if name not in grouped_param_names:
                groups.append(ParameterGroup(params={name: param}, prior=param.prior, flat=flat))
        
        return groups
        
    def with_params(
        self: ModelT,
        params: dict[str, Parameter] | dict[str, float] | None = None,
        check_missing: bool = False,
        check_unknown: bool = False,
        fix_others = False,
        **param_kwargs: dict[str, Parameter] | dict[str, float],
    ) -> ModelT:
        """Returns a model the same type as `self`, but with core parameters updated from a dictionary.
        
        Args:
            params (dict[str, Parameter] | dict[str, float] | None, optional):      The parameter dictionary to updated from.
                                                                                    Parameters can also be specified with key-word arguments.
                                                                                    Defaults to `None`.
            fix_others (bool):                                                      Whether or not to fix any parameters in the model that were not passed. Defaults to `False`.
            check_missing (bool):                                                   Specifies to check that all model parameters are passed. Defaults to `False`.
            check_unknown (bool):                                                   Specifies to check that no unknown parameters were passed. Defaults to `False`.
                                                                               

        Returns:
            ModelT: The model with the specific parameter changes.
        """
        params = params if params is not None else {}
        params.update(param_kwargs)
        
        # First, generate an ordered, input flat params array
        new_params = self.params(include_fixed=True)
        
        # Validate the callers's input
        unknown_params = set(params.keys() - new_params.keys())
        if check_unknown and len(unknown_params) != 0:
            raise Exception(f"Error: the following parameters were passed but are not in the model: {unknown_params}")
        params = {k: v for k, v in params.items() if k not in unknown_params}
        
        if check_missing or fix_others:
            missing_params = set(new_params.keys() - params.keys())
            if check_missing and len(missing_params) != 0:
                raise Exception(f"Error: the following model parameters were missing: {missing_params}")
            if fix_others:
                for missing_param_name in missing_params:
                    new_params[missing_param_name] = dataclasses.replace(new_params[missing_param_name], fixed=True)
                        
        def is_convertible_to_float(x):
            try:
                float(x)
                return True
            except (ValueError, TypeError):
                return False

        # Convert to an array of parameters instead of floats
        if all(is_convertible_to_float(v) for v in params.values()):            
            for name, value in params.items():
                # TODO create specs for the full parameter objects such that we can get and use the built-in scales
                new_params[name] = dataclasses.replace(new_params[name], value=jnp.array(value), scale=1.0)
        else:
            new_params.update(params)
        new_flat_params = list(new_params.values())
        
        # Get the current flat parameter object
        params_tree, static = partition(self, self._core_object_spec, self._param_object_spec, is_leaf=is_valid_param)
        flat_params, treedef = jax.tree.flatten(params_tree, is_leaf=is_valid_param)
        
        # We allow the caller to pass None for name and then we update the name. Otherwise names should match
        for i, param in enumerate(flat_params):
            if new_flat_params[i].name == None:
                new_flat_params[i] = dataclasses.replace(new_flat_params[i], name=param.name)
        
        # Create the update tree and return
        new_params_tree = jax.tree.unflatten(treedef, new_flat_params)
        combined: Model = combine(new_params_tree, static, is_leaf=is_valid_param)
        return combined
    
    def with_param_groups(self: ModelT, param_groups: ParameterGroup | list[ParameterGroup], flat=False) -> ModelT:
        # Method 1
        # joint_names = ['length', 'zn', 'k1']
        # joint_prior = Distribution(...)
        # group_info = ParameterGroup(names=joint_names, prior=joint_prior)
        # 
        # model = ReceiverModel(sr_mtsj1=Semirigid().with_param_group(group_info))
        #
        # Parameter
        # self._parameter_groups
        # ParameterGroup
        if not isinstance(param_groups, list):
            param_groups = [param_groups]
        
        all_params = self.params(include_fixed=True, flat=flat)
        
        param_groups_old = self._param_groups.copy() if self._param_groups is not None else []
        for param_group_old in param_groups_old:
            if param_group_old.flat != flat:
                raise Exception('Cannot mix flat and non-flat parameter groups')
        
        param_groups_new = param_groups_old
        param_groups_new.extend(param_groups)
        param_groups_new = [param_group.resolve_params(all_params) for param_group in param_groups_new]
        param_groups_new = [dataclasses.replace(param_group_new, flat=flat) for param_group_new in param_groups_new]
        return dataclasses.replace(self, _param_groups=param_groups_new)
        
    
    def with_fixed_params(self: ModelT, *params, check_unknown=True) -> ModelT:
        """Returns a version of self with the specified parameters fixed.

        Args:
            params (str | Sequence[str]): The parameter names, specified as string arguments
            check_unknown (bool):           Specifies to check that no unknown parameters were passed. Defaults to `True`.

        Returns:
            ModelT: The new model with the fixed parameters.
        """
        if isinstance(params, str):
            params = [params]
        params = set(params)
            
        current_params = self.params()        
        current_param_names = set(current_params.keys())
        
        if check_unknown:
            for param_name in params:
                if param_name not in current_param_names:
                    raise Exception(f"Specified parameter '{param_name}'' not found in model")
        
        new_params = current_params.copy()
        for name, param in current_params.items():
            if name in params:
                new_params[name] = param.as_fixed()
        return self.with_params(new_params)
    
    def with_free_params(self: ModelT, *params, fix_others=True, check_unknown=True) -> ModelT:
        """Returns a version of self with the specified parameters free.

        Args:
            *params (str):                  The parameter names, specified as string arguments.
            fix_others (bool):              Specifies that other parameters not specified should be explicitly fixed. Defaults to `True`.
            check_unknown (bool):           Specifies to check that no unknown parameters were passed. Defaults to `True`.

        Returns:
            ModelT: The new model with the free parameters.
        """        
        if isinstance(params, str):
            params = [params]
        params = set(params)
        current_params = self.params(include_fixed=True)
        current_param_names = set(current_params.keys())
        
        if check_unknown:
            for param_name in params:
                if param_name not in current_param_names:
                    raise Exception(f"Specified parameter '{param_name}'' not found in model")
        
        new_params = current_params.copy()
        for name, param in current_params.items():
            if name in params:
                new_params[name] = param.as_free()
            elif fix_others:
                new_params[name] = param.as_fixed()
        return self.with_params(new_params)    
    
    def with_free_submodels(self: ModelT, free_submodels: 'Model' | Sequence['Model'] | str | Sequence[str]) -> ModelT:
        """Returns the current model with all parameters fixed except those in the specified submodels.

        The submodels can be any models that reference the parameters in this model.
        Specifically, `free_submodels` can consist of direct children of this model,
        submodels of those direct children, and any models that are built using the
        parameters of this model. Then, only the parameters that are found in both
        the submodels and this model are set to be free, and all others are fixed.

        Args:
            free_submodels (Model | Sequence[Model] | str | Sequence[str]):
                The submodels to set this model's free parameters by.
                If a string or sequence of strings is passed, `getattr`
                is simply called on `self` to retrieve the model instances.

        Returns:
            ModelT: A new model with the parameters not in `free_submodels` fixed.
        """
        if isinstance(free_submodels, Model) or isinstance(free_submodels, str):
            free_submodels = [free_submodels]
        
        if len(free_submodels) != 0 and isinstance(free_submodels[0], str):
            free_submodels = [getattr(self, name) for name in free_submodels]

        allowed_free_param_values = [param for source in free_submodels for param in source.params().values()]
        allowed_free_params = {k: v for k, v in self.params().items() if any(v is p for p in allowed_free_param_values)}

        return self.with_params(allowed_free_params, fix_others=True)          
    
    def to_skrf(self, freq: Frequency | skrf.Frequency, **kwargs) -> skrf.Network:
        """Converts the model to a numpy array at the specified frequency.
        
        The internal primary property in `self.primary_property` is used for the conversion.

        Args:
            freq (pmrf.Frequency | skrf.Frequency): The frequency object.

        Returns:
            skrf.Network: The resultant skrf Network.
        """
        if isinstance(freq, Frequency):
            model_freq = freq
            measured_freq = freq.to_skrf()
        else:
            model_freq = Frequency.from_skrf(freq)
            measured_freq = freq
        
        f, fname = self.primary_function, self.primary_property
        kwargs = kwargs or {}
        kwargs.update({
            fname: f(model_freq),
            'frequency': measured_freq,
            'name': kwargs.get('name', self.name),
            'z0': self._z0,
        })

        return skrf.Network(**kwargs)    
    
def make_reconstruct_function(
    model: Model,
    flat = False,
    return_params = False,
    numpy_input = False,
) -> tuple[Callable, ModelParametersT]:
    """Generate a reconstruct function to parametrically reconstruct a model, either from flat parameters or from the dynamic component of the model.
    
    Args:
        model (Model):                              The model to generate the feature functions for.
        flat (bool):                                Whether the feature function should accept a flat array as input.
        return_params (bool):                       Specifies that the parameters should be returned alongside the feature function.
                                                    For the flat case, the parameters are returned as an array. Defaults to `False`.
        numpy_input (bool):                         Specifies whether the return function should accept a numpy array as opposed to a jax array.
                                                    Mutually exclusive with `flat` since `numpy_input` implies it.

    Returns:
        tuple[FeatureFunction, ModelParameters]:    The feature function, alongside the partitioned or flattened model parameters.
    """    
    flat = flat or numpy_input
    params_tree, static = model.partition()
    
    if flat:
        params_out, unravel_fn = flatten_util.ravel_pytree(params_tree)
        
        if jnp.isscalar(params_out) or params_out.shape[0] == 0:
            raise Exception("Error: no free model parameters found to make feature function")
        
        def reconstruct_fn_jax(flat_params) -> Model:
            params_tree_recon = unravel_fn(flat_params)
            return combine(params_tree_recon, static)
    else:
        params_out = params_tree
        def reconstruct_fn_jax(params_tree) -> Model:
            return combine(params_tree, static)
        
    if numpy_input:
        params_out = np.array(params_out)
        reconstruct_fn = lambda x: reconstruct_fn_jax(jnp.array(x))
    else:
        reconstruct_fn = reconstruct_fn_jax
        
    if return_params:
        return reconstruct_fn, params_out
    else:
        reconstruct_fn
        
def make_feature_function(
    model: Model,
    features: FeatureInputT,
    freq: Frequency | skrf.Frequency,
    dtype: jnp.dtype = jnp.complex128,
    flat = False,
    return_params = False,
    numpy_input = False,
    nderiv = 0,
) -> tuple[FeatureFunctionT, ModelParametersT] | tuple[FeatureFunctionT, ModelParametersT, Callable]:
    """Generate a feature function to parametrically extract model features.
    
    This function returns a callable feature function to extract model features,
    alongside model parameters. The function can be just-in-time compiled using jax,
    to enable its efficient, machine-code level computation.
    
    The function generated accepts the model parameters, in either Pytree
    or flattened (raveled) formated, and returns the resultant model feature matrix.
    This function is convenient for lower-level use in order to remove close over any details of the model,
    and make use of a purely parametric function (e.g. for fitting or sampling).

    Args:
        model (Model):                              The model to generate the feature functions for.
        features (FeatureT | FeatureList):          The list of features. See `extract_features` for more information.
        freq (pmrf.Frequency):                      The frequency to extract the features at, treated as a static argument.
        dtype (jnp.dtype, optional):                The data type of the final out feature matrix.        
        flat (bool):                                Whether the feature function should accept a flat array as input.
        return_params (bool):                       Specifies that the parameters should be returned alongside the feature function.
                                                    For the flat case, the parameters are returned as an array. Defaults to `False`.
        numpy_input (bool):                         Specifies whether the return function should accept a numpy array as opposed to a jax array.
                                                    Only for the flat case.        
        nderiv (int):                               The number of derives to take of the feature function. Default to 0 for the original function.

    Returns:
        tuple[FeatureFunction, ModelParameters]:    The feature function, alongside the partitioned or flattened model parameters.
    """
    from pmrf._features import extract_features
    
    flat = flat or numpy_input
    if isinstance(freq, skrf.Frequency):
        freq = Frequency.from_skrf(freq)
    
    reconstruct_fn, params_out = make_reconstruct_function(model, flat=flat, return_params=True)
    
    if flat:
        if jnp.isscalar(params_out) or params_out.shape[0] == 0:
            raise Exception("Error: no free model parameters found to make feature function")
        
        def feature_fn_jax(flat_params) -> jnp.ndarray:
            model_recon = reconstruct_fn(flat_params)
            return extract_features(model_recon, features, freq, dtype=dtype)
    else:
        def feature_fn_jax(tree_params) -> jnp.ndarray:
            model_recon = reconstruct_fn(tree_params)
            return extract_features(model_recon, features, freq, dtype=dtype)

    for _ in range(nderiv):
        feature_fn_jax = jax.jacfwd(feature_fn_jax)
        
    feature_fn_jax = jax.jit(feature_fn_jax)
        
    if numpy_input:
        params_out = np.array(params_out)
        feature_fn = lambda x: feature_fn_jax(jnp.array(x))
    else:
        feature_fn = feature_fn_jax        
    
    if return_params:
        return feature_fn, params_out
    else:
        return feature_fn    