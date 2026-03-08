"""
The main model class.

This module defines :class:`pmrf.Model`, a frozen, JAX-compatible, Equinox module.

"""

import inspect
from functools import partial
from copy import copy, deepcopy
from typing import Callable, Sequence, Iterator, Self, ClassVar
import dataclasses
from dataclasses import fields, is_dataclass
from functools import update_wrapper
from collections.abc import Sequence
from typing import Sequence, Callable
from typing_extensions import dataclass_transform  # Or 'from typing import...' if >= 3.11

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import PyTree
from jax.tree_util import GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey
import equinox as eqx
import numpy as np
import skrf as skrf
from numpyro.distributions import Distribution, Uniform as UniformDistribution

from pmrf.rf_functions.conversions import a2s, s2a, s2z, z2s, s2y, y2s
from pmrf.math_functions import FUNC_LOOKUP
from pmrf.parameters import Parameter, ParameterGroup, is_valid_param, as_param
from pmrf.distributions.joint import JointDistribution
from pmrf.constants import PRIMARY_PROPERTIES
from pmrf.frequency import Frequency
from pmrf.util import classproperty, is_overridden, get_first_underlying_type, is_convertible_to_float
from pmrf.util.tree import nodes_by_type, value_at_path, partition, combine
from pmrf.field import field  # Import your newly created field

Z0_WARNING = \
r"""
WARNING: You have created a model with characteristic impedance other than 50 ohm.
Working with multiple models in ParamRF with differing characteristic impedances
is not yet officially supported and you may encounter subtle bugs. For now, it is
recommended to keep the default z0 and convert your results at the end.
"""

@dataclass_transform(field_specifiers=(field, eqx.field, dataclasses.field))
class ModelMeta(type(eqx.Module)):
    def __new__(mcs, name, bases, namespace, **kwargs):
        annotations = namespace.get('__annotations__', {})
        
        for field_name, field_types in annotations.items():
            field_kwargs = {}
            default = namespace.get(field_name, dataclasses.MISSING)
            
            field_type = get_first_underlying_type(field_types)
            is_param_type = field_type is not None and isinstance(field_type, type) and issubclass(field_type, Parameter)
            
            # 1. Handle explicit Field declarations (e.g., x = prf.field(...))
            if isinstance(default, dataclasses.Field):
                if is_param_type:
                    has_converter = default.metadata is not None and "converter" in default.metadata
                    
                    # If they didn't provide their own converter, inject ours
                    if not has_converter:
                        valid_keys = getattr(default, '__slots__', None) or vars(default).keys()
                        
                        field_kwargs = {
                            k: getattr(default, k) for k in valid_keys
                            if not k.startswith('_') and k not in ('name', 'type')
                        }
                        
                        field_kwargs['converter'] = lambda x, fn=field_name: as_param(x, fixed=False)
                        
                        if 'metadata' in field_kwargs and field_kwargs['metadata']:
                            field_kwargs['metadata'] = dict(field_kwargs['metadata'])
                            field_kwargs['metadata'].pop('converter', None)
                        
                        field_kwargs = {k: v for k, v in field_kwargs.items() if v is not dataclasses.MISSING}
                        
                        # USE CUSTOM FIELD: Repackage the explicitly defined field using prf.field
                        namespace[field_name] = field(**field_kwargs)
                continue

            # 2. Handle standard type hints (e.g., x: Parameter = 10)
            if is_param_type:
                if default is not dataclasses.MISSING and not isinstance(default, Parameter):                
                    if isinstance(default, tuple):
                        raise Exception(f"Expected a parameter for default '{field_name}' in class {name} but found a tuple.")
                    field_kwargs['default'] = default
                
                field_kwargs['converter'] = lambda x, fn=field_name: as_param(x, fixed=False)
            
            # 3. Apply default_factory to avoid Python's mutable default trap
            if default is not dataclasses.MISSING:
                if any(isinstance(default, m_type) for m_type in {list, dict, tuple, eqx.Module, jnp.ndarray}):
                    field_kwargs['default_factory'] = lambda default=default: deepcopy(default)
                    field_kwargs.pop('default', None)
            
            # 4. Inject the final configured field back into the namespace
            if len(field_kwargs) != 0:
                namespace[field_name] = field(**field_kwargs)
                
        return super().__new__(mcs, name, bases, namespace, **kwargs)

# 2. Assign the metaclass to your base Model
class Model(eqx.Module, metaclass=ModelMeta):
    """
    Overview
    --------
    This base class is used to represent any computable RF network, referred to in
    **ParamRF** as a "Model". This class can be overriden for defining complex models,
    or can be utilized indirectly by combining models already provides in :mod:`pmrf.models`.
    
    This class is abstract and should not be instantiated directly. Derive from :class:`Model`
    and override one of the primary property functions (e.g. :meth:`.__call__`, :meth:`.s`, :meth:`.a`).

    The model is an Equinox ``Module`` (immutable, dataclass-like) and is
    treated as a JAX PyTree. Parameters are declared using standard dataclass
    field syntax with types like :class:`pmrf.Parameter`.

    Usage
    -----
    - Define new models by sub-classing the model and adding custom parameters and/or sub-models
    - Construct models by passing parameters and/or submodels to the initializer (like a dataclass).
    - Retrieve parameter information via methods such as :meth:`.named_params`, :meth:`.param_names`, :meth:`.flat_params`, etc..
    - Use `with_xxx` functions to modify fields, models and parameters within the model e.g. :meth:`.with_params`, :meth:`.with_fields`.
    - Use "past tense" functions to modify the model in conjunction with another model or data e.g. :meth:`.terminated`, :meth:`.flipped`.

    See also the :mod:`pmrf.fitting` and :mod:`pmrf.sampling` modules for details on model fitting and sampling.

    Methods & Properties Summary
    ----------------------------

    **Defaults / Primary**
    
    ================================= ====================================================================
    Method / Property                 Description
    ================================= ====================================================================
    :attr:`DEFAULT_NAMED_PARAMS`      Mapping from parameter name to :class:`Parameter`.
    :attr:`DEFAULT_PARAM_NAMES`       Default parameter names for the model.
    :attr:`DEFAULT_PARAMS`            Default parameters for the model.
    :attr:`primary_function`          The primary function (``s`` or ``a``) as a callable.
    :attr:`primary_property`          The primary property (e.g. ``"s"``, ``"a"``) as a string.
    ================================= ====================================================================

    **Introspection Properties**

    ================================= ====================================================================
    Method / Property                 Description
    ================================= ====================================================================
    :attr:`number_of_ports`           Number of ports.
    :attr:`nports`                    Alias of :attr:`number_of_ports`.
    :attr:`port_tuples`               All (m, n) port index pairs.
    :attr:`num_params`                Number of free parameters.
    :attr:`num_flat_params`           Number of free, flattened parameters.
    ================================= ====================================================================

    **Core API**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`__call__`                  Build the model. Should be overridden by sub-classes.
    :meth:`primary`                   Dispatch to the primary function for the given frequency.
    :meth:`s`                         Scattering (S) parameter matrix.
    :meth:`a`                         ABCD parameter matrix.
    :meth:`z`                         Impedance (Z) parameter matrix.
    :meth:`y`                         Admittance (Y) parameter matrix.
    :meth:`s_jacobian`                Jacobian of the S-parameters w.r.t free parameters.
    :meth:`a_jacobian`                Jacobian of the ABCD-parameters w.r.t free parameters.
    :meth:`z_jacobian`                Jacobian of the Z-parameters w.r.t free parameters.
    :meth:`y_jacobian`                Jacobian of the Y-parameters w.r.t free parameters.
    ================================= ====================================================================

    **Function Tools**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`func_jacobian`             Calculate the Jacobian of an arbitrary function w.r.t parameters.
    :meth:`func_sensitivity`          Calculate the sensitivity of an arbitrary function w.r.t parameters.
    :meth:`func_samples`              Evaluate an arbitrary function over parameter samples.
    ================================= ====================================================================

    **Model Inspection & Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`children`                  Returns the immediate submodels.
    :meth:`submodels`                 Returns all nested submodels (depth-first).
    :meth:`partition`                 Partition model into parameters and static trees.
    :meth:`flipped`                   Return a version of the model with ports flipped.
    :meth:`renumbered`                Return a version of the model with ports renumbered.
    :meth:`terminated`                Return a new model terminated by another (e.g. load).
    ================================= ====================================================================

    **Parameter Inspection**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`named_params`              Named model parameter objects as a dict.
    :meth:`named_param_values`        Named model parameter values as a dict of jax arrays.
    :meth:`param_names`               Model parameter names as a list.
    :meth:`param`                     A single model parameter object by name.
    :meth:`params`                    Model parameters as a list.
    :meth:`param_value`               A single model parameter value by name.
    :meth:`param_values`              Model parameter values as a list of jax arrays.
    :meth:`named_flat_params`         Named flattened model parameter objects as a dict.
    :meth:`named_flat_param_values`   Named flattened model parameter values as a dict.
    :meth:`flat_param_names`          Flattened parameter names as a list.
    :meth:`flat_params`               Flattened parameters as a list.
    :meth:`flat_param_values`         Flattened model parameter values as a flat array.
    :meth:`flat_param_bounds`         Flattened model parameter bounds as jax arrays.
    :meth:`param_groups`              Return all parameter groups relevant to this model.
    :meth:`distribution`              Joint distribution over (flattened) parameters.
    ================================= ====================================================================

    **Parameter Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`with_params`               Return a model with parameters updated.
    :meth:`with_mapped_params`        Apply a map function to parameters.
    :meth:`with_fixed_params`         Return a model with specified parameters fixed.
    :meth:`with_free_params`          Return a model with specified parameters free.
    :meth:`with_free_params_only`     Return a model with ONLY the specified parameters free.
    :meth:`with_all_params_fixed`     Return a model with all parameters fixed.
    :meth:`with_all_params_free`      Return a model with all parameters free.
    ================================= ====================================================================

    **Parameter Group Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`with_param_groups`         Return a model with parameter groups appended.
    :meth:`with_demoted_param_groups` Recursively demote parameter groups to deepest submodel.
    :meth:`with_no_param_groups`      Return a model with all parameter groups removed.
    ================================= ====================================================================

    **Distribution Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`with_mapped_distributions` Apply a map function to the parameter distributions.
    :meth:`with_uniform_distributions` Return a model with uniform distributions set.
    ================================= ====================================================================
    
    
    **Field & Model Manipulation**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`with_defaults`             Return this model type with default initialization args.
    :meth:`with_models`               Combines this model with free parameters in other models.
    :meth:`with_fields`               Return a copy with dataclass-style field replacements.
    :meth:`with_name`                 Return a copy of this model with a different name.
    :meth:`with_submodel_fields`      Dataclass-style field replacements on a nested sub-model.
    :meth:`with_free_submodels`       Free all parameters in the given submodels.
    :meth:`with_free_submodels_only`  Returns a model with ONLY the specified submodels freed.
    :meth:`with_fixed_submodels`      Fix all parameters in the given submodels.
    :meth:`with_tied_submodels`       Tie submodels structurally to a shared model.
    :meth:`tied`                      Return the model with self tied to a shared model.
    :meth:`with_injected_params`      Inject parameters from a shared model into target submodels.
    ================================= ====================================================================

    **Plotting, File, & Conversion Utilities**

    ================================= ====================================================================
    Method                            Description
    ================================= ====================================================================
    :meth:`plot_func`                 Evaluate and plot an arbitrary function of the model.
    :meth:`plot_func_samples`         Evaluate and plot a function over parameter samples.
    :meth:`to_skrf`                   Convert the model at frequencies to an :class:`skrf.Network`.
    :meth:`export_touchstone`         Export the model response to a Touchstone file.
    ================================= ====================================================================    

    Attributes
    ----------
    name : str or None
        An optional name for the model instance.
    separator : str
        The separator character used when flattening nested parameter names (default is '_').
    metadata : dict
        A dictionary for storing arbitrary metadata associated with the model.

    Examples
    --------
    A ``PiCLC`` network ("foundational" model with fixed parameters and equations):

    .. code-block:: python

        import jax.numpy as jnp
        import pmrf as prf        

        class PiCLC(prf.Model):
            C1: prf.Parameter = 1.0e-12
            L:  prf.Parameter = 1.0e-9
            C2: prf.Parameter = 1.0e-12

            def a(self, freq: prf.Frequency) -> jnp.ndarray:
                w = freq.w
                Y1, Y2, Y3 = (1j * w * self.C1), (1j * w * self.C2), 1 / (1j * w * self.L)
                return jnp.array([
                    [1 + Y2 / Y3,        1 / Y3],
                    [Y1 + Y2 + Y1*Y2/Y3, 1 + Y1 / Y3],
                ]).transpose(2, 0, 1)

    An ``RLC`` network ("circuit" model with free parameters built using cascading)

    .. code-block:: python

        import pmrf as prf
        from pmrf.models import Resistor, Capacitor, Inductor, Cascade
        from pmrf.parameters import Uniform

        class RLC(prf.Model):
            res: Resistor = Resistor(Uniform(9.0, 11.0))
            ind: Inductor = Inductor(Uniform(0.0, 10.0, scale=1e-9))
            cap: Capacitor = Capacitor(Uniform(0.0, 10.0, scale=1e-12))

            def __call__(self) -> prf.Model:
                return self.res ** self.ind ** self.cap.terminated()
            
    """
    # Public init fields
    name: str | None = field(default=None, kw_only=True, static=True)
    z0: complex = field(default=50.0+0j, kw_only=True, static=True)
    
    # Private fields
    _separator: str = field(default='_', kw_only=True, repr=False, static=True, init=False)
    _metadata: dict = field(default_factory=dict, kw_only=True, repr=False, static=True, init=False)
    _param_groups: list = field(default_factory=list, kw_only=True, repr=False, static=True, init=False)

    # Class variables
    _transparent: ClassVar[bool] = False

    # ---- Internal initialization methods -------------------------------------------------

    def __init_subclass__(cls, transparent: bool = False, **kwargs):
        """Customize subclass construction."""        
        super().__init_subclass__(**kwargs)        

        cls._transparent = transparent

        # --- Helper for Z0 Validation ---
        def _validate_z0(instance, stacklevel):
            if not jnp.isscalar(instance.z0):
                raise Exception("Only scalar port impedances are currently supported.")
            
            # Ensure the warning only fires for the final instantiated class
            if type(instance) is cls:
                if instance.z0 != 50 and instance.z0 != (50.0+0j):
                    import warnings
                    warnings.warn(Z0_WARNING, UserWarning, stacklevel=stacklevel)
        # --------------------------------

        if '__init__' in cls.__dict__:
            user_init = cls.__dict__['__init__']
            sig = inspect.signature(user_init)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            
            def wrapped_init(self, *args, **init_kwargs):
                user_kwargs = {}
                base_kwargs = {}
                
                valid_fields = {f.name for f in dataclasses.fields(type(self))}
                for k, v in init_kwargs.items():
                    if accepts_kwargs or k in sig.parameters:
                        user_kwargs[k] = v
                    elif k in valid_fields or k in {"name", "z0"}:
                        base_kwargs[k] = v
                    else:
                        raise TypeError(f"{type(self).__name__}.__init__() got an unexpected keyword argument '{k}'")
                
                for k, v in base_kwargs.items():
                    object.__setattr__(self, k, v)
                    
                user_init(self, *args, **user_kwargs)
                
                # SIMPLIFIED: Only apply defaults/converters to fields the user's init missed!
                for f in dataclasses.fields(type(self)):
                    if not hasattr(self, f.name):
                        val = dataclasses.MISSING
                        
                        # Grab the default from the metaclass blueprint
                        if f.default is not dataclasses.MISSING:
                            val = f.default
                        elif f.default_factory is not dataclasses.MISSING:
                            val = f.default_factory()
                        
                        # If a default existed, convert it and set it
                        if val is not dataclasses.MISSING:
                            converter = f.metadata.get("converter") if hasattr(f, "metadata") else None
                            if converter is not None:
                                val = converter(val)
                            object.__setattr__(self, f.name, val)
                
                _validate_z0(self, stacklevel=2)
                
                if hasattr(self, '__post_init__'):
                    self.__post_init__()

            update_wrapper(wrapped_init, user_init)
            cls.__init__ = wrapped_init       

        else:
            user_post_init = getattr(cls, '__post_init__', None)
            
            def wrapped_post_init(self, *args, **kwargs_pi):
                if user_post_init is not None:
                    user_post_init(self, *args, **kwargs_pi)
                
                _validate_z0(self, stacklevel=3)
                        
            cls.__post_init__ = wrapped_post_init
            
        # --- Implement dynamic functions (s_mag, s_mn_mag, etc.) ---
        def make_dynamic_method(prop_name, func):
            def dynamic_method(self, *args, **kwargs):
                matrix = getattr(self, prop_name)(*args, **kwargs)
                return func(matrix)
            return dynamic_method
            
        for prop in PRIMARY_PROPERTIES:
            for suffix, lookup in FUNC_LOOKUP.items():
                func = lookup[1]
                
                # Base function (e.g. s_mag)
                func_name = f"{prop}_{suffix}"
                if not hasattr(cls, func_name):  # Protect user overrides!
                    m = make_dynamic_method(prop, func)
                    m._pmrf_auto = True
                    setattr(cls, func_name, m)
                
                # Indexed function (e.g. s_mn_mag)
                func_name_mn = f"{prop}_mn_{suffix}"
                if not hasattr(cls, func_name_mn):
                    m_mn = make_dynamic_method(f"{prop}_mn", func)
                    m_mn._pmrf_auto = True
                    setattr(cls, func_name_mn, m_mn)

    # ---- Internal PyTree manipulation, introspection and helpers -------------------------------------------------
    
    @property
    def _param_value_spec(self) -> PyTree:
        """PyTree spec for **all** parameter values within the model (arrays)."""
        def is_param_value(path, node):
            if len(path) == 0 or not eqx.is_inexact_array(node):
                return False
            return hasattr(path[-1], 'name') and path[-1].name == 'value'
        
        return jax.tree.map_with_path(is_param_value, self, is_leaf=lambda node: eqx.is_inexact_array(node))
        
    @property
    def _param_object_spec(self) -> PyTree:
        """PyTree spec for all :class:`Parameter` objects within the model."""
        return jax.tree.map(is_valid_param, self, is_leaf=lambda node: is_valid_param(node))
    
    @property
    def _core_value_spec(self) -> PyTree:
        """PyTree spec for core (non-derived) parameter values."""        
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
        """PyTree spec for core (non-derived) :class:`Parameter` objects."""        
        return self._param_object_spec
        # return jax.tree.map(lambda param, core_spec: is_valid_param(param) and core_spec.value, self, self.core_value_spec, is_leaf=lambda node: is_valid_param(node))

    @property
    def _free_value_spec(self) -> PyTree:
        """PyTree spec for **free** parameter values (arrays)."""        
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
        """PyTree spec for **free** :class:`Parameter` objects."""        
        # A Pytree filter for all free Model `Parameter` objects.
        return jax.tree.map(lambda param, fit_spec: is_valid_param(param) and fit_spec.value, self, self._free_value_spec, is_leaf=lambda node: is_valid_param(node))               
    
    def _path_to_param_name(self, path) -> str:
        """Convert a PyTree path to a fully-qualified parameter name."""
        name_fields = []
        node = self

        for item in path:
            if isinstance(item, GetAttrKey):
                k = item.name
                next_node = getattr(node, k)
                
                # 1. Determine transparency
                is_transparent = getattr(node, '_transparent', False)
                if not is_transparent and is_dataclass(node):
                    field_obj = next((f for f in fields(node) if f.name == k), None)
                    if field_obj is not None:
                        is_transparent = field_obj.metadata.get('transparent', False)

                # 2. Extract user override
                explicit_name = getattr(next_node, 'name', None)
                
                # 3. Rule application
                if explicit_name is not None:
                    # User explicitly named it.
                    if is_transparent:
                        name_fields.append(explicit_name) # Always use the explicit name
                    else:
                        name_fields.append(k) # Standard fields keep their namespace prefix
                else:
                    # No explicit name. We MUST use the variable name 'k' as fallback,
                    # UNLESS it's a transparent generic container (where dict/list keys handle it).
                    if is_transparent and isinstance(next_node, (list, tuple, dict)):
                        pass
                    else:
                        name_fields.append(k)
                        
                node = next_node
                
            elif isinstance(item, DictKey):
                k = item.key
                node = node[k]
                
                # Rule: Dictionaries ALWAYS use the key.
                name_fields.append(str(k))
                    
            elif isinstance(item, (SequenceKey, FlattenedIndexKey)):
                idx = item.idx if hasattr(item, 'idx') else item.key
                node = node[idx]
                
                # Rule: Sequences promote explicitly named models, 
                # otherwise use "model_{idx}" or just "{idx}".
                explicit_name = getattr(node, 'name', None)
                if explicit_name is not None:
                    name_fields.append(explicit_name)
                else:
                    name_fields.append(str(idx))
                    
            else:
                raise Exception(f"Unsupported key type in path: {type(item)}")
                
        return self._separator.join(name_fields)

    def _with_stripped_metadata(self: Self) -> Self:
        def strip_metadata_recursive(obj, memo=None):
            if memo is None:
                memo = {}
            
            # Return immediately if we've already processed this object (preserves shared references)
            obj_id = id(obj)
            if obj_id in memo:
                return memo[obj_id]

            # Case 1: It's an Equinox Module
            if isinstance(obj, Model):
                updates = {}
                
                # 1. Recursively clean fields and sort by init status
                for field in dataclasses.fields(obj):
                    if field.name == '_metadata':
                        continue
                    
                    val = getattr(obj, field.name)
                    processed_val = strip_metadata_recursive(val, memo)
                    
                    updates[field.name] = processed_val
                
                for k, v in updates.items():
                    object.__setattr__(obj, k, v)
                
                # 4. Clear the metadata for this specific module safely
                if hasattr(obj, '_metadata'):
                    object.__setattr__(obj, '_metadata', dict())
                    
                memo[obj_id] = obj
                return obj

            # Case 2: Standard containers
            elif isinstance(obj, (list, tuple)):
                obj = type(obj)(strip_metadata_recursive(x, memo) for x in obj)
                memo[obj_id] = obj
                return obj

            elif isinstance(obj, dict):
                obj = {k: strip_metadata_recursive(v, memo) for k, v in obj.items()}
                memo[obj_id] = obj
                return obj

            # Case 3: Leaf nodes
            else:
                return obj
            
        return strip_metadata_recursive(copy(self))
    
    def _saveable(self: Self) -> Self:
        def strip_unsaveable_recursive(obj, memo=None):
            if memo is None:
                memo = {}
            
            obj_id = id(obj)
            if obj_id in memo:
                return memo[obj_id]

            # Case 1: It's an Equinox Module
            if isinstance(obj, Model):
                updates = {}
                
                for f in dataclasses.fields(obj):
                    # Check for our custom save flag
                    if f.metadata.get('save', True) is False:
                        if f.default is not dataclasses.MISSING:
                            new_val = f.default
                        elif f.default_factory is not dataclasses.MISSING:
                            new_val = f.default_factory()
                        else:
                            new_val = None
                    else:
                        val = getattr(obj, f.name)
                        new_val = strip_unsaveable_recursive(val, memo)
                        
                    updates[f.name] = new_val
                
                # Override non-init fields
                for k, v in updates.items():
                    object.__setattr__(obj, k, v)

                memo[obj_id] = obj
                return obj

            # Case 2: Standard containers
            elif isinstance(obj, (list, tuple)):
                obj = type(obj)(strip_unsaveable_recursive(x, memo) for x in obj)
                memo[obj_id] = obj
                return obj

            elif isinstance(obj, dict):
                obj = {k: strip_unsaveable_recursive(v, memo) for k, v in obj.items()}
                memo[obj_id] = obj
                return obj

            # Case 3: Leaf nodes
            else:
                return obj

        clean_model = strip_unsaveable_recursive(copy(self))
        return clean_model._with_stripped_metadata()
    
    def _iter_params(
        self,
        param_filter: str | Sequence[str] | Sequence[Parameter] | Callable[[str], bool] = None,
        *,
        include_fixed: bool = False,
        flatten: bool = False,
        submodels: 'Model | Sequence[Model] | str | Sequence[str] | None' = None,
    ) -> Iterator[tuple[str, Parameter]]:
        """Iterate over (name, Parameter) pairs in internal order."""
        spec = self._core_object_spec if include_fixed else self._free_object_spec
        params_tree = eqx.filter(self, spec, is_leaf=is_valid_param)
        path_and_params, _ = jax.tree.flatten_with_path(params_tree, is_leaf=is_valid_param)
        params: list[tuple[str, Parameter]] = [(self._path_to_param_name(path), param) for path, param in path_and_params]

        # Parameter filtering
        if param_filter is not None:
            # Normalization
            if isinstance(param_filter, str):
                param_filter = [param_filter]

            # Apply filter
            if isinstance(param_filter, Sequence) and isinstance(param_filter[0], str):
                params = [(k, v) for k, v in params if k in param_filter]
            elif isinstance(param_filter, Sequence) and isinstance(param_filter[0], Parameter):
                filter_ids = [id(v) for v in param_filter]
                params = [(k, v) for k, v in params if id(v) in filter_ids]
            elif isinstance(param_filter, Callable):
                params = [(k, v) for k, v in params if param_filter(k)]
            else:
                raise Exception(f"Unknown filter type passed for parameters: {param_filter}")

        # Submodel filtering
        if submodels is not None:
            if isinstance(submodels, (Model, str)):
                submodels: list[Model] = [submodels]
            if isinstance(submodels[0], str):
                submodels: list[Model] = [getattr(self, name) for name in submodels]
            if not isinstance(submodels[0], Model):
                raise Exception(f"Got unknown type when expecting a model or string. Type was: {submodels}")

            allowed = {id(p) for sm in submodels for p in sm.params(include_fixed=include_fixed)}
            params = [(k, v) for k, v in params if id(v) in allowed]

        # Flatten multi-dimensional parameters if requested
        if flatten:
            flat_params: list[tuple[str, Parameter]] = []
            for name, param in params:
                if param.size > 1 or param.flat_names is not None:
                    flattened_params = param.flattened(separator=self._separator)
                    for i, subparam in enumerate(flattened_params):
                        suffix = subparam.name if subparam.name is not None else str(i)
                        flat_params.append((f"{name}{self._separator}{suffix}", subparam))
                else:
                    flat_params.append((name, param))
            params = flat_params

        yield from params
    
    # ---- Defaults / Primary ---------------------------------------------------    
    
    @classproperty
    def DEFAULT_NAMED_PARAMS(cls) -> dict[str, Parameter]:
        """Default named parameters for the model.

        Returns
        -------
        dict[str, Parameter]
            Mapping from parameter name to :class:`Parameter`.
        """
        instance = cls()
        return instance.named_params()
    
    @classproperty
    def DEFAULT_PARAM_NAMES(cls) -> list[str]:
        """Default parameter names for the model.

        Returns
        -------
        list[str]
        """
        instance = cls()
        return instance.param_names()
    
    @classproperty
    def DEFAULT_PARAMS(cls) -> list[Parameter]:
        """Default parameters for the model.

        Returns
        -------
        list[str]
        """
        instance = cls()
        return instance.params()
    
    @property
    def primary_function(self) -> Callable[[Frequency], jnp.ndarray]:
        """The primary function (``s`` or ``a``) as a callable.

        The primary function is the first overridden among
        :data:`PRIMARY_PROPERTIES`, unless ``__call__`` is overridden,
        in which case the primary function of the built model is returned.

        Returns
        -------
        Callable[[Frequency], jnp.ndarray]

        Raises
        ------
        NotImplementedError
            If no primary property is overridden.
        """
        return getattr(self, self.primary_property)
            
    @property
    def primary_property(self) -> str:
        """The primary property (e.g. ``"s"``, ``"a"``) as a string.

        The primary property is the first overridden among
        :data:`PRIMARY_PROPERTIES`, unless ``__call__`` is overridden,
        in which case the primary property of the built model is returned.

        Returns
        -------
        str

        Raises
        ------
        NotImplementedError
            If no primary property is overridden.
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

    # ---- Introspection properties --------------------------------------------------------
    
    @property
    @eqx.filter_jit
    def number_of_ports(self) -> int:
        """Number of ports.

        Returns
        -------
        int
        """
        freq = Frequency(1, 2, 2)
        eval = jax.eval_shape(lambda: self.s(freq))
        return eval.shape[1]

    @property
    def nports(self) -> int:
        """Alias of :attr:`number_of_ports`."""
        return self.number_of_ports
    
    @property
    def port_tuples(self) -> list[tuple[int, int]]:
        """All (m, n) port index pairs.

        Returns
        -------
        list[tuple[int, int]]
        """
        return [(y, x) for x in range(self.nports) for y in range(self.nports)]        
    
    @property
    def num_params(self) -> int:
        """Number of free parameters.

        Returns
        -------
        int
        """
        return len(self.params())

    @property
    def num_flat_params(self) -> int:
        """Number of free, **flattened** parameters.

        Returns
        -------
        int
        """
        return len(self.flat_params())
    
    # ---- Core API -------------------------------------------------------------
    
    def __call__(self) -> 'Model':
        """Build the model.

        This function should be over-ridden by sub-classes.
        It is useful in defining complex models that are built
        using several sub-models (as opposed to equation-based models).

        Returns
        -------
        Model

        Raises
        ------
        NotImplementedError
            In the base class; override in derived classes to build
            a compositional representation.
        """     
        raise NotImplementedError
    
    @eqx.filter_jit
    def primary(self, freq: Frequency) -> jnp.ndarray:
        """Dispatch to the primary function for the given frequency."""        
        primary_function = self.primary_function
        return primary_function(freq)
    
    @eqx.filter_jit
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Scattering parameter matrix.

        If a different parameter type (a, z, y) is primary, this converts it to S.
        
        Note that, in ParamRF, the **power wave** definition of S-parameters
        should be used. If you have a formulation in terms of another definition
        (such as traveling waves), simply use :meth:`pmrf.rf_functions.s2s`
        (or :meth:`pmrf.rf_functions.renormalize_s` if you need to change
        impedance too).

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            S-parameter matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, '__call__'):
            return self().s(freq)

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary(freq)

        # 2. Return or Convert
        if primary_prop == 's':
            return val
        elif primary_prop == 'a':
            return a2s(val, self.z0)
        elif primary_prop == 'z':
            return z2s(val, self.z0)
        elif primary_prop == 'y':
            return y2s(val, self.z0)
        
        raise NotImplementedError(f"Conversion from '{primary_prop}' to 's' is not implemented.")
    
    @eqx.filter_jit
    def a(self, freq: Frequency) -> jnp.ndarray:
        """ABCD parameter matrix.

        If a different parameter type is primary, this converts it to A.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            ABCD matrix with shape ``(nf, 2, 2)``.
        """        
        if is_overridden(type(self), Model, '__call__'):
            return self().a(freq)
        
        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary(freq)

        # 2. Return or Convert
        if primary_prop == 'a':
            return val
        
        # Convert via S parameters (Hub strategy)
        if primary_prop == 's':
            s = val
        elif primary_prop == 'z':
            s = z2s(val, self.z0)
        elif primary_prop == 'y':
            s = y2s(val, self.z0)
        else:
            raise NotImplementedError(f"Conversion from '{primary_prop}' to 'a' is not implemented.")
            
        return s2a(s, self.z0)

    @eqx.filter_jit
    def z(self, freq: Frequency) -> jnp.ndarray:
        """Impedance (Z) parameter matrix.

        If a different parameter type is primary, this converts it to Z.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            Z matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, '__call__'):
            return self().z(freq)

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary(freq)

        # 2. Return or Convert
        if primary_prop == 'z':
            return val

        # Convert via S parameters (Hub strategy)
        if primary_prop == 's':
            s = val
        elif primary_prop == 'a':
            s = a2s(val, self.z0)
        elif primary_prop == 'y':
            s = y2s(val, self.z0)
        else:
            raise NotImplementedError(f"Conversion from '{primary_prop}' to 'z' is not implemented.")

        return s2z(s, self.z0)

    @eqx.filter_jit
    def y(self, freq: Frequency) -> jnp.ndarray:
        """Admittance (Y) parameter matrix.

        If a different parameter type is primary, this converts it to Y.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            Y matrix with shape ``(nf, n, n)``.
        """
        if is_overridden(type(self), Model, '__call__'):
            return self().y(freq)

        # 1. Fetch primary
        primary_prop = self.primary_property
        val = self.primary(freq)

        # 2. Return or Convert
        if primary_prop == 'y':
            return val

        # Convert via S parameters (Hub strategy)
        if primary_prop == 's':
            s = val
        elif primary_prop == 'a':
            s = a2s(val, self.z0)
        elif primary_prop == 'z':
            s = z2s(val, self.z0)
        else:
            raise NotImplementedError(f"Conversion from '{primary_prop}' to 'y' is not implemented.")

        return s2y(s, self.z0)        
    
    @eqx.filter_jit
    def s_jacobian(self: Self, freq: Frequency) -> dict[str, jnp.ndarray]:
        """Calculate the Jacobian of the S-parameters with respect to free parameters.

        See :meth:`.func_jacobian`.

        Parameters
        ----------
        freq : Frequency
            The frequency grid to evaluate the S-parameters over.

        Returns
        -------
        dict[str, jnp.ndarray]
            A dictionary mapping flat parameter names to their gradient 
            arrays. Each array has shape (n_freqs, n_ports, n_ports).
        """
        return self.func_jacobian(lambda mdl, f: mdl.s(f), freq)
    
    @eqx.filter_jit
    def a_jacobian(self: Self, freq: Frequency) -> dict[str, jnp.ndarray]:
        """Calculate the Jacobian of the ABCD-parameters with respect to free parameters.

        See :meth:`.func_jacobian`.

        Parameters
        ----------
        freq : Frequency
            The frequency grid to evaluate the ABCD-parameters over.

        Returns
        -------
        dict[str, jnp.ndarray]
            A dictionary mapping flat parameter names to their gradient 
            arrays. Each array has shape (n_freqs, n_ports, n_ports).
        """
        return self.func_jacobian(lambda mdl, f: mdl.a(f), freq)
    
    @eqx.filter_jit
    def z_jacobian(self: Self, freq: Frequency) -> dict[str, jnp.ndarray]:
        """Calculate the Jacobian of the Z-parameters with respect to free parameters.

        See :meth:`.func_jacobian`.

        Parameters
        ----------
        freq : Frequency
            The frequency grid to evaluate the Z-parameters over.

        Returns
        -------
        dict[str, jnp.ndarray]
            A dictionary mapping flat parameter names to their gradient 
            arrays. Each array has shape (n_freqs, n_ports, n_ports).
        """
        return self.func_jacobian(lambda mdl, f: mdl.z(f), freq)
    
    @eqx.filter_jit
    def y_jacobian(self: Self, freq: Frequency) -> dict[str, jnp.ndarray]:
        """Calculate the Jacobian of the Y-parameters with respect to free parameters.

        See :meth:`.func_jacobian`.

        Parameters
        ----------
        freq : Frequency
            The frequency grid to evaluate the Y-parameters over.

        Returns
        -------
        dict[str, jnp.ndarray]
            A dictionary mapping flat parameter names to their gradient 
            arrays. Each array has shape (n_freqs, n_ports, n_ports).
        """
        return self.func_jacobian(lambda mdl, f: mdl.y(f), freq)
    
    # ---- Function tools --------------------------------------------------        

    @eqx.filter_jit
    def func_jacobian(
        self: Self, 
        func: Callable[['Model', Frequency], jnp.ndarray], 
        freq: Frequency
    ) -> dict[str, jnp.ndarray]:
        """Calculate the Jacobian of an arbitrary function with respect to free parameters.

        This uses forward-mode automatic differentiation to compute the gradients 
        of the provided function with respect to each free parameter in the model.

        Parameters
        ----------
        func : Callable[[Model, Frequency], jnp.ndarray]
            Function to differentiate. Must take a Model and a Frequency object 
            and return a jnp.ndarray of any shape.
        freq : Frequency
            The frequency grid to evaluate the function over.

        Returns
        -------
        dict[str, jnp.ndarray]
            A dictionary mapping flat parameter names to their gradient 
            arrays. Each array has the same shape as the output of `func`.
        """
        def func_from_flat(flat_params_array: jnp.ndarray) -> jnp.ndarray:
            sampled_model = self.with_params(flat_params_array)
            return func(sampled_model, freq)

        # Calculate the Jacobian. By default, JAX appends the parameter dimension 
        # to the end of the output shape: (*func_shape, num_params)
        jac_array = jax.jacfwd(func_from_flat)(self.flat_param_values())
        
        # Move the parameter dimension to the front: (num_params, *func_shape)
        jac_moved = jnp.moveaxis(jac_array, -1, 0)
        
        param_names = self.flat_param_names()
        
        # Map each slice to its corresponding parameter name
        return {name: jac_moved[i] for i, name in enumerate(param_names)}
    
    @eqx.filter_jit
    def func_sensitivity(
        self: Self, 
        func: Callable[['Model', Frequency], jnp.ndarray], 
        freq: Frequency
    ) -> dict[str, jnp.ndarray]:
        r"""Calculate the relative (normalized) sensitivity of an arbitrary function.

        This computes the fractional change in the function's output given a 
        fractional change in each free parameter. Mathematically, it evaluates:
        $$S_{rel} = \frac{\partial y}{\partial \theta} \frac{\theta}{y}$$

        Parameters
        ----------
        func : Callable[[Model, Frequency], jnp.ndarray]
            Function to evaluate. Must take a Model and a Frequency object and 
            return a jnp.ndarray of any shape.
        freq : Frequency
            The frequency grid to evaluate the function over.

        Returns
        -------
        dict[str, jnp.ndarray]
            A dictionary mapping flat parameter names to their normalized 
            sensitivity arrays. Each array has the same shape as the output 
            of `func`.
        """
        # 1. Get nominal value and absolute jacobian
        y_nom = func(self, freq)
        abs_jac = self.func_jacobian(func, freq)
        
        # 2. Get nominal parameter values
        param_vals = self.named_flat_param_values()
        
        # 3. Normalize to get relative sensitivity.
        # Prevent division by zero by replacing exact zeros with a tiny epsilon.
        y_safe = jnp.where(y_nom == 0, 1e-15, y_nom)
        
        return {
            name: jac * (param_vals[name] / y_safe)
            for name, jac in abs_jac.items()
        }    

    @eqx.filter_jit
    def func_samples(
        self, 
        func: Callable[['Model', Frequency], jnp.ndarray], 
        freq: Frequency,
        *,
        key: jax.Array, 
        num_samples: int = 1000
    ) -> jnp.ndarray:
        """
        Evaluates an arbitrary function over samples drawn from the 
        model's distribution.

        Parameters
        ----------
        func : Callable[[Model], jnp.ndarray]
            A function that takes a Model instance and returns a JAX array.
        prng_key : jax.Array
            JAX random key for sampling.
        num_samples : int, default=1000
            Number of models to sample from the joint distribution.

        Returns
        -------
        jnp.ndarray
            The function evaluated over all samples. Shape will be 
            (num_samples, *func_output_shape).
        """
        # 1. Get the joint distribution and sample it
        dist = self.distribution()
        flat_param_samples = dist.sample(key, sample_shape=(num_samples,))

        # 2. Define the single-sample evaluation, passing freq explicitly
        def evaluate_single(flat_params_array):
            sampled_model = self.with_params(flat_params_array)
            return func(sampled_model, freq)

        # 3. Vectorize over the samples
        return jax.vmap(evaluate_single)(flat_param_samples)  
    
    # ---- Magic methods and copying --------------------------------------------------

    def copy(self: Self) -> Self:
        """Returns a deepcopy of self.

        Returns
        -------
        Model
        """        
        return deepcopy(self)    

    def __getattr__(self, name: str):
        """
        Dynamic dispatch for scikit-rf plotting methods.
        
        Captures calls like `model.plot_s_db(freq)` and redirects them 
        to `model.to_skrf(freq).plot_s_db()`.
        """
        if name.startswith('plot_'):
            def plotter(freq: Frequency, *args, **kwargs):
                # 1. Convert to scikit-rf Network at the specified frequency
                ntwk = self.to_skrf(freq)
                
                # 2. Check if the generated Network actually supports this plot type
                if not hasattr(ntwk, name):
                    raise AttributeError(f"scikit-rf Network object has no attribute '{name}'")
                
                # 3. Call the scikit-rf plot method with remaining args (e.g. labels, colors)
                return getattr(ntwk, name)(*args, **kwargs)
            return plotter
            
        # Standard fallback if the attribute isn't a plot command
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")    
    
    def __pow__(self, other: 'Model') -> 'Model':
        """Cascade/terminatino composition operator ``**``."""        
        if other.nports == 1:
            from pmrf.models import Terminated
            return Terminated(self, other)
        else:
            from pmrf.models import Cascade
            return Cascade([self, other])
    
    def __getitem__(self, key: str | Sequence[str]):
        if isinstance(key, str):
            return self.param_value(key)
        else:
            named_param_values = self.named_param_values()
            for k in key:
                if k not in named_param_values.keys():
                    raise Exception(f"Parameter name '{k}' was passed but is not a free parameter")
            return [v for k, v in named_param_values.items() if k in key]
        
    def __repr__(self):
        from pmrf.parameters import Parameter 
        
        model_param_fields = []
        other_fields = []
        base_fields = []
        
        for f in dataclasses.fields(self):
            # Skip hidden/internal fields like _separator and _metadata
            if f.repr is False:
                continue
            
            val = getattr(self, f.name)
            val_repr = repr(val)
            
            # Indent multi-line strings (like nested models) for perfect alignment
            indented_val_repr = val_repr.replace('\n', '\n    ')
            formatted_field = f"    {f.name}={indented_val_repr}"
            
            # Sort into the three buckets:
            
            # 1. Base fields (name and z0) go at the very bottom
            if f.name == "name":
                base_fields.append(formatted_field)
            elif f.name == "z0":
                base_fields.append(formatted_field)
                
            # 2. Models and Parameters go at the very top
            elif isinstance(val, (Model, Parameter)):
                model_param_fields.append(formatted_field)
                
            # 3. Everything else (bools, ints, floats) goes in the middle
            else:
                other_fields.append(formatted_field)
            
        # Combine the lists in the requested order
        all_fields_str = model_param_fields + other_fields + base_fields
        joined_fields = ",\n".join(all_fields_str)
        
        return f"{self.__class__.__name__}(\n{joined_fields}\n)"

    # ---- Model inspection --------------------------------------------------    
    
    def children(self) -> list['Model']:
        """Returns the immediate submodels.

        Returns
        -------
        list[Model]
        """
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Model)]
    
    def submodels(self) -> list['Model']:
        """Returns all nested submodels (depth-first), excluding ``self``.

        Returns
        -------
        list[Model]
        """
        return nodes_by_type(self, Model)[1:]        

    # ---- Model Manipulation --------------------------------------------------    

    def partition(self: Self, include_fixed=False, param_objects=False) -> tuple[Self, Self]:        
        """Partition model into (parameters, static) trees.
        
        This is useful for internal use, or for inspecting the model
        and its parameters.
        
        Parameters
        ----------
        include_fixed : bool, default=False
            Include fixed parameters in the parameter tree.
        param_objects : bool, default=False
            If ``True``, keep full :class:`Parameter` objects; otherwise filter to ``.value``.

        Returns
        -------
        (Self, Self)
            ``(params_tree, static_tree)``
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

    def flipped(self, **kwargs) -> 'Model':
        """Return a version of the model with ports flipped.
        
        See :class:`pmrf.models.composite.transformed.Flipped`.

        Returns
        -------
        Model
        """
        from pmrf.models import Flipped
        if isinstance(self, Flipped):
            return self.model
        return Flipped(self, **kwargs)

    def renumbered(self, from_ports: tuple[int], to_ports: tuple[int]= None, **kwargs) -> 'Model':
        """Return a version of the model with ports renumbered.
        
        See :class:`pmrf.models.composite.transformed.Renumbered`.

        from_ports : tuple[int]
            The original port indices that map to `to_ports`.
        to_ports : tuple[int]
            The new port indices.
            
        Returns
        -------
        Model
        """
        from pmrf.models import Renumbered
        return Renumbered(self, from_ports, to_ports, **kwargs)
    
    def terminated(self, load: 'Model' = None, **kwargs) -> 'Model':
        """Returns a new model that contains this model terminated in another.
        
        See :class:`pmrf.models.composite.transformed.Terminated`.

        Parameters
        ----------
        load : Model, optional
            Load network. Defaults to a SHORT.

        Returns
        -------
        Model
        """
        from pmrf.models import SHORT
        from pmrf.models import Terminated
        load = load or SHORT
        return Terminated(self, load, **kwargs)
       
    # ---- Parameter inspection -------------------------------------------------- 
    
    def named_params(self, param_filter: str | Sequence[str] | Sequence[Parameter] | Callable[[str], bool] = None, *, include_fixed=False, submodels: 'Model' | Sequence['Model'] | str | Sequence[str] | None = None) -> dict[str, Parameter]:
        """Named model parameters as a dict.

        Keys are fully-qualified parameter names.
        The order matches the internal flattened array order.

        Parameters
        ----------
        param_filter : str | Sequence[str] | Sequence[Parameter] | Callable[[str], bool], default=None
            A filter indicating which parameters to return. For the default case, all parameters are returned.
        include_fixed : bool, default=False
            Include fixed parameters.
        submodels : Model | Sequence[Model] | str | Sequence[str] | None, optional
            Restrict to parameters used by the given submodel(s). If strings are
            provided, ``getattr(self, name)`` is used.

        Returns
        -------
        dict[str, Parameter]
        """
        return dict(self._iter_params(param_filter=param_filter, include_fixed=include_fixed, submodels=submodels))
    
    def named_param_values(self, scaled=False, **kwargs) -> dict[str, jnp.ndarray]:
        """Named model parameter values as a dict of jax arrays.

        See :meth:`.named_params`.

        Parameters
        ----------
        scaled : bool, default=False
            Whether or not to scale the returned values by the parameter scales.
        **kwargs
            Additional key-word arguments as in  :meth:`.named_params`.

        Returns
        -------
        dict[str, jnp.ndarray]
        """     
        if scaled:
            return {n: jnp.array(p) for n, p in (self._iter_params(**kwargs))}
        else:
            return {n: p.value for n, p in (self._iter_params(**kwargs))}    

    def param_names(self, *args, **kwargs) -> list[str]:
        """
        Return model parameter names as a list.

        See :meth:`.named_params`.
        """
        return list(self.named_params(*args, **kwargs).keys())

    def param(self, name: str, *args, **kwargs) -> Parameter:
        """
        Return a single model parameter by name.

        See :meth:`.named_params`.
        """
        return self.named_params(*args, **kwargs)[name]
    
    def params(self, *args, **kwargs) -> list[Parameter]:
        """
        Return model parameters as a list.

        See :meth:`.named_params`.
        """
        return list(self.named_params(*args, **kwargs).values())
    
    def param_value(self, name: str, *args, **kwargs) -> jnp.ndarray:
        """
        Return a single model parameter value by name as a single jax array.

        See :meth:`.named_param_values`.
        """
        return self.named_param_values(*args, **kwargs)[name]

    def param_values(self, *args, **kwargs) -> list[jnp.ndarray]:
        """
        Return model parameter values as a list of jax arrays.

        See :meth:`.named_param_values`.
        """
        return list(self.named_param_values(*args, **kwargs).values())
    
    def named_flat_params(self, include_fixed=False, submodels: 'Model' | Sequence['Model'] | str | Sequence[str] | None = None) -> dict[str, Parameter]:
        """Named flattened model parameters as a dict.

        Flat parameters are a de-vectorized version of
        the internal parameters of the model. The returned
        parameter objects therefore are not necessarily
        equal to the internal model objects.
        
        Keys are fully-qualified parameter names with de-vectorized suffixes added.
        The order matches the internal flattened array order.

        Parameters
        ----------
        include_fixed : bool, default=False
            Include fixed parameters.
        submodels : Model | Sequence[Model] | str | Sequence[str] | None, optional
            Restrict to parameters used by the given submodel(s). If strings are
            provided, ``getattr(self, name)`` is used.

        Returns
        -------
        dict[str, Parameter]
        """
        return dict(self._iter_params(flatten=True, include_fixed=include_fixed, submodels=submodels))
    
    def named_flat_param_values(self, scaled=False, return_floats=False, **kwargs) -> dict[str, jnp.ndarray]:
        """Named flattened model parameter values as a dict of jax arrays.

        See :meth:`.named_flat_params`.

        Parameters
        ----------
        scaled : bool, default=False
            Whether or not to scale the returned values by the parameter scales.
        **kwargs
            Additional key-word arguments as in  :meth:`.named_params`.

        Returns
        -------
        dict[str, jnp.ndarray]
        """     
        if scaled:
            retval = {n: jnp.array(p) for n, p in (self._iter_params(flatten=True, **kwargs))}
        else:
            retval = {n: p.value for n, p in (self._iter_params(flatten=True, **kwargs))}
            
        if return_floats:
            retval = {k: float(np.array(v)) for k, v in retval.items()}
        return retval
         
    def flat_param_names(self, *args, **kwargs) -> list[str]:
        """
        Return flattened parameter names as a list.

        See :meth:`.named_flat_params`.
        """
        return list(self.named_flat_params(*args, **kwargs).keys())    
    
    def flat_params(self, *args, **kwargs) -> list[Parameter]:
        """
        Return flattened parameters as a list.

        See :meth:`.named_flat_params`.
        """
        return list(self.named_flat_params(*args, **kwargs).values())
    
    def flat_param_values(self, *args, **kwargs) -> jnp.ndarray:
        """
        Return flattened model parameter values as a jax arrays.

        See :meth:`.named_flat_param_values`.
        """
        return jnp.array(list(self.named_flat_param_values(*args, **kwargs).values())).reshape(-1)

    def flat_param_bounds(self, **kwargs) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return flattened model parameter bounds as jax arrays.
        
        Note that a minimum and maximum percentile is used to get the bounds
        for any non-uniform distribution.

        Equivalent to getting the bounds from :meth:`.distribution`,
        which key-word arguments are forwarded to.
        """
        return self.distribution(**kwargs).bounds
    
    def param_groups(self, include_fixed=False, explicit_only=False) -> list[ParameterGroup]:
        """Return all parameter groups relevant to this model, including submodels.

        This function recursively traverses submodels to collect their parameter groups,
        adjusting parameter names to match the current model's scope.

        Priority is given to groups defined in the parent model. If a parameter is 
        grouped explicitly in `self._param_groups`, it will be removed from any 
        groups returned by submodels.

        Parameters
        ----------
        include_fixed : bool, default=False
            Include groups involving fixed parameters.

        Returns
        -------
        list[ParameterGroup]
        """
        if explicit_only:
            return deepcopy(self._param_groups)
        
        # 0. Identify valid parameters for the current mode (Free vs All)
        # We use named_flat_params to get the definitive list of "active" parameters.
        # This handles the logic for whether parameters are fixed or not.
        all_valid_params = self.named_flat_params(include_fixed=include_fixed)
        valid_param_names = set(all_valid_params.keys())

        # 1. Start with local, explicit groups defined in this model
        # We only keep groups that contain at least one parameter that is valid 
        # (i.e. not fixed, unless include_fixed=True).
        groups = []
        for group in self._param_groups:
            # We check if the group overlaps with the valid parameters.
            # If the intersection is empty, it means all parameters in the group 
            # are fixed (or don't exist), so we exclude the group.
            if not set(group.param_names).isdisjoint(valid_param_names):
                groups.append(deepcopy(group))

        # 2. Traverse submodels to get their groups recursively
        # We use tree_flatten_with_path to find all Model instances within self.
        # We treat Model instances as leaves so we don't traverse into their individual parameters here.
        path_and_nodes, _ = jax.tree_util.tree_flatten_with_path(
            self, 
            is_leaf=lambda x: isinstance(x, Model) and x is not self
        )

        for path, node in path_and_nodes:
            # Check if the node is a submodel (and not self, though is_leaf handles that mostly)
            if isinstance(node, Model) and node is not self:
                # Calculate the prefix for this submodel (e.g., "amplifier_")
                relative_name = self._path_to_param_name(path)
                prefix = f"{relative_name}{self._separator}" if relative_name else ""

                # Recursively get groups from the submodel
                sub_groups = node.param_groups(include_fixed=include_fixed)

                # "Lift" the submodel groups into the current namespace
                for sub_group in sub_groups:
                    new_names = [prefix + name for name in sub_group.param_names]
                    # Create a new group with the updated names
                    lifted_group = dataclasses.replace(sub_group, param_names=new_names)
                    groups.append(lifted_group)

        # 3. Deduplication and Conflict Resolution
        # We prioritize groups that appear earlier in the list (Parent groups > Submodel groups).
        # We filter the list to ensure every parameter appears in exactly one group.
        
        final_groups = []
        seen_params = set()

        for group in groups:
            # Find parameters in this group that haven't been claimed by a higher-priority group
            valid_names = [name for name in group.param_names if name not in seen_params]
            
            # If the group has valid parameters left, add it
            if valid_names:
                # If the group shrank (because parent claimed some params), update it
                if len(valid_names) != len(group.param_names):
                    group = dataclasses.replace(group, param_names=valid_names)
                
                final_groups.append(group)
                seen_params.update(valid_names)

        # 4. Handle Orphans
        # Any parameter in the entire model that wasn't caught in the steps above 
        # (mostly local parameters of `self` that weren't in `_param_groups`) gets a singleton group.
        all_params = self.named_flat_params(include_fixed=include_fixed)
        
        for name, param in all_params.items():
            if name not in seen_params:
                final_groups.append(ParameterGroup(param_names=[name], distribution=param.distribution))
                seen_params.add(name)

        return final_groups
    
    def distribution(self, param_groups: bool = True) -> JointDistribution:
        """Joint distribution over (flattened) parameters.
        
        Parameters
        ----------
        param_groups : bool, optional
            Whether or not to use the internal parameter groups
            to create the joint distribution. Defaults to ``True``.
        
        Returns
        -------
        JointParameterDistribution
        """
        if param_groups:
            groups = self.param_groups()
            group_names = [pg.param_names for pg in groups]
            group_dists = [pg.distribution for pg in groups]
        else:
            named_flat_params = self.named_flat_params()
            group_names = [[name] for name in named_flat_params.keys()]
            group_dists = [param.distribution for param in named_flat_params.values()]
            
        return JointDistribution(distributions=group_dists, distribution_names=group_names, param_names=self.flat_param_names())
    
    # ---- Parameter Manipulation --------------------------------------------------            

    def with_params(
        self: Self,
        params: dict[str, Parameter] | dict[str, float] | jnp.ndarray | None = None,
        check_missing: bool = False,
        check_unknown: bool = True,
        fix_others = False,
        include_fixed = False,
        **param_kwargs: dict[str, Parameter] | dict[str, float],
    ) -> Self:
        """Return a new model with parameters updated.

        This is a multi-purpose function that updates parameters differently
        based on the types pass.

        Parameters
        ----------
        params : dict[str, Parameter] | dict[str, float] | jnp.ndarray | None, optional
            Parameter updates. If an array, **all** values must be provided
            (matching ``flat_params`` order). You may also pass keyword args.
        check_missing : bool, default=False
            Require that all model parameters are specified.
        check_unknown : bool, default=True
            Error if unknown parameter keys are provided.
        fix_others : bool, default=False
            Fix any parameters not explicitly passed.
        include_fixed : bool, default=False
            Include fixed parameters when interpreting ``params`` mapping.
        **param_kwargs : dict
            Additional parameter updates by name.

        Returns
        -------
        Self

        Raises
        ------
        Exception
            If shape/order mismatches, unknown/missing names (when checked),
            or if arrays are found outside of Parameters.
        """
        if include_fixed:
            raise Exception('Not yet supported')
        
        # Deal with the sample case i.e. an array-like object
        if not isinstance(params, dict) and len(param_kwargs) == 0:
            params = jnp.array(params)
            if params.shape[0] != self.num_flat_params:
                raise Exception(f'Expected {self.num_flat_params} flat parameters but was passed {params.shape[0]}')
            params_tree, static = self.partition(include_fixed=include_fixed)
            params_out, unravel_fn = flatten_util.ravel_pytree(params_tree)
            
            if jnp.isscalar(params_out) or params_out.shape[0] == 0:
                raise Exception("Error: no free model parameters found to make feature function")
            
            params_tree_recon = unravel_fn(params)
            return combine(params_tree_recon, static)

        params = params if params is not None else {}
        params.update(param_kwargs)
    
        # Generate an ordered, input flat params array for verification
        new_params = self.named_params(include_fixed=True)

        # ---- NEW: Pre-process to handle flattened keys with suffixes (e.g., 'z_real') ----
        # We must identify keys in `params` that are not in `new_params` (parents)
        # but ARE in the flattened view.
        
        parent_keys = set(new_params.keys())
        input_keys = set(params.keys())
        
        # Keys that are not top-level parameters
        potential_flat_keys = input_keys - parent_keys
        
        if potential_flat_keys:
            # Iterate over parents to find which flattened keys belong to them.
            # We only search parents that aren't ALREADY being fully replaced.
            parents_to_scan = [p for p in parent_keys if p not in params]
            
            for parent_name in parents_to_scan:
                parent_param = new_params[parent_name]
                
                # Optimization: only checking multi-dimensional parameters
                if parent_param.size > 0: 
                    # We must replicate the _iter_params name generation logic exactly
                    sub_params = parent_param.flattened(separator=self._separator)
                    
                    updates_found = False
                    new_sub_values = []
                    
                    # Reconstruct the value array from current values + updates
                    for i, sub_p in enumerate(sub_params):
                        suffix = sub_p.name if sub_p.name is not None else str(i)
                        flat_name = f"{parent_name}{self._separator}{suffix}"
                        
                        if flat_name in params:
                            val = params[flat_name]
                            # Handle single-element arrays or scalars
                            if hasattr(val, 'item') and val.size == 1:
                                val = val.item()
                            try:
                                val = float(val)
                            except:
                                raise Exception(f"Value for flat parameter '{flat_name}' must be convertible to float. Got: {val}")
                            new_sub_values.append(val)
                            
                            # Remove the flat key so it doesn't trigger 'unknown parameter' errors
                            del params[flat_name]
                            updates_found = True
                        else:
                            new_sub_values.append(sub_p.value)
                    
                    if updates_found:
                        # Re-assemble the parent parameter
                        new_val_flat = jnp.array(new_sub_values)
                        new_val_shaped = new_val_flat.reshape(parent_param.value.shape)
                        
                        # Update the params dict with the FULL parent object
                        # This ensures it hits the "Case 1" logic in the rest of the function
                        params[parent_name] = dataclasses.replace(parent_param, value=new_val_shaped)            
    
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

        # Convert to an array of parameters instead of floats
        if all(is_convertible_to_float(v) for v in params.values()):            
            for name, value in params.items():
                # TODO create specs for the full parameter objects such that we can get and use the built-in scales
                new_params[name] = dataclasses.replace(new_params[name], value=jnp.array(value))
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

    def with_mapped_params(
        self: Self, 
        mapper: Callable[[Parameter], Parameter], 
        param_filter: str | Sequence[str] | Callable[[str], bool] | None = None, 
        *, 
        map_others: Callable[[Parameter], Parameter] | None = None,
        prefixes: bool = False
    ) -> Self:
        """Return a model with specified parameters mapped.

        Parameters
        ----------
        mapper : Callable[[Parameter], Parameter]
            The map to apply to each parameter in the filter (or all if no filter).
        param_filter : str | Sequence[str] | Callable[[str], bool] | None, default=None
            Parameter names to map. If None, applies mapper to all parameters.
        map_others : Callable[[Parameter], Parameter] | None, default=None
            An optional map to apply to all parameters NOT in the filter.
        prefixes : bool, default=False
            Specifies that, when a string or list of strings is passed
            in `param_filter`, these must be interpreted as parameter prefixes
            to map and not full path names. Defaults to `False.`            

        Returns
        -------
        Self
        """
        current_params = self.named_params()        
        current_param_names = set(current_params.keys())
        
        # NEW: If no filter is provided, target all parameters in the model
        if param_filter is None:
            resolved_filter = current_param_names
        else:
            if isinstance(param_filter, str):
                param_filter = [param_filter]

            if isinstance(param_filter, list) and param_filter and isinstance(param_filter[0], str) and prefixes:
                for prefix in param_filter:
                    if not any(name.startswith(prefix) for name in current_param_names):
                        raise ValueError(f"Specified prefix '{prefix}' does not match any parameters in the model")
                
                valid_prefixes = tuple(param_filter)
                param_filter = lambda p: p.startswith(valid_prefixes)            
                
            if isinstance(param_filter, Callable):
                # Assuming self.param_names() returns a list of valid parameter name strings
                param_filter = [p for p in self.param_names() if param_filter(p)]
            
            resolved_filter = set(param_filter)
            
            for param_name in resolved_filter:
                if param_name not in current_param_names:
                    raise ValueError(f"Specified parameter '{param_name}' not found in model")
        
        new_params = current_params.copy()
        for name, param in current_params.items():
            if name in resolved_filter:
                new_params[name] = mapper(param)
            elif map_others is not None:
                new_params[name] = map_others(param)
                
        return self.with_params(new_params)   
        
    def with_fixed_params(self: Self, param_filter: str | Sequence[str] | Callable[[str], bool], free_others: bool = False, **kwargs) -> Self:
        """Return a model with specified parameters fixed.

        This maps each parameter in the filter, calling :meth:`Parameter.as_fixed` on each.

        See :meth:`.with_mapped_params`.

        Parameters
        ----------
        free_others : bool, default=False
            Also free all parameters not in the filter.        

        Returns
        -------
        Self
        """
        map_others = None
        if free_others:
            map_others = lambda p: p.as_free()

        return self.with_mapped_params(lambda p: p.as_fixed(), param_filter=param_filter, map_others=map_others, **kwargs)
    
    def with_free_params(self: Self, param_filter: str | Sequence[str] | Sequence[Parameter] | Callable[[str], bool], *, fix_others: bool = False, **kwargs) -> Self:
        """Free the specified parameters.

        This maps each parameter in the filter, calling :meth:`Parameter.as_free` on each.

        See :meth:`.with_mapped_params`.

        Parameters
        ----------
        fix_others : bool, default=False
            Also fix all parameters not in the filter.

        Returns
        -------
        Self
        """
        map_others = None
        if fix_others:
            map_others = lambda p: p.as_free()

        return self.with_mapped_params(lambda p: p.as_fixed(), param_filter=param_filter, map_others=map_others, **kwargs)        
    
    def with_free_params_only(self: Self, params: str | list[str] | Callable[[str], bool], **kwargs) -> Self:
        """Returns a model with only the specified parameters freed.
        
        This is an alias for calling :meth:`.`with_free_params``
        with `fix_others=True`.

        See :meth:`.`with_free_params``.
        """
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_free_params_only`.")
        return self.with_free_params(params, **kwargs)

    def with_all_params_fixed(self: Self, **kwargs) -> Self:
        """Returns a model with all parameters fixed.
        
        This is an alias for calling :meth:`.`with_free_params``
        with `fix_others=True` and no parameters passed.

        See :meth:`.`with_free_params``.
        """
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_all_params_fixed`.")
        return self.with_free_params({}, **kwargs)

    def with_all_params_free(self: Self, **kwargs) -> Self:
        """Returns a model with all parameters free.
        
        This is an alias for calling :meth:`.`with_free_params``
        with all parameters passed.

        See :meth:`.`with_free_params``.
        """
        return self.with_free_params(self.param_names(include_fixed=True), **kwargs)
    
    # ---- Parameter group manipulation --------------------------------------------------            
    
    def with_param_groups(self: Self, param_groups: ParameterGroup | list[ParameterGroup]) -> Self:
        """Return a model with parameter groups appended, replacing existing relationships.
        
        This method implements an "atomic replacement" policy. If *any* parameter in 
        an existing group is claimed by a new group, the *entire* existing group is 
        removed. 
        
        This ensures that groups defining joint distributions are not left in an 
        invalid broken state (e.g. having a dimension removed). Parameters that were 
        in the removed group but not in the new group will revert to being ungrouped 
        (handled by `param_groups` as singleton groups).

        Parameters
        ----------
        param_groups : ParameterGroup or list[ParameterGroup]
            Group(s) to add.

        Returns
        -------
        Self
        """       
        if not isinstance(param_groups, list):
            param_groups = [param_groups]
        
        # 1. Identify all parameter names claimed by the NEW groups
        new_claimed_params = set()
        for group in param_groups:
            # Assumes the field name is 'param_names' per your previous context
            new_claimed_params.update(group.param_names)

        # 2. Filter OLD groups (Atomic check)
        current_groups = self._param_groups if self._param_groups is not None else []
        kept_existing_groups = []
        
        for group in current_groups:
            # Check for intersection: Does this existing group contain ANY parameter 
            # that is now being redefined in the new groups?
            existing_group_params = set(group.param_names)
            
            if existing_group_params.isdisjoint(new_claimed_params):
                # No conflict: Keep this group entirely
                kept_existing_groups.append(group)
            else:
                # Conflict found: Discard this group entirely. 
                # Note: Parameters in this group that were NOT in 'new_claimed_params' 
                # are now effectively "released" and will be treated as singletons 
                # by the main param_groups() method.
                pass

        # 3. Combine
        new_list = kept_existing_groups + param_groups
        
        new_model = copy(self)
        object.__setattr__(new_model, '_param_groups', new_list)
        return new_model
    
    def with_demoted_param_groups(self: Self) -> Self:
        """Recursively demote parameter groups to the deepest possible submodel.

        This method identifies parameter groups where every parameter belongs to the same 
        immediate submodel. It moves those groups to the submodel, stripping the prefix.
        It then recursively calls this method on the submodels to ensure groups continue 
        moving down the hierarchy as far as possible.

        Returns
        -------
        Self
            A new model instance with parameter groups distributed to their lowest 
            relevant submodels.
        """
        # 1. Identify immediate submodels and their prefixes
        submodel_prefixes = {} 
        for f in dataclasses.fields(self):
            if isinstance(getattr(self, f.name), Model):
                prefix = f.name + self._separator
                submodel_prefixes[prefix] = f.name

        # 2. Sort current groups into "keep" (stay here) or "demote" (move to child)
        groups_to_keep = []
        submodel_groups = {name: [] for name in submodel_prefixes.values()}
        
        current_groups = self._param_groups if self._param_groups is not None else []

        for group in current_groups:
            demoted = False
            for prefix, field_name in submodel_prefixes.items():
                # Check if ALL parameters in the group belong to this submodel
                if all(name.startswith(prefix) for name in group.param_names):
                    # Strip prefix
                    new_names = [name[len(prefix):] for name in group.param_names]
                    new_group = dataclasses.replace(group, param_names=new_names)
                    submodel_groups[field_name].append(new_group)
                    demoted = True
                    break
            
            if not demoted:
                groups_to_keep.append(group)

        # 3. Apply updates to submodels AND recurse
        new_fields = {}
        
        # We iterate over all submodels (even if they didn't receive new groups from us)
        # because they might have their *own* local groups that need demoting further down.
        for prefix, field_name in submodel_prefixes.items():
            child_model: Model = getattr(self, field_name)
            
            # A. Push: Add the groups we demoted from the current level
            groups_to_push = submodel_groups[field_name]
            if groups_to_push:
                child_model = child_model.with_param_groups(groups_to_push)
            
            # B. Recurse: Ask the child to demote its groups (including the ones we just pushed)
            child_model = child_model.with_demoted_param_groups()
            
            new_fields[field_name] = child_model

        # 4. Return updated model
        new_model = dataclasses.replace(self, **new_fields)
        object.__setattr__(new_model, '_param_groups', groups_to_keep)
        return new_model
    
    def with_no_param_groups(self: Self) -> Self:
        """Return a new model with all parameter groups removed recursively.

        This clears the `_param_groups` of the current model and traverses
        all nested submodels (and sequences of submodels) to remove their 
        parameter groups as well.

        Returns
        -------
        Self
            A new model instance with no parameter groups.
        """
        new_fields = {}  # Removed '_param_groups': [] from the initialization
        
        for f in dataclasses.fields(self):
            # Skip the target field since we handle it at the end
            if f.name == '_param_groups':
                continue
                
            child = getattr(self, f.name)
            
            # 1. Recurse into direct submodels
            if isinstance(child, Model):
                new_fields[f.name] = child.with_no_param_groups()
                
            # 2. Recurse into sequences of submodels (e.g., in composites like Cascade)
            elif isinstance(child, (list, tuple)):
                # Only process the sequence if it actually contains at least one Model
                if any(isinstance(x, Model) for x in child):
                    new_fields[f.name] = type(child)(
                        x.with_no_param_groups() if isinstance(x, Model) else x 
                        for x in child
                    )
                    
        new_model = dataclasses.replace(self, **new_fields)
        object.__setattr__(new_model, '_param_groups', [])
        return new_model
    
    # ---- Distribution manipulation --------------------------------------------------

    def with_mapped_distributions(
        self: Self, 
        mapper: Callable[[Distribution], Distribution], 
        dist_filter: Callable[[Distribution], bool] | None = None, 
        *, 
        map_others: Callable[[Distribution], Distribution] | None = None,
        param_groups: bool = False
    ) -> Self:
        """Return a model with a function applied to its parameter distributions.

        This method allows for bulk-updates of distributions, such as widening variances 
        or changing distribution types.

        If ``param_groups`` is False, the mapping is applied to the distributions 
        of individual parameters (flattened).

        If ``param_groups`` is True, the mapping is applied to the distributions 
        of :class:`ParameterGroup` objects. This mode is recursive: it will traverse 
        the model tree and apply the mapping to all explicit parameter groups in all submodels.

        Parameters
        ----------
        mapper : Callable[[Distribution], Distribution]
            Function that takes a distribution and returns a new one.
        dist_filter : Callable[[Distribution], bool] | None, default=None
            A predicate function. If provided, the mapping is only applied to 
            distributions where ``dist_filter(dist)`` is True. If None, applies to all.
        map_others : Callable[[Distribution], Distribution] | None, default=None
            An optional map to apply to all distributions NOT in the filter.
        param_groups : bool, default=False
            If True, map distributions on parameter groups (recursively). 
            If False, map distributions on individual parameters (flat).

        Returns
        -------
        Self
            A new model with updated distributions.
        """
        mapped_model = self

        if param_groups:
            # 1. Map Local Groups (Current Level)
            current_groups = self._param_groups if self._param_groups is not None else []
            for group in current_groups:
                if dist_filter is None or dist_filter(group.distribution):
                    mapped_model = mapped_model.with_param_groups(group.with_distribution(mapper(group.distribution)))
                elif map_others is not None:
                    mapped_model = mapped_model.with_param_groups(group.with_distribution(map_others(group.distribution)))

            # 2. Recurse into Submodels
            new_submodels = {}
            for f in dataclasses.fields(mapped_model):
                child = getattr(mapped_model, f.name)
                # Check if the field is a direct submodel
                if isinstance(child, Model):
                    # Recursive call
                    updated_child = child.with_mapped_distributions(
                        mapper, 
                        dist_filter, 
                        map_others=map_others, 
                        param_groups=True
                    )
                    new_submodels[f.name] = updated_child
            
            # Apply submodel updates if any
            if new_submodels:
                mapped_model = dataclasses.replace(mapped_model, **new_submodels)

        else:
            # 3. Existing logic for individual params (Global via named_params)
            new_params = {}
            for name, param in self.named_params().items():
                if dist_filter is None or dist_filter(param.distribution):
                    new_params[name] = param.with_distribution(mapper(param.distribution))
                elif map_others is not None:
                    new_params[name] = param.with_distribution(map_others(param.distribution))
            
            # Apply all parameter updates at once
            if new_params:
                mapped_model = mapped_model.with_params(new_params)
                    
        return mapped_model
    
    def with_uniform_distributions(self, percentage: float, param_filter: str | Sequence[str] | Callable[[str], bool] = None, *, respect_bounds=False, remove_param_groups=True):
        """Return a model with uniform distributions set centered on current parameter values.

        The distributions are defined with bounds calculated as ``value * (1.0 +/- percentage)``.

        Parameters
        ----------
        percentage : float
            The fractional width of the uniform distribution (e.g. 0.1 = 10%).
        filter: str | Sequence[str] | Callable[[str], bool], default=None
            The parameters to updated with new uniform distributions. For the default case, all are updated.
        respect_bounds: bool, default=False
            Whether or not the `min` and `max` bounds of the current distributions should be respected.
            If `True`, new bounds will not go larger than past these bounds.
        remove_param_groups: bool, default=True
            Whether to remove parameter groups recursively when setting the uniform distributions.
            Otherwise, the joint distribution of the model may not be the desired uniform distribution.

        Returns
        -------
        Self
            A new model with updated parameter distributions.
        """        
        updates = {}
        for name, param in self.named_params(param_filter).items():
            new_min = param * (1.0 - percentage) / param.scale
            new_max = param * (1.0 + percentage) / param.scale

            if respect_bounds:
                new_min = max(new_min, param.min)
                new_max = min(new_max, param.max)

            distribution = UniformDistribution(new_min, new_max)
            updates[name] = param.with_distribution(distribution)
            
        new_model = self.with_params(updates)
        if remove_param_groups:
            new_model = new_model.with_no_param_groups()
        return new_model
    
    # ---- Field and model manipulation --------------------------------------------------            
    
    @classmethod
    def with_defaults(cls, *args, **kwargs) -> type[Self]:
        """Return this model type with default initialization arguments.
        
        This method is very useful in utilizing an existing model
        with default values, without having to create a new
        model type via inheritance.

        Arguments are forwarded as if they were passed to `__init__`.

        Returns
        -------
        type[Model]
        """            
        class DefaultsWrapper:
            def __init__(self, p):
                self.p = p   # underlying partial

            def __call__(self, *args, **kwargs):
                return self.p(*args, **kwargs)

            # chaining
            def with_defaults(self, *args, **kwargs):
                # merge new defaults after existing ones
                new_args = self.p.args + args
                new_kwargs = {**self.p.keywords, **kwargs} if self.p.keywords else kwargs
                return DefaultsWrapper(partial(self.p.func, *new_args, **new_kwargs))
        return DefaultsWrapper(partial(cls, *args, **kwargs))
    
    def with_models(self: Self, models: Self | Sequence[Self]) -> Self:
        """Combines this model with free parameters in other models.
        
        This is useful to combine separate models obtained from fitting
        the same initial model with different free parameters.

        Parameters
        ----------
        models : Model or Sequence[Model]
            The other models to combine this model with.

        Returns
        -------
        Model
        """  
        if not isinstance(models, Sequence):
            models = [models]

        combined = self
        for other in models:
            combined = combined.with_params(other.named_params())
            combined = combined.with_param_groups(other._param_groups)
        return combined
    
    def with_fields(self: Self, *args, **kwargs) -> Self:
        """
        Return a copy of this model with dataclass-style field replacements.

        Parameters are forwarded to :func:`dataclasses.replace`.
        """
        return dataclasses.replace(self, *args, **kwargs)
    
    def with_name(self: Self, name: str | None) -> Self:
        """
        Return a copy of this model with a different name.
        """
        return dataclasses.replace(self, name=name)
    
    def with_submodel_fields(self: Self, submodel: str | Sequence[str], *args, **kwargs) -> Self:
        """
        Return a copy of this model with dataclass-style field replacements on a nested sub-model.

        Parameters are forwarded to :func:`dataclasses.replace`.

        Parameters
        ----------
        submodel : str | Sequence[str]
            The name of the submodel (or sequence of names) to traverse.
            Can be a single string with a path e.g. 'submodel1.submodel2',
            or a list of submodels e.g. ['submodel1', 'submodel2'].
        """
        # Normalize input to a list of strings
        if isinstance(submodel, str) and submodel.find('.'):
            path = submodel.split('.')
        else:
            path = [submodel] if isinstance(submodel, str) else list(submodel)
        
        if not path:
            # If path is empty, apply fields to self (standard with_fields behavior)
            return self.with_fields(*args, **kwargs)

        target_key = path[0]

        if len(path) == 1:
            # Base case: We are at the parent of the final target submodel
            updated_child = getattr(self, target_key).with_fields(*args, **kwargs)
        else:
            # Recursive step: Tell the child to handle the rest of the path
            child = getattr(self, target_key)
            updated_child = child.with_submodel_fields(path[1:], *args, **kwargs)

        # Return a copy of 'self' with the new version of the child
        return self.with_fields(**{target_key: updated_child})    
    
    def with_free_submodels(self: Self, submodels: 'Model' | Sequence['Model'] | str | Sequence[str], include_fixed=False, fix_others=False) -> Self:
        """Free all parameters in the given submodels.

        Submodels parameters are obtained using :meth:`.param_names`.,
        and subsequently freed using :meth:`.`with_free_params``.
        
        Parameters
        ----------
        submodels : Model | Sequence[Model] | str | Sequence[str]
            Submodels whose parameters should be free.
        include_fixed : bool, default=False
            Also free parameters that are currently fixed in the submodels.
        fix_others : bool, default=False
            Fix all other submodels.

        Returns
        -------
        Self
        """        
        model_param_names = self.param_names(include_fixed=include_fixed, submodels=submodels)
        return self.with_free_params(model_param_names, fix_others=fix_others)

    def with_free_submodels_only(self: Self, *args, **kwargs) -> Self:
        """Returns a model with only the specified submodels freed.
        
        This is an alias for calling :meth:`.`with_free_submodels``
        with `fix_others=True`.

        See :meth:`.`with_free_params``.
        """     
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_free_submodels_only`.")
        return self.with_free_submodels(*args, **kwargs)
    
    def with_fixed_submodels(self: Self, submodels: 'Model' | Sequence['Model'] | str | Sequence[str]) -> Self:
        """Fix all parameters in the given submodels.

        Submodels parameters are obtained using :meth:`.param_names`.,
        and subsequently fixed using :meth:`.`with_fixed_params``.
        
        Parameters
        ----------
        submodels : Model | Sequence[Model] | str | Sequence[str]
            Submodels whose parameters should be fixed.

        Returns
        -------
        Self
        """        
        model_param_names = self.param_names(submodels=submodels)
        return self.with_fixed_params(model_param_names)
    
    def with_tied_submodels(
        self: Self, 
        submodel_attrs: str | Sequence[str], 
        shared_model: 'Model'
    ) -> Self:
        """
        Return a copy of the model with specified submodels structurally tied to a shared model.

        This method prepares submodels to act as structural proxies during optimization. 
        It prevents the optimizer from seeing duplicate free parameters by fixing the target 
        submodel's parameters. If the target submodel's type matches the shared model, 
        only the overlapping free parameters are fixed. If the types differ, the target 
        submodel is entirely replaced by a fully fixed copy of the shared model.

        This is typically used in `__post_init__` to set up the model structure, 
        and paired with :meth:`.with_injected_params` during the forward pass.

        Parameters
        ----------
        submodel_attrs : str | Sequence[str]
            The attribute name(s) of the internal submodel(s) to tie.
        shared_model : Model
            The external model to tie the submodels to.

        Returns
        -------
        Self
            A new model instance with the specified submodels tied.
        """
        if isinstance(submodel_attrs, str):
            submodel_attrs = [submodel_attrs]
            
        modified_self = self
        for attr in submodel_attrs:
            current_submodel = getattr(modified_self, attr)
            
            # If types match, fix the target's parameters that match the shared model's FREE parameters
            if isinstance(current_submodel, type(shared_model)):
                shared_free_params = shared_model.param_names()
                params_to_fix = [f"{attr}{modified_self._separator}{p}" for p in shared_free_params]
                modified_self = modified_self.with_fixed_params(params_to_fix)
            else:
                # If types differ, completely replace the target with a fully fixed version of the shared model
                modified_self = modified_self.with_fields(**{attr: shared_model.with_all_params_fixed()})
                
        return modified_self
    
    def tied(
        self: Self, 
        shared_model: 'Model'
    ) -> 'Model':
        """
        Return the model with self structurally tied to a shared model.
        
        Same as `with_tied_submodels` but operates directly on self.

        Parameters
        ----------
        shared_model : Model
            The external model to tie the submodels to.

        Returns
        -------
        Self
            A new model instance with self tied to the specified model.
        """
        if isinstance(self, type(shared_model)):
            return self.with_fixed_params(shared_model.param_names())
        else:
            return shared_model.with_all_params_fixed()

    def with_injected_params(
        self: Self, 
        submodel_attrs: str | Sequence[str], 
        shared_model: 'Model'
    ) -> Self:
        """
        Return a copy of the model with free parameters from a shared model injected into target submodels.

        This method dynamically overrides the parameter values of internal submodels 
        using the values from an external shared model. It is designed to be called 
        during the forward pass (e.g., inside `__call__`) to enforce hard equality 
        constraints on parameters that were structurally tied using :meth:`.with_tied_submodels`.

        Parameters
        ----------
        submodel_attrs : str | Sequence[str]
            The attribute name(s) of the internal submodel(s) to inject parameters into.
        shared_model : Model
            The external model providing the free parameter values.

        Returns
        -------
        Self
            A new model instance with the updated parameter values injected.
        """
            
        if isinstance(submodel_attrs, str):
            submodel_attrs = [submodel_attrs]
            
        modified_self = self
        for attr in submodel_attrs:
            # Prefix the shared model's free parameters with the target submodel's attribute name
            injected_params = {
                f"{attr}{modified_self._separator}{k}": v 
                for k, v in shared_model.named_params().items()
            }
            modified_self = modified_self.with_params(injected_params)
            
        return modified_self
    
    # ---- Plotting --------------------------------------------------    

    def plot_func(
        self,
        func: Callable[['Model', Frequency], jnp.ndarray],
        freq: Frequency,
        *,
        ax = None,
        label: str | None = None,
        color: str | None = None,
        **kwargs
    ):
        """Evaluate and plot an arbitrary function of the current model.

        This method evaluates the provided function using the model's current 
        parameter values and plots the resulting response over frequency.

        Parameters
        ----------
        func : Callable[[Model, Frequency], jnp.ndarray]
            Function to evaluate. Must take a Model and a Frequency object and 
            return a jnp.ndarray of shape (n_freqs,).
        freq : Frequency
            Frequency grid to evaluate over.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, the current axes (`plt.gca()`) are used.
        label : str, optional
            Label for the plotted line (used in legends).
        color : str, optional
            Color for the line. If None, uses the matplotlib color cycle.
        **kwargs : dict
            Additional keyword arguments forwarded to `matplotlib.pyplot.plot` 
            (e.g., `linestyle`, `linewidth`, `alpha`).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            ax = plt.gca()

        # 1. Evaluate the function on the current model
        y_val = func(self, freq)
        y_val = np.asarray(y_val)
        
        # Extract the x-axis automatically from the frequency object
        x_axis = np.asarray(freq.f_scaled) 

        # 2. Plotting logic
        # Assemble kwargs safely to avoid passing multiple 'color' or 'label' arguments
        plot_kwargs = kwargs.copy()
        if label is not None:
            plot_kwargs['label'] = label
        if color is not None:
            plot_kwargs['color'] = color

        ax.plot(x_axis, y_val, **plot_kwargs)
        
        return ax    

    def plot_func_samples(
        self,
        func: Callable[['Model', Frequency], jnp.ndarray],
        freq: Frequency,
        *,
        key: jax.Array,
        num_samples: int = 1000,
        contours: bool = True,
        ax = None,
        label: str | None = None,
        color: str = 'C0',
        alpha: float = 0.1,
    ):
        """Evaluate and plot a function over samples from the parameter distribution.

        This method draws samples from the model's joint parameter distribution, 
        evaluates the provided function for each sample, and plots the resulting 
        responses over frequency.

        Parameters
        ----------
        func : Callable[[Model, Frequency], jnp.ndarray]
            Function to evaluate. Must take a Model and a Frequency object and 
            return a jnp.ndarray of shape (n_freqs,).
        freq : Frequency
            Frequency grid to evaluate over.
        key : jax.Array
            PRNG key for sampling the distribution.
        num_samples : int, default=1000
            Number of samples to draw.
        contours : bool, default=True
            If True, plots the mean response and filled contours corresponding 
            to 1, 2, and 3 standard deviations. If False, plots all individual 
            sample responses as transparent lines.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, the current axes (`plt.gca()`) are used.
        label : str, optional
            Label for the mean line (used in legends).
        color : str, default='C0'
            Base color for the lines and shaded regions.
        alpha : float, default=0.1
            Transparency of the individual lines (when `contours=False`). 

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            ax = plt.gca()

        # 1. Evaluate the ensemble
        y_samples = self.func_samples(func, freq, key, num_samples)
        y_samples = np.asarray(y_samples)
        
        # Extract the x-axis automatically from the frequency object
        x_axis = np.asarray(freq.f_scaled) 

        # 2. Calculate central tendency
        y_mean = np.mean(y_samples, axis=0)

        # 3. Plotting logic
        if not contours:
            # Plot all individual samples
            # Transpose y_samples so matplotlib interprets columns as individual lines
            ax.plot(x_axis, y_samples.T, color=color, alpha=alpha)
            # Plot the mean as a solid line on top
            ax.plot(x_axis, y_mean, color=color, label=label, linewidth=2)
            
        else:
            # Plot mean line
            ax.plot(x_axis, y_mean, color=color, label=label, linewidth=2)
            
            # Plot contours for 1, 2, and 3 standard deviations
            y_std = np.std(y_samples, axis=0)
            
            # Decreasing opacity for outer standard deviations
            for i, sig_alpha in zip([1, 2, 3], [0.3, 0.2, 0.1]):
                ax.fill_between(
                    x_axis, 
                    y_mean - i * y_std, 
                    y_mean + i * y_std, 
                    color=color, 
                    alpha=sig_alpha, 
                    linewidth=0
                )
        
        return ax
        
    
    # ---- File and conversion utilities  --------------------------------------------------            
    
    def to_skrf(self, frequency: Frequency | skrf.Frequency, sigma=0.0, **kwargs) -> skrf.Network:
        """Convert the model at frequencies to an :class:`skrf.Network`.

        The active primary property (``self.primary_property``) is used.

        Parameters
        ----------
        frequency : pmrf.Frequency | skrf.Frequency
            Frequency grid.
        sigma : float, default=0.0
            If nonzero, add complex Gaussian noise with stdev ``sigma`` to ``s``.
        **kwargs
            Forwarded to :class:`skrf.Network` constructor.

        Returns
        -------
        skrf.Network
        """
        if isinstance(frequency, Frequency):
            model_freq = frequency
            measured_freq = frequency.to_skrf()
        else:
            model_freq = Frequency.from_skrf(frequency)
            measured_freq = frequency
        
        fval, fname = self.primary(model_freq), self.primary_property
        kwargs = kwargs or {}
        kwargs.update({
            fname: fval,
            'frequency': measured_freq,
            'name': kwargs.get('name', self.name),
            'z0': self.z0,
        })
        ntwk = skrf.Network(**kwargs)
        if sigma != 0.0:
            ntwk.s += (np.random.normal(0, sigma, ntwk.s.shape) + 1j * np.random.normal(0, sigma, ntwk.s.shape))
        return ntwk        
    
    def export_touchstone(self, filename: str, frequency: Frequency | skrf.Frequency, sigma: float = 0.0, **skrf_kwargs):
        """Export the model response to a Touchstone file via scikit-rf.

        Parameters
        ----------
        filename : str
        frequency : Frequency | skrf.Frequency
        sigma : float, default=0.0
            Additive complex noise std for S-parameters.
        **skrf_kwargs
            Forwarded to :meth:`skrf.Network.write_touchstone`.

        Returns
        -------
        Any
            Return value of ``Network.write_touchstone``.
        """        
        if not isinstance(filename, str):
            raise Exception('Filename must be a string')
        
        ntwk = self.to_skrf(frequency, sigma=sigma)
        retval = ntwk.write_touchstone(filename, **skrf_kwargs)
        return retval
    
Model.__module__ = "pmrf"