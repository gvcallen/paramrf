"""pmrf.model
================

Core model base class and utilities for ParamRF.

This module defines :class:`Model`, a frozen, JAX-compatible, Equinox module
representing an RF network, along with helper utilities like :func:`wrap`.

"""

from functools import cached_property, partial
from copy import deepcopy
from typing import Callable, Sequence, TypeVar, Iterator, Self
import dataclasses
from dataclasses import fields, is_dataclass
from functools import update_wrapper
import jax.numpy as jnp
import numpy as np
import jsonpickle
from collections.abc import Sequence
from typing import Sequence, Callable, BinaryIO, TypeVar
import os

import skrf as skrf
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import PyTree
from jax import flatten_util
from jax.tree_util import SequenceKey, GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey
import equinox as eqx
from numpyro.distributions import Distribution, Uniform as UniformDistribution

from pmrf.network_collection import NetworkCollection
from pmrf.functions.conversions import a2s, s2a, s2z, z2s, s2y, y2s
from pmrf.functions.math import FUNC_LOOKUP
from pmrf.parameters import Parameter, ParameterGroup, is_valid_param, as_param
from pmrf.distributions.parameter import JointParameterDistribution
from pmrf.constants import PRIMARY_PROPERTIES
from pmrf.frequency import Frequency
from pmrf._util import field, classproperty, is_overridden, get_first_underlying_type, is_convertible_to_float
from pmrf._tree import nodes_by_type, value_at_path, partition, combine

class Model(eqx.Module):
    """
    Overview
    --------
    This base class is used to represent any computable RF network, referred to in
    **ParamRF** as a "Model". This class is abstract and should not be
    instantiated directly. Derive from :class:`Model` and override one of
    the primary property functions (e.g. :meth:`__call__`, :meth:`s`, :meth:`a`).

    The model is an Equinox ``Module`` (immutable, dataclass-like) and is
    treated as a JAX PyTree. Parameters are declared using standard dataclass
    field syntax with types like :class:`pmrf.Parameter`.

    Usage
    -----
    - Define new models by sub-classing the model and adding custom parameters and/or sub-models
    - Construct models by passing parameters and/or submodels to the initializer (like a dataclass).
    - Retrieve parameter information via methods such as :meth:`named_params`, :meth:`param_names`, :meth:`flat_params`, etc..
    - Use `with_xxx` functions to modify fields, models and parameters within the model e.g. :meth:`with_params`, :meth:`with_fields`.
    - Use "past tense" functions to modify the model in conjuction with another model or data e.g. :meth:`terminated`, :meth:`fitted`.

    See also the :mod:`pmrf.fitting` and :mod:`pmrf.sampling` modules for details on model fitting and sampling.

    Attributes
    ----------
    name : str or None
        An optional name for the model instance.
    separator : str
        The separator character used when flattening nested parameter names (default is '_').
    metadata : dict
        A dictionary for storing arbitrary metadata associated with the model (e.g., fit results).

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
    # Public instance fields
    name: str | None = field(default=None, kw_only=True, static=True)
    separator: str = field(default='_', kw_only=True, static=True)
    metadata: dict = field(default_factory=lambda: dict(), kw_only=True, static=True)
    
    # Private instance fields
    _z0: complex = field(default=50.0+0j, kw_only=True, static=True)
    _param_groups: list[ParameterGroup] = field(default_factory=lambda: list(), kw_only=True, repr=False, static=True)

    # ---- Internal initialization methods -------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        """Customize subclass construction.

        - Applies default names to default ``Model``/``Parameter`` fields.
        - Adds converters for ``Parameter``-typed fields (``asparam``).
        - Protects against mutable-default pitfalls by using ``default_factory``.
        - Dynamically injects convenience methods like ``s_mag``, ``s_mn_mag``,
          based on :data:`PRIMARY_PROPERTIES` and :data:`FUNC_LOOKUP`.
        """        
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
                
                field_kwargs['converter'] = lambda x, field_name=field_name: as_param(x, name=field_name, fixed=True)
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
                
                m = make_dynamic_method(prop, func)
                m._pmrf_auto = True
                setattr(cls, func_name, m)
                
                # Then the index function function e.g. s_mn_mag
                func_name = f"{prop}_mn_{suffix}"
                func = lookup[1]
                m = make_dynamic_method(f"{prop}_mn", func)
                m._pmrf_auto = True
                setattr(cls, func_name, m)

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
    
    def _path_to_param_name(self, path) -> str | list[str]:
        """Convert a PyTree path to a fully-qualified parameter name."""
        fields = []

        for i, item in enumerate(path):
            if isinstance(item, GetAttrKey):
                fields.append(item.name)
            elif isinstance(item, DictKey):
                fields.pop() # don't include the dict variable name in the path
                fields.append(item.key)
            elif isinstance(item, SequenceKey) or isinstance(item, FlattenedIndexKey):
                # This code adds the ability for models to be named and their names used for the parameter name instead of the list variable name and index
                value = value_at_path(self, path[:i+1])
                if hasattr(value, 'name') and not value.name is None:
                    fields.pop()
                    fields.append(value.name)
                else:
                    fields.append(str(item.idx))
        return self.separator.join(fields)
    
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
                # 1. Recursively clean all fields first
                new_fields = {}
                for field in dataclasses.fields(obj):
                    # Skip metadata, we will overwrite it later
                    if field.name == 'metadata':
                        continue
                    
                    # Recurse on the field's value
                    val = getattr(obj, field.name)
                    new_fields[field.name] = strip_metadata_recursive(val, memo)
                
                # 2. Create the new instance with updated children
                # using dataclasses.replace (standard) or obj.with_fields (your custom method)
                new_obj = dataclasses.replace(obj, **new_fields)
                
                # 3. Clear the metadata for this specific module
                if hasattr(new_obj, "metadata"):
                    new_obj = dataclasses.replace(new_obj, metadata=dict())
                    
                memo[obj_id] = new_obj
                return new_obj

            # Case 2: It's a list/tuple (standard container)
            elif isinstance(obj, (list, tuple)):
                # Reconstruct the list/tuple with cleaned items
                new_obj = type(obj)(strip_metadata_recursive(x, memo) for x in obj)
                memo[obj_id] = new_obj
                return new_obj

            # Case 3: It's a dict
            elif isinstance(obj, dict):
                new_obj = {k: strip_metadata_recursive(v, memo) for k, v in obj.items()}
                memo[obj_id] = new_obj
                return new_obj

            # Case 4: It's a leaf (int, string, array, etc.) - leave as is
            else:
                return obj
            
        return strip_metadata_recursive(self)
    
    def _saveable(self: Self) -> Self:
        return self._with_stripped_metadata()
    
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
    
    def _iter_params(
        self,
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
                    flattened_params = param.flattened(separator=self.separator)
                    for i, subparam in enumerate(flattened_params):
                        suffix = subparam.name if subparam.name is not None else str(i)
                        flat_params.append((f"{name}{self.separator}{suffix}", subparam))
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
    
    def copy(self: Self) -> Self:
        """Returns a deepcopy of self.

        Returns
        -------
        Model
        """        
        return deepcopy(self)    

    # ---- Introspection properties --------------------------------------------------------
    
    @cached_property
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
    def z0(self) -> jnp.ndarray:
        """Internal characteristic impedance.

        Returns
        -------
        jnp.ndarray
        """
        return jnp.array(self._z0)
    
    @cached_property
    def num_params(self) -> int:
        """Number of free parameters.

        Returns
        -------
        int
        """
        return len(self.params())

    @cached_property
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
        raise NotImplementedError(f"Error: cannot use the __call__ method to build a model in the Model class directly")
    
    @eqx.filter_jit
    def primary(self, freq: Frequency) -> jnp.ndarray:
        """Dispatch to the primary function for the given frequency."""        
        primary_function = self.primary_function
        return primary_function(freq)
    
    @eqx.filter_jit
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Scattering parameter matrix.

        If a different parameter type (a, z, y) is primary, this converts it to S.

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

        prioritized = () # for future expansion
        unprioritized = tuple(p for p in PRIMARY_PROPERTIES if p not in prioritized)

        for prop in prioritized + unprioritized:
            if prop == 's': continue
            if is_overridden(type(self), Model, prop):
                val = getattr(self, prop)(freq)
                if prop == 'a': return a2s(val, self.z0)
                if prop == 'z': return z2s(val, self.z0)
                if prop == 'y': return y2s(val, self.z0)
        
        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overriden, which are the only ones supported currently")

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
        
        prioritized = () # for future expansion
        unprioritized = tuple(p for p in PRIMARY_PROPERTIES if p not in prioritized)

        for prop in prioritized + unprioritized:
            if prop == 'a': continue
            if is_overridden(type(self), Model, prop):
                val = getattr(self, prop)(freq)
                if prop == 's': return s2a(val, self.z0)
                if prop == 'z': return s2a(z2s(val, self.z0), self.z0)
                if prop == 'y': return s2a(y2s(val, self.z0), self.z0)

        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overriden, which are the only ones supported currently")

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

        prioritized = () # for future expansion
        unprioritized = tuple(p for p in PRIMARY_PROPERTIES if p not in prioritized)

        for prop in prioritized + unprioritized:
            if prop == 'z': continue
            if is_overridden(type(self), Model, prop):
                val = getattr(self, prop)(freq)
                if prop == 's': return s2z(val, self.z0)
                if prop == 'a': return s2z(a2s(val, self.z0), self.z0)
                if prop == 'y': return s2z(y2s(val, self.z0), self.z0)

        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overriden, which are the only ones supported currently")

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

        prioritized = () # for future expansion
        unprioritized = tuple(p for p in PRIMARY_PROPERTIES if p not in prioritized)

        for prop in prioritized + unprioritized:
            if prop == 'y': continue
            if is_overridden(type(self), Model, prop):
                val = getattr(self, prop)(freq)
                if prop == 's': return s2y(val, self.z0)
                if prop == 'a': return s2y(a2s(val, self.z0), self.z0)
                if prop == 'z': return s2y(z2s(val, self.z0), self.z0)

        raise NotImplementedError(f"No primary properties in {PRIMARY_PROPERTIES} are overriden, which are the only ones supported currently")
    
    # ---- Structure utilities --------------------------------------------------    
    
    def children(self) -> list['Model']:
        """Immediate submodels.

        Returns
        -------
        list[Model]
        """
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Model)]
    
    def submodels(self) -> list['Model']:
        """All nested submodels (depth-first), excluding ``self``.

        Returns
        -------
        list[Model]
        """
        return nodes_by_type(self, Model)[1:]
    
    # ---- Magic methods --------------------------------------------------

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
        """Cascade composition operator ``**``."""        
        from pmrf.models.containers import Cascade
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
        
    # ---- Container model building --------------------------------------------------    
    
    def connected(self, others: 'Model' | Sequence['Model'], ports: Sequence[int | Sequence[int]]) -> 'Model':
        """Return a version of the model with specified ports connected.

        See :class:``pmrf.models.containers.Connected``.

        Returns
        -------
        Model
        """        
        from pmrf.models.containers import Connected
        if isinstance(others, Model):
            others = [others]
        models = [self, *others]
        return Connected(models, ports)

    def flipped(self) -> 'Model':
        """Return a version of the model with ports flipped.

        Returns
        -------
        Model
        """
        from pmrf.models.containers import Flipped
        if isinstance(self, Flipped):
            return self.model
        return Flipped(self)
    
    def terminated(self, load: 'Model' = None) -> 'Model':
        """Returns a new model that contains this 2-port model terminated in a 1-port load (default: SHORT).

        Parameters
        ----------
        load : Model, optional
            Load network. Defaults to a SHORT.

        Returns
        -------
        Model
        """
        from pmrf.models.lumped import SHORT
        from pmrf.models.containers import Cascade
        load = load or SHORT
        terminated_model = Cascade((self, load))
        return terminated_model
       
    # ---- Parameter inspection -------------------------------------------------- 
    
    def named_params(self, include_fixed=False, submodels: 'Model' | Sequence['Model'] | str | Sequence[str] | None = None) -> dict[str, Parameter]:
        """Named model parameters as a dict.

        Keys are fully-qualified parameter names.
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
        return dict(self._iter_params(include_fixed=include_fixed, submodels=submodels))
    
    def named_param_values(self, scaled=False, **kwargs) -> dict[str, jnp.ndarray]:
        """Named model parameter values as a dict of jax arrays.

        See :meth:`named_params`.

        Parameters
        ----------
        scaled : bool, default=False
            Whether or not to scale the returned values by the parameter scales.
        **kwargs
            Additional key-word arguments as in  :meth:`named_params`.

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

        See :meth:`named_params`.
        """
        return list(self.named_params(*args, **kwargs).keys())

    def param(self, name: str, *args, **kwargs) -> Parameter:
        """
        Return a single model parameter by name.

        See :meth:`named_params`.
        """
        return self.named_params(*args, **kwargs)[name]
    
    def params(self, *args, **kwargs) -> list[Parameter]:
        """
        Return model parameters as a list.

        See :meth:`named_params`.
        """
        return list(self.named_params(*args, **kwargs).values())
    
    def param_value(self, name: str, *args, **kwargs) -> jnp.ndarray:
        """
        Return a single model parameter value by name as a single jax array.

        See :meth:`named_param_values`.
        """
        return self.named_param_values(*args, **kwargs)[name]

    def param_values(self, *args, **kwargs) -> list[jnp.ndarray]:
        """
        Return model parameter values as a list of jax arrays.

        See :meth:`named_param_values`.
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

        See :meth:`named_flat_params`.

        Parameters
        ----------
        scaled : bool, default=False
            Whether or not to scale the returned values by the parameter scales.
        **kwargs
            Additional key-word arguments as in  :meth:`named_params`.

        Returns
        -------
        dict[str, jnp.ndarray]
        """     
        if scaled:
            retval = {n: jnp.array(p) for n, p in (self._iter_params(flatten=True, **kwargs))}
        else:
            retval = {n: p.value for n, p in (self._iter_params(flatten=True, **kwargs))}
            
            
        if return_floats:
            retval = {k: float(v) for k, v in retval.items()}
        return retval
         
    def flat_param_names(self, *args, **kwargs) -> list[str]:
        """
        Return flattened parameter names as a list.

        See :meth:`named_flat_params`.
        """
        return list(self.named_flat_params(*args, **kwargs).keys())    
    
    def flat_params(self, *args, **kwargs) -> list[Parameter]:
        """
        Return flattened parameters as a list.

        See :meth:`named_flat_params`.
        """
        return list(self.named_flat_params(*args, **kwargs).values())
    
    def flat_param_values(self, *args, **kwargs) -> jnp.ndarray:
        """
        Return flattened model parameter values as a jax arrays.

        See :meth:`named_flat_param_values`.
        """
        return jnp.array(list(self.named_flat_param_values(*args, **kwargs).values())).reshape(-1)
    
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
                prefix = f"{relative_name}{self.separator}" if relative_name else ""

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
    
    def distribution(self, param_groups: bool = True) -> JointParameterDistribution:
        """Joint distribution over (flattened) parameters.
        
        Parameters
        ----------
        param_grous : bool, optional
            Whether or not to use the internal parameter groups
            to create the joint distribution. Defaults to ``True``.
        
        Returns
        -------
        JointParameterDistribution
        """
        if param_groups:        
            return JointParameterDistribution(self.param_groups(), self.flat_param_names())
        else:
            param_groups = [ParameterGroup(param_names=[name], distribution=param.distribution) for name, param in self.named_flat_params().items()]
            return JointParameterDistribution(param_groups, self.flat_param_names())
    
    # ---- Parameter manipulation --------------------------------------------------            

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
            raise Exception("Setting a model's parameters with include_fixed is not yet supported")
        
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
                    sub_params = parent_param.flattened(separator=self.separator)
                    
                    updates_found = False
                    new_sub_values = []
                    
                    # Reconstruct the value array from current values + updates
                    for i, sub_p in enumerate(sub_params):
                        suffix = sub_p.name if sub_p.name is not None else str(i)
                        flat_name = f"{parent_name}{self.separator}{suffix}"
                        
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
        
    def with_fixed_params(self: Self, params: str | Sequence[str] | Callable[[str], bool], check_unknown=True) -> Self:
        """Return a model with specified parameters fixed.

        Parameters
        ----------
        params : str | Sequence[str] | Callable[[str], bool]
            Parameter names to fix.
        check_unknown : bool, default=True
            Error if any provided name does not exist.

        Returns
        -------
        Self
        """
        if isinstance(params, str):
            params = [params]
            
        if isinstance(params, Callable):
            params = [p for p in self.param_names() if params(p)]
        
        params = set(params)
            
        current_params = self.named_params()        
        current_param_names = set(current_params.keys())
        
        if check_unknown:
            for param_name in params:
                if param_name not in current_param_names:
                    raise Exception(f"Specified parameter '{param_name}' not found in model")
        
        new_params = current_params.copy()
        for name, param in current_params.items():
            if name in params:
                new_params[name] = param.as_fixed()
        return self.with_params(new_params)
    
    def with_free_params(self: Self, params: str | list[str] | Callable[[str], bool], fix_others=False) -> Self:
        """Free the specified parameters.

        Parameters
        ----------
        params : str | Sequence[str] | Sequence[Parameter]
            Parameters to set free.
        fix_others : bool, default=True
            Fix parameters not specified.

        Returns
        -------
        Self
        """
        if isinstance(params, str):
            params = [params]

        if isinstance(params, Callable):
            params = [p for p in self.param_names() if params(p)]        
        
        if len(params) > 0 and isinstance(params[0], Parameter):
            param_id_to_name: dict[Parameter, str] = {id(v): k for k, v in self.named_params().items()}
            params = [param_id_to_name[id(param)] for param in params]
        
        params = set(params)
        current_params = self.named_params(include_fixed=True)
        current_param_names = set(current_params.keys())
        
        for param_name in params:
            if param_name not in current_param_names:
                raise Exception(f"Specified parameter '{param_name}' not found in model")
        
        new_params = current_params.copy()
        for name, param in current_params.items():
            if name in params:
                new_params[name] = param.as_free()
            elif fix_others:
                new_params[name] = param.as_fixed()
        return self.with_params(new_params)
    
    def with_free_params_only(self: Self, *args, **kwargs) -> Self:
        """Returns a model with only the specified parameters freed.
        
        This is an alias for calling :meth:``with_free_params``
        with `fix_others=True`.

        See :meth:``with_free_params``.
        """
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_free_params_only`.")
        return self.with_free_params(*args, **kwargs)

    def with_all_params_fixed(self: Self, **kwargs) -> Self:
        """Returns a model with all parameters fixed.
        
        This is an alias for calling :meth:``with_free_params``
        with `fix_others=True` and no parameters passed.

        See :meth:``with_free_params``.
        """
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_all_params_fixed`.")
        return self.with_free_params({}, **kwargs)

    def with_all_params_free(self: Self, **kwargs) -> Self:
        """Returns a model with all parameters free.
        
        This is an alias for calling :meth:``with_free_params``
        with all parameters passed.

        See :meth:``with_free_params``.
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
        
        return dataclasses.replace(self, _param_groups=new_list)
    
    def with_param_groups_demoted(self: Self) -> Self:
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
                prefix = f.name + self.separator
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
            child_model = child_model.with_param_groups_demoted()
            
            new_fields[field_name] = child_model

        # 4. Return updated model
        return dataclasses.replace(self, _param_groups=groups_to_keep, **new_fields)
    
    # ---- Distribution manipulation --------------------------------------------------            
    
    def with_uniform_distributions(self, percentage=0.01):
        """Return a model with uniform distributions set centered on current parameter values.

        The distributions are defined with bounds calculated as ``value * (1.0 +/- percentage)``.

        Parameters
        ----------
        percentage : float, default=0.01
            The fractional width of the uniform distribution (e.g. 0.01 = 1%).

        Returns
        -------
        Self
            A new model with updated parameter distributions.
        """        
        updates = {}
        for name, param in self.named_params().items():
            distribution = UniformDistribution(param * (1.0 - percentage) / param.scale, param * (1.0 + percentage) / param.scale)
            updates[name] = param.with_distribution(distribution)
            
        return self.with_params(updates)       

    def with_distributions_mapped(self, map_fn: Callable[[Distribution], Distribution], filter_fn: Callable[[Distribution], bool] | None = None, param_groups=False):
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
        map_fn : Callable[[Distribution], Distribution]
            Function that takes a distribution and returns a new one.
        filter_fn : Callable[[Distribution], bool], optional
            A predicate function. If provided, the mapping is only applied to 
            distributions where ``filter_fn(dist)`` is True.
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
                do_map = True if filter_fn is None else filter_fn(group.distribution)
                if do_map:
                    # Update local groups using existing method to ensure safe replacement
                    mapped_model = mapped_model.with_param_groups(group.with_distribution(map_fn(group.distribution)))

            # 2. Recurse into Submodels
            new_submodels = {}
            for f in dataclasses.fields(mapped_model):
                child = getattr(mapped_model, f.name)
                # Check if the field is a direct submodel
                if isinstance(child, Model):
                    # Recursive call
                    updated_child = child.with_distributions_mapped(map_fn, filter_fn, param_groups=True)
                    new_submodels[f.name] = updated_child
            
            # Apply submodel updates if any
            if new_submodels:
                mapped_model = dataclasses.replace(mapped_model, **new_submodels)

        else:
            # Existing logic for individual params (Global via named_params)
            for name, param in self.named_params().items():
                do_map = True if filter_fn is None else filter_fn(param.distribution)
                if do_map:
                    # Fixed: Added 'map_fn' wrapper which was missing in the snippet
                    new_param = param.with_distribution(map_fn(param.distribution))
                    mapped_model = mapped_model.with_params({name: new_param})
                    
        return mapped_model
    
    # ---- Field and model manipulation --------------------------------------------------            
    
    @classmethod
    def with_defaults(cls, *args, **kwargs) -> type['Model']:
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

        Submodels parameters are obtained using :meth:`param_names`.,
        and subsequently freed using :meth:``with_free_params``.
        
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
        
        This is an alias for calling :meth:``with_free_submodels``
        with `fix_others=True`.

        See :meth:``with_free_params``.
        """     
        kwargs.setdefault('fix_others', True)
        if kwargs['fix_others'] == False:
            raise Exception("Cannot pass fix_others == False for `with_free_submodels_only`.")
        return self.with_free_submodels(*args, **kwargs)
    
    def with_fixed_submodels(self: Self, submodels: 'Model' | Sequence['Model'] | str | Sequence[str]) -> Self:
        """Fixed all parameters in the given submodels.

        Submodels parameters are obtained using :meth:`param_names`.,
        and subsequently fixed using :meth:``with_fixed_params``.
        
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
            'z0': self._z0,
        })
        ntwk = skrf.Network(**kwargs)
        if sigma != 0.0:
            ntwk.s += (np.random.normal(0, sigma, ntwk.s.shape) + 1j * np.random.normal(0, sigma, ntwk.s.shape))
        return ntwk    
    
    def write_hdf(self, group: h5py.Group):
        """Write model into an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            Target group. Two subgroups are created: ``raw`` and ``params``.
        """
        model_save = self._saveable()

        params_tree, static_tree = model_save.partition(include_fixed=True, param_objects=True)
        params = model_save.named_params()
        model_raw_grp = group.create_group('raw')
        model_raw_grp.create_dataset('combined', data=jsonpickle.encode(model_save))
        model_raw_grp.create_dataset('params', data=jsonpickle.encode(params_tree))
        model_raw_grp.create_dataset('static', data=jsonpickle.encode(static_tree))
        
        params_grp = group.create_group('params')
        for name, initial_param in params.items():
            params_grp[name] = initial_param.to_json()        
            
    @classmethod
    def read_hdf(cls, group: h5py.Group) -> Self:
        """Read model from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            Group previously created via :meth:`write_hdf`.

        Returns
        -------
        Self | None
            The decoded model, or ``None`` if decoding fails.
        """        
        model_raw_grp = group['raw']
        if 'combined' in model_raw_grp:
            combined_json = model_raw_grp['combined'][()]
            combined_json = combined_json.decode('utf-8') if isinstance(combined_json, bytes) else combined_json
            combined_tree = jsonpickle.decode(combined_json)
            return combined_tree
        
        params_json = model_raw_grp['params'][()]
        params_json = params_json.decode('utf-8') if isinstance(params_json, bytes) else params_json
        static_json = model_raw_grp['static'][()]
        static_json = static_json.decode('utf-8') if isinstance(static_json, bytes) else static_json
        
        try:
            params_tree = jsonpickle.decode(params_json)
            static_tree = jsonpickle.decode(static_json)
            
            return cls(eqx.combine(params_tree, static_tree))
        except Exception as e:
            import logging
            logging.warning(f'Error while trying to load model: {e}')
            return None
        
    def save(self, target: str | BinaryIO):
        """Serialize the model to disk in pickle format.

        The recommended file extension for ParamRF models is `.prf`.

        Parameters
        ----------
        filepath : str
            Destination file path.
        """
        model_save = self._saveable()

        data = jsonpickle.encode(model_save)
        
        if isinstance(target, (str, os.PathLike)):
            with open(target, "w", encoding="utf8") as f:
                f.write(data)
        else:
            target.write(data)
            
    @classmethod
    def load(cls, source: str | BinaryIO) -> Self:
        """Load a model from a pickled file.

        The recommended file extension for ParamRF models is `.prf`.

        Parameters
        ----------
        filepath : str
            Source JSON path.

        Returns
        -------
        Self
        """        
        if isinstance(source, (str, os.PathLike)):
            with open(source, "r", encoding="utf8") as f:
                data = f.read()
        else:
            data = source.read()

        return jsonpickle.decode(data)
    
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

def wrap(
    func: Callable,
    *args,
    as_numpy: bool = False,
) -> Callable:
    """
    Wrap a function/method taking ``(model, frequency, *args, **kwargs)`` into one that accepts flat arrays.

    The wrapper accepts ``(theta, [f], *args, **kwargs)`` where ``theta`` is a flat
    parameter array, and optionally ``f`` is a frequency array with a provided unit.

    If a :class:`Frequency` object is passed in ``*args``, it is closed over and
    the wrapped function only accepts ``theta``.

    Supported call signatures:
    
    * ``wrap(userfunc, model, "MHz")``
    * ``wrap(userfunc, model, freq)``
    * ``wrap(model.userfunc, "MHz")``
    * ``wrap(model.userfunc, freq)``

    Parameters
    ----------
    func : callable
        Function or bound method taking ``(model, frequency, ...)``.
    *args : tuple
        Variable length argument list. Expects either ``(model, Frequency)``,
        ``(model, unit_str)``, or just ``(Frequency,)`` / ``(unit_str,)``
        if ``func`` is a bound method.
    as_numpy : bool, optional
        If True, JIT the wrapper and convert inputs/outputs to NumPy arrays.
        Default is False.

    Returns
    -------
    wrapper : callable
        Wrapped function accepting ``(theta_array, [f_array], *args, **kwargs)``.

    Raises
    ------
    TypeError
        If the frequency argument is neither a unit string nor a ``Frequency`` object.
    """
    # Determine if func is a bound method
    if hasattr(func, '__self__') and func.__self__ is not None:
        model: Model = func.__self__
        func = func.__func__
        args = list(args)
    else:
        model: Model = args[0]
        args = args[1:]

    # Now args[0] should be Frequency or unit
    frequency_or_unit = args[0] if args else "Hz"

    # Validate
    if not isinstance(frequency_or_unit, (str, Frequency)):
        raise TypeError(f"Expected pmrf.Frequency or unit string as second argument, got type {type(frequency_or_unit)}")

    use_variable_frequency = isinstance(frequency_or_unit, str)

    # Define wrapped function
    def wrapped_with_freq(theta: jnp.ndarray, f_array: jnp.ndarray, *f_args, **f_kwargs):
        new_model = model.with_params(theta)
        frequency = Frequency.from_f(f_array, unit=frequency_or_unit)
        return func(new_model, frequency, *f_args, **f_kwargs)

    def wrapped_fixed_freq(theta: jnp.ndarray, *f_args, **f_kwargs):
        new_model = model.with_params(theta)
        return func(new_model, frequency_or_unit, *f_args, **f_kwargs)

    wrapped_fn = wrapped_with_freq if use_variable_frequency else wrapped_fixed_freq

    if as_numpy:
        raw_fn = jax.jit(wrapped_fn)
        if use_variable_frequency:
            wrapped_fn = lambda theta, f, *a, **kw: np.array(raw_fn(jnp.array(theta), jnp.array(f), *a, **kw))
        else:
            wrapped_fn = lambda theta, *a, **kw: np.array(raw_fn(jnp.array(theta), *a, **kw))

    # Forward metadata
    update_wrapper(wrapped_fn, func)
    
    return wrapped_fn