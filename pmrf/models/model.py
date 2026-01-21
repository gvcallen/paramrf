"""pmrf.model
================

Core model base class and utilities for ParamRF.

This module defines :class:`Model`, a frozen, JAX-compatible, Equinox module
representing an RF network, along with helper utilities like :func:`wrap`.

"""

from functools import cached_property, partial
from copy import deepcopy
from typing import Callable, Sequence, TypeVar, Any
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
from pmrf.functions.conversions import a2s, s2a
from pmrf.functions.math import FUNC_LOOKUP
from pmrf.parameters import Parameter, ParameterGroup, is_valid_param, asparam
from pmrf.distributions.parameter import JointParameterDistribution
from pmrf.constants import PRIMARY_PROPERTIES, IndexArray, ModelT
from pmrf.frequency import Frequency
from pmrf._util import field, classproperty, is_overridden, get_first_underlying_type
from pmrf._tree import nodes_by_type, value_at_path, partition, combine

ModelT = TypeVar('ModelT', bound='Model')

class Model(eqx.Module):
    """
    Overview
    --------
    Base class representing a computable RF network, referred to in
    **ParamRF** as a "Model". This class is abstract and should not be
    instantiated directly. Derive from :class:`Model` and override one of
    the primary property functions (e.g. :meth:`__call__`, :meth:`s`, :meth:`a`).

    The model is an Equinox ``Module`` (immutable, dataclass-like) and is
    treated as a JAX PyTree. Parameters are declared using standard dataclass
    field syntax with types like :class:`pmrf.Parameter`.

    Usage
    -------------
    - Define sub-classes with custom parameters and sub-models
    - Construct models by passing parameters and/or submodels to the initializer (like a dataclass).
    - Use `with_xxx` functions to update the model e.g. :meth:`with_params`, :meth:`with_replaced`.
    - Retrieve parameter information via methods such as :meth:`named_params`, :meth:`param_names`, :meth:`flat_params`, etc..

    See also the :mod:`pmrf.fitting` and :mod:`pmrf.sampling` modules for details on model fitting and sampling.

    Examples
    -------
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
    # Instance fields
    name: str | None = field(default=None, kw_only=True, static=True)
    separator: str = field(default='_', kw_only=True, static=True)
    metadata: dict = field(default_factory=lambda: dict(), kw_only=True, static=True)
    _z0: complex = field(default=50.0+0j, kw_only=True, static=True)
    _param_groups: list[ParameterGroup] = field(default_factory=lambda: list(), kw_only=True, repr=False, static=True)

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
                
                field_kwargs['converter'] = lambda x, field_name=field_name: asparam(x, name=field_name, fixed=True)
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

    # ---- PyTree manipulation and helpers -------------------------------------------------
    
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
    
    def path_to_param_name(self, path) -> str | list[str]:
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
    
    def partition(self: ModelT, include_fixed=False, param_objects=False) -> tuple[ModelT, ModelT]:        
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
        (ModelT, ModelT)
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
    
    # ---- Defaults / Primary ---------------------------------------------------    
    
    @classproperty
    def DEFAULT_PARAMS(cls) -> dict[str, Parameter]:
        """Default parameters for the model.

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
    
    @property
    def primary_function(self) -> Callable[[Frequency], jnp.ndarray]:
        """Primary function (``s`` or ``a``) as a callable.

        Returns
        -------
        Callable[[Frequency], jnp.ndarray]
        """
        return getattr(self, self.primary_property)
            
    @property
    def primary_property(self) -> str:
        """Name of the primary property (e.g. ``"s"``, ``"a"``).

        The primary property is the first overridden among
        :data:`PRIMARY_PROPERTIES`. If ``__call__`` is overridden, delegation
        occurs to the returned model.

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
    
    def copy(self: ModelT) -> ModelT:
        return deepcopy(self)    

    # ---- Introspection --------------------------------------------------------
    
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
        return len(self.named_params())   

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
        """Build a compositional circuit representing this model.

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
    def a(self, freq: Frequency) -> jnp.ndarray:
        """ABCD parameter matrix.

        If only :meth:`s` is implemented, converts via :func:`pmrf.functions.conversions.s2a`.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            ABCD matrix with shape ``(nf, 2, 2)``.

        Raises
        ------
        NotImplementedError
            If neither ``a`` nor ``s`` is implemented in a subclass.
        """        
        if is_overridden(type(self), Model, '__call__'):
            return self().a(freq)

        if not is_overridden(type(self), Model, 's'):
            raise NotImplementedError(f"Error: model sub-classes currently *have* to implement the 's' or the 'a' function, but class {type(self)} has neither")
        
        s = self.s(freq)
        return s2a(s, self.z0)
    
    @eqx.filter_jit
    def s(self, freq: Frequency) -> jnp.ndarray:
        """Scattering parameter matrix.

        If only :meth:`a` is implemented, converts via :func:`pmrf.functions.conversions.a2s`.

        Parameters
        ----------
        freq : Frequency
            Frequency grid.

        Returns
        -------
        jnp.ndarray
            S-parameter matrix with shape ``(nf, n, n)``.

        Raises
        ------
        NotImplementedError
            If neither ``a`` nor ``s`` is implemented in a subclass.
        """
        if is_overridden(type(self), Model, '__call__'):
            return self().s(freq)

        if not is_overridden(type(self), Model, 'a'):
            raise NotImplementedError(f"Error: model sub-classes currently *have* to implement the 's' or the 'a' function, but class {type(self)} has neither")
        
        a = self.a(freq)
        return a2s(a, self.z0)
    
    # ---- Structure utilities --------------------------------------------------    
    
    def children(self) -> list['Model']:
        """Immediate submodels.

        Returns
        -------
        list[Model]
        """
        return [node for node in eqx.tree_flatten_one_level(self)[0] if isinstance(node, Model)]
    
    def named_children(self) -> dict[str, 'Model']:
        """Immediate submodels keyed by their ``name`` attribute.

        Returns
        -------
        dict[str, Model]
        """
        return {child.name: child for child in self.children()}
    
    def submodels(self) -> list['Model']:
        """All nested submodels (depth-first), excluding ``self``.

        Returns
        -------
        list[Model]
        """
        return nodes_by_type(self, Model)[1:]
    
    # ---- Container model building --------------------------------------------------    

    def __pow__(self, other: 'Model') -> 'Model':
        """Cascade composition operator ``**``."""        
        from pmrf.models.containers import Cascade
        return Cascade([self, other])
    
    def connected(self, others: 'Model' | Sequence['Model'], ports: Sequence[int | Sequence[int]]) -> 'Model':
        """Return a version of the model with some ports connected.

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
    
    # ---- Model fitting --------------------------------------------------    
    
    def fitted(self: ModelT, measured: str | skrf.Network | NetworkCollection, **kwargs) -> ModelT:
        """Returns a new model fitted to measured data.

        This is an alternative API to the fitting submodule.
        Internally, a fitter is first created using `pmrf.fitting.Fitter(...)`, and then `fitter.fit(...)` is called,
        with the fitted model being returned. Fit results are stored in the model's metadata with key 'fit_results'.
        
        Key-word arguments are split into 'init' and 'fit' key-word arguments appropriately.

        Parameters
        ----------
        measured (prf.NetworkCollection | skrf.Network | str):
            The measured data.

        Returns:
            Model: The fitted model.
        """        
        from pmrf.fitting import Fitter, FITTER_INIT_PARAMS
        init_kwargs = {k: kwargs.pop(k) for k in FITTER_INIT_PARAMS if k in kwargs}
        return Fitter(self, **init_kwargs).fit(measured, **kwargs).fitted_model

    def with_submodels_fitted(self: ModelT, measured: str | skrf.Network | NetworkCollection, **kwargs) -> ModelT:
        """Returns a new model with its submodels fitted to measured data.

        This is an alternative API to the fitting submodule.
        Internally, a fitter is first created using `pmrf.fitting.Fitter(...)`, and then `fitter.fit_submodels(...)` is called,
        with the fitted model being returned. Fit results are stored in the model's metadata with key 'fit_results'.

        Key-word arguments are split into 'init' and 'fit' key-word arguments appropriately.

        Parameters
        ----------
        measured (prf.NetworkCollection | skrf.Network | str):
            The measured data.

        Returns:
            Model: The fitted model.
        """        
        from pmrf.fitting import Fitter, FITTER_INIT_PARAMS
        init_kwargs = {k: kwargs.pop(k) for k in FITTER_INIT_PARAMS if k in kwargs}
        return Fitter(model=self, **init_kwargs).fit_submodels(measured, **kwargs).fitted_model
    
    # ---- Parameter querying --------------------------------------------------        
    
    def params(self, *args, **kwargs) -> list[Parameter]:
        return list(self.named_params(*args, **kwargs).values())
    
    def named_params(self, include_fixed=False, flat=False, flat_params=False, values=False, scaled_values=False, submodels: 'Model' | Sequence['Model'] | str | Sequence[str] | None = None) -> dict[str, Parameter]:
        """Return model parameters as a dict (or flattened array).

        Keys are fully-qualified parameter names. The order matches the internal
        flattened array order.

        Parameters
        ----------
        include_fixed : bool, default=False
            Include fixed parameters.
        flat : bool, default=False
            If ``True``, return a 1D ``jnp.ndarray`` of parameter **values**.
        flat_params : bool, default=False
            If ``True``, return a dict of **manufactured** parameter objects
            (flattened) instead of an array.
        submodels : Model | Sequence[Model] | str | Sequence[str] | None, optional
            Restrict to parameters used by the given submodel(s). If strings are
            provided, ``getattr(self, name)`` is used.

        Returns
        -------
        dict[str, Parameter] or jnp.ndarray
        """
        spec = self._core_object_spec if include_fixed else self._free_object_spec
        params_tree = eqx.filter(self, spec, is_leaf=is_valid_param)
        path_and_params = jax.tree.flatten_with_path(params_tree, is_leaf=is_valid_param)
        _params: dict[str, Parameter] = {self.path_to_param_name(path): param for path, param in path_and_params[0]}          

        if flat | flat_params:
            flat_params_dict = {}
            for name, param in _params.items():
                if param.ndim > 1:
                    param_flattened = param.flatten(separator=self.separator)
                    for i, subparam in enumerate(param_flattened):
                        flat_params_dict[f'{name}{self.separator}{i}'] = subparam
                else:
                    flat_params_dict[name] = param
            _params = flat_params_dict

        if submodels is not None:
            if isinstance(submodels, Model) or isinstance(submodels, str):
                submodels = [submodels]
            if len(submodels) != 0 and isinstance(submodels[0], str):
                submodels = [getattr(self, name) for name in submodels]
            submodel_param_values = [param for submodel in submodels for param in submodel.params(include_fixed=include_fixed, flat=flat).values()]
            _params = {k: v for k, v in _params.items() if any(v is p for p in submodel_param_values)}
            
        if flat and not flat_params:
            return jnp.array([p.value for p in _params.values()])
        
        if values or scaled_values:
            if scaled_values:
                return {k: jnp.array(p) for k, p in _params.items()}
            else:
                return {k: p.value for k, p in _params.items()}
        return _params
    
    def named_param_values(self, scaled=False, *args, **kwargs) -> dict[str, jnp.ndarray]:
        kwargs['values'] = True
        kwargs['scaled_values'] = scaled
        return self.named_params(*args, **kwargs)
    
    def flat_params(self, *args, **kwargs) -> jnp.ndarray:
        """Shorthand for ``named_params(flat=True, ...)``."""        
        return self.named_params(*args, flat=True, **kwargs)
    
    def flat_param_names(self) -> list[str]:
        """Flat parameter names (matching :meth:`flat_params` order)."""        
        return list(self.named_params(flat_params=True).keys())
    
    def submodel_params(self, submodels: 'Model' | Sequence['Model'] | str | Sequence[str], **kwargs):
        if isinstance(submodels, Model) or isinstance(submodels, str):
            submodels: list[Model] = [submodels]
        
        if len(submodels) != 0 and isinstance(submodels[0], str):
            submodels: list[Model] = [getattr(self, name) for name in submodels]

        param_values = [param for source in submodels for param in source.named_params(**kwargs).values()]
        params = {k: v for k, v in self.named_params(**kwargs).items() if any(v is p for p in param_values)}        
        return params
    
    def param_names(self, *args, **kwargs):
        """Parameter names for current (possibly filtered) selection."""        
        params = self.named_params(*args, **kwargs)
        return list(params.keys())
        
    def param_groups(self, include_fixed=False) -> list[ParameterGroup]:
        """Return all parameter groups relevant to this model.

        This function is useful for fitters that need to take into account
        e.g. any constraints or correlated priors. Only groups that were
        create with the same `flat` flag are returned.
        
        Currently, only groups for the current model are considered
        (i.e. groups in children models are ignored).

        Parameters
        ----------
        include_fixed : bool, default=False
            Include groups involving fixed parameters.

        Returns
        -------
        list[ParameterGroup]
        """        
        
        params = self.named_params(flat=True, flat_params=True, include_fixed=include_fixed)
        groups = [group for group in self._param_groups]

        grouped_param_names = {name for group in groups for name in group.parameter_names}
        for name, param in params.items():
            if name not in grouped_param_names:
                groups.append(ParameterGroup(param_names=[name], dist=param.distribution))
        
        return groups
    
    def distribution(self) -> Distribution:
        """Joint distribution over (flattened) parameters."""        
        return JointParameterDistribution(self.param_groups(), self.flat_param_names())
    
    # ---- Parameter manipulation --------------------------------------------------            

    def with_params(
        self: ModelT,
        params: dict[str, Parameter] | dict[str, float] | jnp.ndarray | None = None,
        check_missing: bool = False,
        check_unknown: bool = False,
        fix_others = False,
        include_fixed = False,
        **param_kwargs: dict[str, Parameter] | dict[str, float],
    ) -> ModelT:
        """Return a new model with core parameters updated.

        Parameters
        ----------
        params : dict[str, Parameter] | dict[str, float] | jnp.ndarray | None, optional
            Parameter updates. If an array, **all** values must be provided
            (matching ``flat_params`` order). You may also pass keyword args.
        check_missing : bool, default=False
            Require that all model parameters are specified.
        check_unknown : bool, default=False
            Error if unknown parameter keys are provided.
        fix_others : bool, default=False
            Fix any parameters not explicitly passed.
        include_fixed : bool, default=False
            Include fixed parameters when interpreting ``params`` mapping.
        **param_kwargs : dict
            Additional parameter updates by name.

        Returns
        -------
        ModelT

        Raises
        ------
        Exception
            If shape/order mismatches, unknown/missing names (when checked),
            or if arrays are found outside of Parameters.
        """
        # Prepare input
        if isinstance(params, dict) or len(param_kwargs) != 0:
            if include_fixed:
                raise Exception('Not yet supported')
            
            params = params if params is not None else {}
            params.update(param_kwargs)
        
            # Generate an ordered, input flat params array for verification
            new_params = self.named_params(include_fixed=True)
        
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
        else:
            params = jnp.array(params)

            if params.shape[0] != self.num_flat_params:
                raise Exception(f'Expected {self.num_flat_params} flat parameters but was passed {params.shape[0]}')
            params_tree, static = self.partition(include_fixed=include_fixed)
            params_out, unravel_fn = flatten_util.ravel_pytree(params_tree)
            
            if jnp.isscalar(params_out) or params_out.shape[0] == 0:
                raise Exception("Error: no free model parameters found to make feature function")
            
            params_tree_recon = unravel_fn(params)
            return combine(params_tree_recon, static)           
        
    def with_flat_params(self, *args, **kwargs):
        """Alias for :meth:`with_params` when passing a flat array."""        
        return self.with_params(*args, **kwargs)
    
    def with_uniform_distributions(self, width_frac=0.01):
        updates = {}
        for name, param in self.named_params().items():
            distribution = UniformDistribution(param * (1.0 - width_frac) / param.scale, param * (1.0 + width_frac) / param.scale)
            updates[name] = param.with_distribution(distribution)
            
        return self.with_params(updates)    
    
    def with_param_groups(self: ModelT, param_groups: ParameterGroup | list[ParameterGroup]) -> ModelT:
        """Return a a model with parameter groups appended.
        
        Parmaeter groups can be used to specify relationships between parameters in the model,
        such as joint priors.

        Parameters
        ----------
        param_groups : ParameterGroup or list[ParameterGroup]
            Group(s) to add.

        Returns
        -------
        ModelT
        """        
        if not isinstance(param_groups, list):
            param_groups = [param_groups]
        
        param_groups_old = self._param_groups.copy() if self._param_groups is not None else []
        param_groups_new = param_groups_old
        param_groups_new.extend(param_groups)
        param_groups_new = [param_group for param_group in param_groups_new]
        return dataclasses.replace(self, _param_groups=param_groups_new)
        
    def with_fixed_params(self: ModelT, params: list[str] | Callable[[str], bool], check_unknown=True) -> ModelT:
        """Return a model with specified parameters fixed.

        Parameters
        ----------
        params : list[str]
            Parameter names to fix.
        check_unknown : bool, default=True
            Error if any provided name does not exist.

        Returns
        -------
        ModelT
        """
        # For legacy callers
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
    
    def with_free_params(self: ModelT, params: list[str] | Callable[[str], bool], fix_others=False) -> ModelT:
        """Free the specified parameters.

        Parameters
        ----------
        params : Sequence[str] | Sequence[Parameter]
            Parameters to set free.
        fix_others : bool, default=True
            Fix parameters not specified.

        Returns
        -------
        ModelT
        """
        if isinstance(params, Callable):
            params = [p for p in self.param_names() if params(p)]        
        
        if isinstance(params[0], Parameter):
            param_id_to_name: dict[Parameter, str] = {id(v): k for k, v in self.named_params().items()}
            params = [param_id_to_name[id(param)] for param in params]
        
        params = set(params)
        current_params = self.named_params(include_fixed=True)
        current_param_names = set(current_params.keys())
        
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
    
    def with_free_params_only(self: ModelT, params: list[str] | Callable[[str], bool]) -> ModelT:
        return self.with_free_params(params, fix_others=True)    
    
    # ---- Field and model manipulation --------------------------------------------------            
    
    @classmethod
    def with_defaults(cls, *args, **kwargs):
        # return partial(cls, *args, **kwargs)
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
    
    def with_models(self: ModelT, other_models: list[ModelT]):
        combined = self
        for other in other_models:
            combined = combined.with_params(other.named_params())
            combined = combined.with_param_groups(other.param_groups())
        return combined
    
    def with_fields(self: ModelT, *args, **kwargs) -> ModelT:
        """Return a copy with dataclass-style field replacements."""        
        return dataclasses.replace(self, *args, **kwargs)
    
    def with_submodel_fields(self: ModelT, submodel_name: str, *args, **kwargs) -> ModelT:
        """Return a copy with dataclass-style field replacements on a submodel."""        
        with_field_kwargs = {submodel_name: getattr(self, submodel_name).with_fields(*args, **kwargs)}
        return self.with_fields(**with_field_kwargs)    
    
    def with_free_submodels(self: ModelT, submodels: 'Model' | Sequence['Model'] | str | Sequence[str], include_fixed=False, fix_others=False) -> ModelT:
        """Free all parameters in the given submodels.

        Submodels parameters are obtained using ``Model.submodel_params``,
        and subsequently freed using ``Model.with_free_params``.
        
        Parameters
        ----------
        free_submodels : Model | Sequence[Model] | str | Sequence[str]
            Submodels whose parameters should be free.

        Returns
        -------
        ModelT
        """        
        model_params = self.submodel_params(submodels, include_fixed=include_fixed)
        return self.with_free_params(list(model_params.keys()), fix_others=fix_others)
    
    def with_fixed_submodels(self: ModelT, submodels: 'Model' | Sequence['Model'] | str | Sequence[str]) -> ModelT:
        """Fixed all parameters in the given submodels.

        Submodels parameters are obtained using ``Model.submodel_params``,
        and subsequently fixed using ``Model.with_fixed_params``.
        
        Parameters
        ----------
        free_submodels : Model | Sequence[Model] | str | Sequence[str]
            Submodels whose parameters should be free.

        Returns
        -------
        ModelT
        """        
        model_params = self.submodel_params(submodels)
        return self.with_fixed_params(list(model_params.keys()))        
    
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
        params_tree, static_tree = self.partition(include_fixed=True, param_objects=True)
        params = self.named_params()
        model_raw_grp = group.create_group('raw')
        model_raw_grp.create_dataset('combined', data=jsonpickle.encode(self))
        model_raw_grp.create_dataset('params', data=jsonpickle.encode(params_tree))
        model_raw_grp.create_dataset('static', data=jsonpickle.encode(static_tree))
        
        params_grp = group.create_group('params')
        for name, initial_param in params.items():
            params_grp[name] = initial_param.to_json()        
            
    @classmethod
    def read_hdf(cls, group: h5py.Group) -> ModelT:
        """Read model from an HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            Group previously created via :meth:`write_hdf`.

        Returns
        -------
        ModelT | None
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

        Parameters
        ----------
        filepath : str
            Destination file path.
        """
        model_save = self.with_fields(metadata=None)

        data = jsonpickle.encode(model_save)
        
        if isinstance(target, (str, os.PathLike)):
            with open(target, "w", encoding="utf8") as f:
                f.write(data)
        else:
            target.write(data)
            
    @classmethod
    def load(cls: ModelT, source: str | BinaryIO) -> ModelT:
        """Load a model from a pickled file.

        Parameters
        ----------
        filepath : str
            Source JSON path.

        Returns
        -------
        ModelT
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
        frequency : Frequency | skrf.Frequency
        filename : str
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
    """Wrap a function/method taking ``(model, frequency, *args, **kwargs)`` into one that accepts flat arrays.

    The wrapper accepts ``(theta, [f], *args, **kwargs)`` where ``theta`` is a flat
    parameter array, and optionally ``f`` is a frequency array with a provided unit.

    Supported calls
    ---------------
    - ``wrap(userfunc, model, "MHz")``
    - ``wrap(userfunc, model, freq)``
    - ``wrap(model.userfunc, "MHz")``
    - ``wrap(model.userfunc, freq)``

    If a :class:`Frequency` object is passed, it is closed over and the wrapped
    function only accepts ``theta``.

    Parameters
    ----------
    func : Callable
        Function or bound method taking ``(model, frequency, ...)``.
    *args : tuple
        Either ``(model, Frequency|unit)`` or just ``(Frequency|unit)`` if ``func`` is bound.
    as_numpy : bool, default=False
        If ``True``, JIT the wrapper and convert inputs/outputs to NumPy arrays.

    Returns
    -------
    Callable
        Wrapped function accepting ``(theta_array, [f_array], *args, **kwargs)``.

    Raises
    ------
    TypeError
        If the frequency argument is neither a unit string nor ``Frequency``.
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
        new_model = model.with_flat_params(theta)
        frequency = Frequency.from_f(f_array, unit=frequency_or_unit)
        return func(new_model, frequency, *f_args, **f_kwargs)

    def wrapped_fixed_freq(theta: jnp.ndarray, *f_args, **f_kwargs):
        new_model = model.with_flat_params(theta)
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