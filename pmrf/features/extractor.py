"""
Extractor helper class for extracting features such as S-parameters from ParamRF Models and scikit-rf Networks.
"""
import re
from typing import Callable

import skrf
import jax.numpy as jnp
import equinox as eqx
from dataclasses import replace

from pmrf.network_collection import NetworkCollection
from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.constants import FeatureSpec

class Extractor(eqx.Module):
    """
    A feature extractor used to extract features (e.g., S-parameters) 
    from Models, as well as scikit-rf Networks or NetworkCollections.
    """
    property: str | Callable[[Model, Frequency], jnp.ndarray] = eqx.field(static=True)
    ports: tuple[int, int] | None = eqx.field(static=True)    
    subattrs: list[str] | None = eqx.field(static=True)

    def __init__(self, property: str | Callable, ports: tuple[int, int] | None, subattrs: list[str] | str | None = None):
        if isinstance(subattrs, str):
            subattrs = [subattrs]
        
        self.property = property
        self.ports = ports
        self.subattrs = subattrs

    def resolve_source(self, source: Model | skrf.Network | NetworkCollection) -> Model | skrf.Network:
        """
        Traverses nested attributes or collections to retrieve the underlying Model or Network.
        
        Parameters
        ----------
        source : Model | skrf.Network | NetworkCollection
            The top-level object to resolve.
            
        Returns
        -------
        Model | skrf.Network
            The resolved target object.
            
        Raises
        ------
        TypeError
            If the resolution does not yield a Model or skrf.Network.
        """
        if self.subattrs is not None:
            for subattr in self.subattrs:
                if isinstance(source, NetworkCollection):
                    source = source[subattr]
                else:
                    source = getattr(source, subattr)
                    
        if not isinstance(source, Model) and not isinstance(source, skrf.Network):
            raise TypeError(f"Specified extractor sub-attrs {self.subattrs} did not resolve the source to a Model or Network.")
        
        return source

    def __call__(self, source: Model | skrf.Network | NetworkCollection, frequency: Frequency) -> jnp.ndarray:
        """
        Extracts and evaluates the requested property for a given frequency.
        
        Parameters
        ----------
        source : Model | skrf.Network | NetworkCollection
            The object from which to extract the feature.
        frequency : Frequency
            The frequency array/object to evaluate against.
            
        Returns
        -------
        jnp.ndarray
            The extracted feature array.
        """
        prop, ports = self.property, self.ports
        source = self.resolve_source(source)

        # Retrieve the evaluated property depending on the source type
        if isinstance(source, Model):
            if callable(prop):
                x = prop(source, frequency)
            else:
                x = getattr(source, prop)(frequency)
        else:
            if isinstance(frequency, Frequency):
                frequency = frequency.to_skrf()
            source = source.interpolate(frequency)

            if callable(prop):
                x = prop(source)
            else:
                x = getattr(source, prop)
                
        # Slice for specific port indices if provided
        if ports is not None:
            m, n = ports
            if m >= source.nports or n >= source.nports:
                raise IndexError(f"Property {prop}{m+1}{n+1} specified but model is a {source.nports}-port")
            x = x[:, m, n]
        
        return x
                    
    @classmethod
    def from_alias(cls, alias: str) -> 'Extractor':
        """
        Constructs an extractor instance from a standardized alias string.
        
        Supports:
        - Nested attributes: 'subattr1.subattr2.s21_deg'
        - Aliases with two-digit port numbers: 's11_mag'
        - General properties: 'a_db'
        
        Parameters
        ----------
        alias : str
            The alias string to parse.
            
        Returns
        -------
        Extractor
            The initialized Extractor.
        """
        fields = alias.split('.')
        if len(fields) > 1:
            subattrs = fields[:-1]
            alias = fields[-1]
        else:
            subattrs = None

        match = re.match(r'^([a-zA-Z]+)(\d)?(\d)?(.*)$', alias)
        if not match:
            raise ValueError(f"Invalid feature alias format: '{alias}'")

        prop_prefix = match.group(1)
        port1 = match.group(2)
        port2 = match.group(3)
        prop_suffix = match.group(4)
        
        prop = prop_prefix + prop_suffix

        # Map 1-indexed string alias (e.g., S11) to 0-indexed port array slices
        if port1 is not None and port2 is not None:
            ports = (int(port1) - 1, int(port2) - 1)
        else:
            ports = None
        
        return cls(property=prop, ports=ports, subattrs=subattrs)


def make_extractors(
    features: FeatureSpec,
    *,
    source: Model | skrf.Network | NetworkCollection | None = None,
    sparam_filter: str | None = None,
) -> list[Extractor]:
    """
    Constructs a list of Extractors from strings (aliases) and/or callables.

    A source can optionally be specified for dynamic setup. For instance, 
    if an alias implies an S-parameter matrix (starts with 's_'), passing 
    a source allows automatic expansion to all valid port tuples based on the network.
    
    Parameters
    ----------
    features : FeatureSpec
        The features for extractor initialization. Can be aliases parsed 
        by :meth:`Extractor.from_alias` or direct callables.
    source : Model | skrf.Network | NetworkCollection | None, optional
        The source object required for dynamic specifications, by default None.
    sparam_filter : {'transmission', 'reflection', None}, optional
        If specified, expands S-parameter aliases starting with 's_' to only 
        include either transmission (i != j) or reflection (i == j) S-parameters 
        based on the source's port count, by default None.

    Returns
    -------
    list[Extractor]
        The instantiated list of Extractor objects.
    """
    if isinstance(features, str) or callable(features):
        features = [features]
   
    extractors = []
    for spec in features:
        if callable(spec):
            extractors.append(Extractor(property=spec))
        else:
            extractor = Extractor.from_alias(spec)
            
            # Dynamically handle port expansion for S-parameters if a source is provided
            if source is not None and sparam_filter is not None and extractor.property.startswith('s_'):
                source_resolved = extractor.resolve_source(source)
                
                if sparam_filter == 'transmission':
                    port_tuples = [pt for pt in source_resolved.port_tuples if pt[0] != pt[1]]
                elif sparam_filter == 'reflection':
                    port_tuples = [pt for pt in source_resolved.port_tuples if pt[0] == pt[1]]
                else:
                    raise ValueError(f"Invalid sparam_filter: '{sparam_filter}'")
                
                for ports in port_tuples:
                    extractors.append(replace(extractor, ports=ports))
            else:
                extractors.append(extractor)
    
    return extractors


def extract_multiple_features(
    extractors: list[Extractor],
    source: Model | skrf.Network | NetworkCollection,
    frequency: Frequency,
    dtype: jnp.dtype = jnp.complex128,
) -> jnp.ndarray:
    """
    Extracts features using a list of Extractors and combines them into a single array.
    
    Parameters
    ----------
    extractors : list[Extractor]
        A list of instantiated Extractor objects.
    source : Model | skrf.Network | NetworkCollection
        The RF component or collection from which to extract features.
    frequency : Frequency
        The frequency array/object to evaluate against.
    dtype : jnp.dtype, optional
        The desired JAX data type for the returned array, by default jnp.complex128.
        
    Returns
    -------
    jnp.ndarray
        A JAX array containing the combined extracted features, stacked along the last axis.
        If the extractors list is empty, returns an empty array.
    """
    if not extractors:
        return jnp.array([], dtype=dtype)

    extracted_data = [extractor(source, frequency) for extractor in extractors]
    feature_matrix = jnp.stack(extracted_data, axis=-1)
    
    return feature_matrix.astype(dtype)