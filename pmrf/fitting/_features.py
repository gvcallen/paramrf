from typing import Sequence, Callable, Union
import re

import skrf
import jax
import jax.numpy as jnp
from jax import flatten_util

from pmrf._constants import FeatureT, FeatureListT
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._tree import combine

FeatureFunctionT = Callable[[Model | jnp.ndarray], jnp.ndarray]
ModelParametersT = Union[Model | jnp.ndarray]

def extract_features(
    source: Model | skrf.Network | Sequence[skrf.Network],
    features: FeatureT | FeatureListT,
    freq: Frequency | skrf.Frequency = None,
) -> jnp.ndarray:
    """Extracts features from a model or a network.
    
    This function allows for an arbitrary number of features (e.g. ['s11', 'a21_mag'])
    to be easily extracted from a model or a measured network.
    The resultant features are combined column-by-column into a matrix
    with frequency in the row dimensions that can easily be used in optimization schemes.
    
    For example, to extract S11 magnitude, specify either the alias 's11_db' or the full feature `('s_db', (0, 0))`.
    Similarly, to extract the phase of the B parameter of the ABCD matrix, specify 'a21_deg'.
    Finally, to extract multiple ports and features, specify a list e.g. `[('y', (0, 0)), ('s_db', (1, 0))]`.
    
    This function also allows a list of lists to be passed, in which case the list is simply flattened.
    This is useful when you want to use the same feature for all networks, in which case you
    can easily use a list comprehension to define the features.

    Args:
        source (Model | skrf.Network | Sequence[skrf.Network]): The source model or network(s) to extract the features from.
                                                                Lists of networks are treated as forming one, stack network with isolated ports.
        features (list[Feature] | list[list[Feature]]):         The features to extract, `[('s', (0, 0))]`, as described in detail above.
        freq (pmrf.Frequency | skrf.Frequency, optional):       The frequency to extract the features at. This will become the row dimension of the resultant matrix.
                                                                This must be passed for `Model` sources. Defaults to `None` e.g. for measured networks,
                                                                in which case the network's internal frequency is used. Otherwise, the network is interpolated.

    Returns:
        np.ndarray: The feature matrix.
    """    
    features = _format_features(features)
    if isinstance(source, skrf.Network):
        freq = source.frequency
        source = [source]
    elif isinstance(source, Sequence) and isinstance(source[0], skrf.Network):
        freq = source[0].frequency
    elif isinstance(source, Model):
        if freq is None:
            raise Exception("Frequency must be passed when extracting features from a model")
        if isinstance(freq, skrf.Frequency):
            freq = Frequency.from_skrf(freq)
    else:
        raise TypeError("Invalid type to extract_features")
    if isinstance(source, Model):
        return _extract_model_features(source, features, freq)
    else:
        return _extract_measured_features(source, features, freq)

def create_stacked_features(base: FeatureT | FeatureListT, ntwks: list[skrf.Network | None]) -> list[FeatureT]:
    """Helper function to generate stacked feature descriptors with port indices offset by network port counts.

    Args:
        base (FeatureT | FeatureListT): Formatted or unformatted base feature or list of base features. See `extract_features` for more information.
        ntwks (list[Network | None]): List of scikit-rf networks. All `None` networks are assumed to have the same number of ports as the first network.

    Returns:
        list[FeatureT]: List of new features with updated port indices.
    """    
    base_features = _format_features(base)
    stacked_features = []

    port_offset = 0
    default_num_ports = None
    for ntwk in ntwks:
        if not ntwk is None:
            default_num_ports = ntwk.number_of_ports
            break
        
    if default_num_ports is None:
        raise Exception("All networks were found to be None")

    for ntwk in ntwks:
        if ntwk is None:
            port_offset += default_num_ports
            continue
        
        for base_type, (m, n) in base_features:
            stacked_features.append((base_type, (m + port_offset, n + port_offset)))
        
        port_offset += ntwk.number_of_ports

    return stacked_features

def make_feature_function(
    model: Model,
    features: FeatureT | FeatureListT,
    freq: Frequency | skrf.Frequency,
    flat = False,
    jit = False,
) -> tuple[FeatureFunctionT, ModelParametersT] | tuple[FeatureFunctionT, ModelParametersT, Callable]:
    """Generate a feature function to parametrically extract model features.
    
    This function returns a callable feature function to extract model features,
    alongside model parameters. The function can be just-in-time compiled using jax,
    to enable its efficient, machine-code level computation.
    
    The function generated accepts the model parameters, in either Pytree
    or flattened (raveled) formated, and returns the resultant model feature matrix
    to be used in fitting procedures.

    Args:
        model (Model):                          The model to generate the feature functions for.
        features (FeatureT | FeatureList):      The list of features. See `extract_features` for more information.
        freq (pmrf.Frequency):                  The frequency to extract the features at, treated as a static argument.
        flat (bool):                            Whether the feature function should accept a flat array as input.
                                                If True, a third argument will be returned, which is a "reconstruct" function
                                                that transforms the flat array back into the full (unraveled-combined) model.
        jit (bool):                             Whether or not to just-in-time compile the function.

    Returns:
        tuple[FeatureFunction, ModelParameters]: The feature function, alongside the partitioned or flattened model parameters.
    """
    if isinstance(freq, skrf.Frequency):
        freq = Frequency.from_skrf(freq)
    
    features = _format_features(features)
    params_tree, static = model.partition()
    
    if flat:
        params_out, unravel_fn = flatten_util.ravel_pytree(params_tree)
        def reconstruct_fn(flat_params) -> Model:
            params_tree_recon = unravel_fn(flat_params)
            return combine(params_tree_recon, static)
            
        def feature_fn(flat_params) -> jnp.ndarray:
            model_recon = reconstruct_fn(flat_params)
            return extract_features(model_recon, features, freq)
    else:
        params_out = params_tree
        def feature_fn(tree_params) -> jnp.ndarray:
            model_recon = combine(tree_params, static)
            return extract_features(model_recon, features, freq)
    
    if jit:
        feature_fn = jax.jit(feature_fn)
        
    if flat:
        return feature_fn, params_out, reconstruct_fn

    return feature_fn, params_out

def _parse_feature_alias(alias: str) -> FeatureT:
    # Converts a feature alias like 's11_mag' to a feature tuple like ('s', (0, 0)).
    # Supports arbitrary feature types (e.g., 's', 't'), two-digit port numbers,
    # and optional suffixes (e.g., '_mag', '_db').
    match = re.match(r'^([a-zA-Z]+)(\d)(\d)(_.+)?$', alias)
    if not match:
        raise ValueError(f"Invalid feature alias format: '{alias}'")

    prefix = match.group(1)
    port1 = int(match.group(2)) - 1
    port2 = int(match.group(3)) - 1
    suffix = match.group(4) or ''

    return (prefix + suffix, (port1, port2))

def _format_features(features: FeatureT | FeatureListT) -> list[FeatureT]:
    if not isinstance(features, list):
        features = [features]
    elif isinstance(features[0], list):
        features = [feature for features_inner in features for feature in features_inner]
    
    for i in range(len(features)):
        if isinstance(features[i], str):
            features[i] = _parse_feature_alias(features[i])
    return features
    
def _extract_model_features(model: Model, features: list[FeatureT], freq: Frequency) -> jnp.ndarray:
    n_frequencies = len(freq)
    n_features = len(features)

    X = jnp.zeros((n_frequencies, n_features), dtype=jnp.complex128)
    for d, feature in enumerate(features):
        prop = feature[0]
        m, n = feature[1]
        x = None
        
        if prop[2:4] == 'mn':
            xfn = getattr(model, prop)
            x = xfn(freq,m,n)
        else:
            xfn = getattr(model, prop)
            x = xfn(freq)[:,m,n]    
            
        X = X.at[:, d].set(x)
        
    return X

def _extract_measured_features(networks: list[skrf.Network], features: list[FeatureT], freq: Frequency) -> jnp.ndarray:
    n_frequencies = len(freq)
    n_features = len(features)

    X = jnp.zeros((n_frequencies, n_features), dtype=jnp.complex128)
    for d, feature in enumerate(features):
        prop = feature[0]
        m, n = feature[1]
        x = None
        
        x = None
        offset = 0 # of the full stacked network
        for ntwk in networks:
            nports = ntwk.nports
            if m >= offset + nports:
                offset += nports
                continue          
                        
            if prop[2:4] == 'mn':
                i = prop.index('_')
                prop_new = prop[0:i]
                if len(prop) > i + 3:
                    prop_new += prop[i+3:]
                prop = prop_new
            x = getattr(ntwk, prop)[:,m-offset,n-offset]
            break
        if x is None:
            raise Exception('Error: port of out bounds')
        
        X = X.at[:, d].set(x)
    return X    