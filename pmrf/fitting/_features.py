from typing import Sequence, Callable, Union

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

def generate_feature_function(
    model: Model,
    features: FeatureListT,
    freq: Frequency | skrf.Frequency,
    flat=False,
    return_unravel_fn=False,
    jit=False,
) -> tuple[FeatureFunctionT, ModelParametersT] | tuple[FeatureFunctionT, ModelParametersT, Callable]:
    """Generate a feature function to easily extract model features.
    
    This function returns a callable feature function to extract model features,
    alongside model parameters. The function can be just-in-time compiled using jax,
    to enable its efficient, machine-code level computation.
    
    The function generated accepts the model parameters, in either Pytree
    or flattened (raveled) formated, and returns the resultant model feature matrix
    to be used in fitting procedures.

    Args:
        model (Model): The model to generate the feature functions for.
        features (list[Feature] | list[list[Feature]]): The list of features. See `extract_features` for more information.
        freq (pmrf.Frequency): The frequency to extract the features at, treated as a static argument.
        flat (bool): Whether the feature function should accept a flat array as input.
                     If True, a third argument will be returned, which is a "reconstruct" function
                     that transforms the flat array back into the full (unraveled-combined) model.
        jit (bool): Whether or not to just-in-time compile the function.

    Returns:
        tuple[FeatureFunction, ModelParameters]: The feature function, alongside the partitioned or flattened model parameters.
    """
    if isinstance(freq, skrf.Frequency):
        freq = Frequency.from_skrf(freq)
    
    features = _process_features(features)
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

def extract_features(
    source: Model | skrf.Network | Sequence[skrf.Network],
    features: list[FeatureT] | list[list[FeatureT]],
    freq: Frequency | skrf.Frequency = None,
) -> jnp.ndarray:
    """Extracts features from a model or a network.
    
    This function allows for an arbitrary number of features (e.g. 's', 's_db')
    for specified ports to be easily extracted from a model or a measured network.
    The resultant features are combined column-by-column into a matrix
    with frequency in the row dimensions that can easily be used in optimization schemes.
    
    For example, to extract S11, specify `[('s', (0, 0))]`.
    Similarly, to extract the magnitude of the B parameter of the ABCD matrix, specify `[('a_mag', (1, 0))]`.
    Finally, to extract multiple ports and features, specify e.g. `[('s', (0, 0)), ('s_db', (1, 0))]`.
    
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
    features = _process_features(features)
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
    
def _extract_model_features(model: Model, features: list[FeatureT], freq: Frequency) -> jnp.ndarray:
    n_frequencies = len(freq)
    n_features = len(features)

    X = jnp.zeros((n_frequencies, n_features), dtype=jnp.complex128)
    for d, feature in enumerate(features):
        prop = feature[0]
        m, n = feature[1]
        x = None
        
        if prop[-2:] == 'mn' and hasattr(model, prop):
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
            
            if prop[-2:] == 'mn':
                prop = prop[0:-3]
            x = getattr(ntwk, prop)[:,m-offset,n-offset]
            break
        if x is None:
            raise Exception('Error: port of out bounds')
        
        X = X.at[:, d].set(x)
    return X    

def _process_features(features: list[FeatureT] | list[list[FeatureT]]) -> list[FeatureT]:
    if isinstance(features[0], list):
        features = [feature for features_inner in features for feature in features_inner]
    return features