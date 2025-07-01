from typing import Sequence, Callable, Union
import re

import skrf
import jax
import jax.numpy as jnp
from jax import flatten_util

from pmrf._constants import FeatureT, FeatureInputT
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf._tree import combine

FeatureFunctionT = Callable[[Model | jnp.ndarray], jnp.ndarray]
ModelParametersT = Union[Model | jnp.ndarray]

def extract_features(
    source: Model | skrf.Network | dict[str, skrf.Network],
    features: FeatureInputT,
    freq: Frequency | skrf.Frequency = None,
) -> jnp.ndarray:
    """Extracts features from a model or a network.
    
    This function allows for an arbitrary number of features (e.g. ['s11', 'a21_mag'])
    to be easily extracted from a model or a measured network. The resultant features
    are combined column-by-column into a matrix with frequency in the row dimension.
    
    Features can either be specified by convenient aliases using strings, or by their full structure.
    As some examples to demonstrate the possibilities:
    - To extract S11 magnitude, specify either the alias 's11_db' or the full tuple `('', 's_db' (0, 0))`.
      Note that, for the tuple, the empty string at the beginning represents the base model (explained below).
    - To extract e.g. the phase of the B parameter of the ABCD matrix, specify 'a21_deg'.
    - To extract features from a submodel or specific network, specify a dictionary with features source "label" as keys
      e.g. {'src_label': s11_db'}. For models, this extracts a feature from a submodel that must be retrievable via `getattr`.
      For measured networks, this extract a feature from the corresponding network with that label in the dictionary.
      This is internally converted to the feature tuple ('src_name', 's_db', (0, 0)').
    - For a list of features, specify a list of any of the above, (or equivalently a dictionary of lists).

    Args:
        source (Model | skrf.Network | dict[str, skrf.Network]):        The source model or network(s) to extract the features from,
                                                                        with missing networks specified using integers for the number of ports at that index.
                                                                        Lists of networks are treated as forming one, stack network with isolated ports.
        features (FeatureInputT):                                       The features to extract, as described in detail above.
        freq (pmrf.Frequency | skrf.Frequency, optional):               The frequency to extract the features at. This will become the row dimension of the resultant matrix.
                                                                        This must be passed for `Model` sources. Defaults to `None` e.g. for measured networks,
                                                                        in which case the network's internal frequency is used. Otherwise, the network is interpolated.

    Returns:
        np.ndarray: The feature matrix.
    """    
    # We format the features to be flat (and parse them in the process)
    features = _format_features(features)
    
    # Get the frequency and format the sources
    if isinstance(source, skrf.Network):
        freq = source.frequency
        source = {'': source}
    elif isinstance(source, dict):
        # Currently only support a single frequency across networks
        freq = list(source.values())[0].frequency
    elif isinstance(source, Model):
        if freq is None:
            raise Exception("Frequency must be passed when extracting features from a model")
        if isinstance(freq, skrf.Frequency):
            freq = Frequency.from_skrf(freq)
    else:
        raise TypeError("Invalid type to extract_features")
    
    # Return the extracted features
    if isinstance(source, Model):
        return _extract_model_features(source, features, freq)
    else:
        return _extract_measured_features(source, features, freq)

def make_feature_function(
    model: Model,
    features: FeatureInputT,
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

def _format_features(features: FeatureInputT) -> list[FeatureT]:
    if isinstance(features, dict):
        raw_features = []
        for label, value in features.items():
            bodies = [value] if not isinstance(value[0], Sequence) else value
            raw_features.extend([(label, body) if isinstance(body, str) else (label, *body) for body in bodies])
    elif not isinstance(features, Sequence):
        raw_features = [features]
    else:
        raw_features = features

    features_out = []
    for raw_feature in raw_features:
        # Options now are 'alias', ('label', 'alias'), ('feature', ports), ('label', 'feature', ports)
        if isinstance(raw_feature, str) == 1:
            label = ''
            feature, ports = _parse_feature_alias(raw_feature)
        elif len(raw_feature) == 2:
            if isinstance(raw_feature[1], str):
                label = raw_feature[0]
                feature, ports = _parse_feature_alias(raw_feature[1])
            else:
                label = ''
                feature, ports = raw_feature
        else:
            label, feature, ports = raw_feature
        
        features_out.append((label, feature, ports))

    return features_out

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
    
def _extract_model_features(model: Model, features: list[FeatureT], freq: Frequency) -> jnp.ndarray:
    n_frequencies = len(freq)
    n_features = len(features)

    X = jnp.zeros((n_frequencies, n_features), dtype=jnp.complex128)
    for d, feature in enumerate(features):
        label, prop, (m, n) = feature[0], feature[1], feature[2]
        
        feature_model = model
        if label != '':
            feature_model = getattr(model, feature[0])

        if prop[2:4] == 'mn':
            xfn = getattr(feature_model, prop)
            x = xfn(freq,m,n)
        else:
            xfn = getattr(feature_model, prop)
            x = xfn(freq)[:,m,n]    
            
        X = X.at[:, d].set(x)
        
    return X

def _extract_measured_features(networks: dict[str, skrf.Network], features: list[FeatureT], freq: Frequency) -> jnp.ndarray:
    n_frequencies = len(freq)
    n_features = len(features)

    X = jnp.zeros((n_frequencies, n_features), dtype=jnp.complex128)
    for d, feature in enumerate(features):
        label, prop, (m, n) = feature[0], feature[1], feature[2]
        x = None
        
        ntwk = networks[label]
        if prop[2:4] == 'mn':
            i = prop.index('_')
            prop_new = prop[0:i]
            if len(prop) > i + 3:
                prop_new += prop[i+3:]
            prop = prop_new
        
        x = getattr(ntwk, prop)[:,m,n]
        X = X.at[:, d].set(x)
    return X    