from typing import Sequence
import re

import skrf
import jax.numpy as jnp

from pmrf.constants import FeatureT, FeatureInputT
from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.network_collection import NetworkCollection
from pmrf._util import defining_class

def extract_features(
    source: Model | skrf.Network | NetworkCollection,
    frequency: Frequency | None,
    features: FeatureInputT,
    dtype: jnp.dtype = jnp.complex128,
) -> jnp.ndarray:
    """Extracts features from a model or a network.
    
    This function allows for an arbitrary number of features (e.g. ['s11', 'a21_mag'])
    to be easily extracted from a model or a measured network. The resultant features
    are combined column-by-column into a matrix with frequency in the row dimension.
    
    Features can either be specified by convenient aliases using strings, or by their full structure.
    As some examples to demonstrate the possibilities:
    - To extract S11 magnitude, specify either the alias 's11_db' or the full tuple `('', 's_db', (0, 0))`.
      Note that, for the tuple, the empty string at the beginning represents the base model (expanded on below).
    - To extract e.g. the phase of the B parameter of the ABCD matrix, specify 'a21_deg'.
    - To extract any other feature in the model that is a function of frequency (e.g. a custom user function), simply use 'myfeature' instead of 's11'.
    - To extract features from a submodel or specific network, specify a dictionary with the submodel as a key
      e.g. {'submodel': 's11_db'}. For models, this extracts a feature from a submodel that must be retrievable via `getattr`.
      Submodels can also be nested e.g. {'src1.submodel1.submodel2': 's11_db'}. For measured networks,
      this extracts a feature from the corresponding network with that label in the dictionary.
      For the above example, is is converted to the feature tuple ('src_name', 's_db', (0, 0)').
    - For a list of features, specify a list of any of the above (or, equivalently, a dictionary of lists).

    Args:
        source (Model | skrf.Network | NetworkCollection):              The source model or network(s) to extract the features from,
                                                                        with missing networks specified using integers for the number of ports at that index.
                                                                        Lists of networks are treated as forming one, stack network with isolated ports.
        features (FeatureInputT):                                       The features to extract, as described in detail above.
        frequency (pmrf.Frequency | skrf.Frequency, optional):          The frequency to extract the features at. This will become the row dimension of the resultant matrix.
                                                                        This must be passed for `Model` sources. Defaults to `None` e.g. for measured networks,
                                                                        in which case the network's internal frequency is used. Otherwise, the network is i508e76ad-05af-4e71-8c22-053f37e3a62dnterpolated.
        dtype (jnp.dtype, optional):                                    The data type of the final out feature matrix.

    Returns:
        np.ndarray: The feature matrix of size M x N, where M is the number of frequencies and N is the number of features.
    """    
    # We format the features to be flat (and parse them in the process)
    features = format_features(features, source=source)
    
    # Get the frequency and format the sources
    if isinstance(source, skrf.Network):
        source_cpy = source.copy()
        source_cpy.name = ''
        source = NetworkCollection([source_cpy])
        frequency = source.frequency
    elif isinstance(source, NetworkCollection):
        # Currently only support a single frequency across networks
        frequency = source.frequency
    elif isinstance(source, Model):
        if frequency is None:
            raise Exception("Frequency must be passed when extracting features from a model")
        if isinstance(frequency, skrf.Frequency):
            frequency = Frequency.from_skrf(frequency)
    else:
        raise TypeError("Invalid type to extract_features")
    
    # Return the extracted features
    if isinstance(source, Model):
        return _extract_model_features(source, features, frequency, dtype=dtype)
    else:
        return _extract_measured_features(source, features, frequency, dtype=dtype)

def format_features(features: FeatureInputT, *, base_label='', source: Model | skrf.Network | NetworkCollection | None = None) -> list[FeatureT]:
    # For the dict case, we just recursively call format features for each attribute and return early
    if isinstance(features, dict):
        features_out = []
        for attr, attr_features in features.items():
            if isinstance(source, NetworkCollection):
                src_obj = source[attr]
            else:
                src_obj = getattr(source, attr)
            features_out.extend(format_features(attr_features, base_label=attr, source=src_obj))
        return features_out
    
    # For the general case we get the features into a sequence of aliases or feature tuples and then expand
    if isinstance(features, str):
        raw_features = [features]
    elif not isinstance(features, Sequence):
        raw_features = [features]
    else:
        raw_features = features

    features_out = []
    for raw_feature in raw_features:
        # Allowed options for each raw feature are:
        #   1) 'alias'
        #   2) ('label', 'alias')
        #   3) ('feature', ports)
        #   4) ('label', 'feature', ports)

        # Option 1:
        if isinstance(raw_feature, str):
            raw_feature_split = raw_feature.split('.')
            if len(raw_feature_split) > 1:
                label = ''.join(raw_feature_split[0:-1])
                raw_feature = raw_feature_split[-1]
            else:
                label = base_label
            feature, ports = _parse_feature_alias(raw_feature)
        # Option 2
        elif len(raw_feature) == 2 and isinstance(raw_feature[1], str):
            label = raw_feature[0]
            feature, ports = _parse_feature_alias(raw_feature[1])
        # Option 3
        elif len(raw_feature) == 2:
            label = base_label
            feature, ports = raw_feature
        # Option 4
        else:
            label, feature, ports = raw_feature    
        features_out.append((label, feature, ports))

    if source is not None:
        no_ports_passed = all([feature_out[2] == (-1, -1) for feature_out in features_out])
        all_labels_equal = all([feature_out[0] == features_out[0][0] for feature_out in features_out])
        # TODO fix defining_class (not working for e.g. s_mag currently)
        # base_features_only = all([defining_class(source, feature_out[1]) is Model for feature_out in features_out])
        if no_ports_passed and all_labels_equal:
            label = features_out[0][0]
            port_tuples = source.port_tuples
            features_out = [(label, feature_out[1], ports) for ports in port_tuples for feature_out in features_out]

    return features_out

def _parse_feature_alias(alias: str) -> tuple[FeatureT, tuple[int, int]]:
    """
    Converts a feature alias like 's11_mag' to a feature tuple like ('s_mag', (0, 0)).
    Supports:
      - Feature names with or without two-digit port numbers.
      - Optional suffixes (e.g., '_mag', '_db')
      - Returns (-1, -1) for features without ports.
    """
    match = re.match(r'^([a-zA-Z]+)(\d)?(\d)?(.*)$', alias)
    if not match:
        raise ValueError(f"Invalid feature alias format: '{alias}'")

    prefix = match.group(1)
    port1 = match.group(2)
    port2 = match.group(3)
    suffix = match.group(4)

    if port1 is not None and port2 is not None:
        ports = (int(port1) - 1, int(port2) - 1)
    else:
        ports = (-1, -1)

    return (prefix + suffix, ports)
    
def _extract_model_features(model: Model, features: list[FeatureT], freq: Frequency, dtype: jnp.dtype) -> jnp.ndarray:
    n_frequencies = len(freq)
    n_features = len(features)

    X = jnp.zeros((n_frequencies, n_features), dtype=dtype)
    for d, feature in enumerate(features):
        label, prop, (m, n) = feature[0], feature[1], feature[2]

        if m >= model.nports or n >= model.nports:
            raise Exception(f'Property {prop}{m+1}{n+1} requested but network is a {model.nports}-port')        
        
        feature_model = model
        if label != '':
            sublabels = label.split('.')
            for sublabel in sublabels:
                feature_model = getattr(feature_model, sublabel)

        if prop[2:4] == 'mn':
            xfn = getattr(feature_model, prop)
            x = xfn(freq,m,n)
        elif m != -1 and n != -1:
            xfn = getattr(feature_model, prop)
            x = xfn(freq)[:,m,n]
        else:
            xfn = getattr(feature_model, prop)
            x = xfn(freq)
            
        X = X.at[:, d].set(x)
        
    return X

def _extract_measured_features(networks: NetworkCollection, features: list[FeatureT], freq: Frequency | skrf.Frequency, dtype: jnp.dtype) -> jnp.ndarray:
    n_frequencies = len(freq)
    n_features = len(features)
    
    if isinstance(freq, Frequency):
        freq = freq.to_skrf()

    X = jnp.zeros((n_frequencies, n_features), dtype=dtype)
    for d, feature in enumerate(features):
        label, prop, (m, n) = feature[0], feature[1], feature[2]

        x = None
        
        ntwk = networks[label]
        if ntwk.frequency != freq:
            ntwk = ntwk.interpolate(freq)

        if m >= ntwk.nports or n >= ntwk.nports:
            raise Exception(f'Property {prop}{m+1}{n+1} requested but network is a {ntwk.nports}-port')
        
        if prop[2:4] == 'mn':
            i = prop.index('_')
            prop_new = prop[0:i]
            if len(prop) > i + 3:
                prop_new += prop[i+3:]
            prop = prop_new
        
        x = getattr(ntwk, prop)[:,m,n]
        X = X.at[:, d].set(x)
    return X