from typing import Sequence
import re

import skrf
import jax.numpy as jnp

from pmrf._constants import FeatureT, FeatureInputT
from pmrf._model import Model
from pmrf._frequency import Frequency

def extract_features(
    source: Model | skrf.Network | dict[str, skrf.Network],
    features: FeatureInputT,
    freq: Frequency | skrf.Frequency = None,
    dtype: jnp.dtype = jnp.complex128,
) -> jnp.ndarray:
    """Extracts features from a model or a network.
    
    This function allows for an arbitrary number of features (e.g. ['s11', 'a21_mag'])
    to be easily extracted from a model or a measured network. The resultant features
    are combined column-by-column into a matrix with frequency in the row dimension.
    
    Features can either be specified by convenient aliases using strings, or by their full structure.
    As some examples to demonstrate the possibilities:
    - To extract S11 magnitude, specify either the alias 's11_db' or the full tuple `('', 's_db' (0, 0))`.
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
        source (Model | skrf.Network | dict[str, skrf.Network]):        The source model or network(s) to extract the features from,
                                                                        with missing networks specified using integers for the number of ports at that index.
                                                                        Lists of networks are treated as forming one, stack network with isolated ports.
        features (FeatureInputT):                                       The features to extract, as described in detail above.
        freq (pmrf.Frequency | skrf.Frequency, optional):               The frequency to extract the features at. This will become the row dimension of the resultant matrix.
                                                                        This must be passed for `Model` sources. Defaults to `None` e.g. for measured networks,
                                                                        in which case the network's internal frequency is used. Otherwise, the network is i508e76ad-05af-4e71-8c22-053f37e3a62dnterpolated.
        dtype (jnp.dtype, optional):                                    The data type of the final out feature matrix.

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
        return _extract_model_features(source, features, freq, dtype=dtype)
    else:
        return _extract_measured_features(source, features, freq, dtype=dtype)

def _format_features(features: FeatureInputT) -> list[FeatureT]:
    if isinstance(features, dict):
        raw_features = []
        for label, value in features.items():
            if isinstance(value, str):
                bodies = [value]
            elif isinstance(value, tuple):
                bodies = [value]
            else: # Sequence
                bodies = value
            raw_features.extend([(label, body) if isinstance(body, str) else (label, *body) for body in bodies])
    elif isinstance(features, str):
        raw_features = [features]
    elif not isinstance(features, Sequence):
        raw_features = [features]
    else:
        raw_features = features

    features_out = []
    for raw_feature in raw_features:
        # Options now are 'alias', ('label', 'alias'), ('feature', ports), ('label', 'feature', ports)
        if isinstance(raw_feature, str):
            raw_feature_split = raw_feature.split('.')
            if len(raw_feature_split) > 1:
                label = ''.join(raw_feature_split[0:-1])
                raw_feature = raw_feature_split[-1]
            else:
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
    """
    Converts a feature alias like 's11_mag' to a feature tuple like ('s_mag', (0, 0)).
    Supports:
      - Feature names with or without two-digit port numbers
      - Optional suffixes (e.g., '_mag', '_db')
      - Returns (-1, -1) for features without ports
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

def _extract_measured_features(networks: dict[str, skrf.Network], features: list[FeatureT], freq: Frequency | skrf.Frequency, dtype: jnp.dtype) -> jnp.ndarray:
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
        
        if prop[2:4] == 'mn':
            i = prop.index('_')
            prop_new = prop[0:i]
            if len(prop) > i + 3:
                prop_new += prop[i+3:]
            prop = prop_new
        
        x = getattr(ntwk, prop)[:,m,n]
        X = X.at[:, d].set(x)
    return X    