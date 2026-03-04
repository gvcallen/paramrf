"""
Extracting features such as S-parameters from models and scikit-rf Networks.
"""
from typing import Sequence, Callable
import re

import skrf
import jax.numpy as jnp

from pmrf.constants import FeatureT, FeatureSpecT
from pmrf.models.model import Model
from pmrf.frequency import Frequency
from pmrf.network_collection import NetworkCollection

def extract_features(
    source: Model | skrf.Network | NetworkCollection,
    frequency: Frequency | None,
    features: FeatureSpecT,
    sparam_kind: str = 'all',
    dtype: jnp.dtype = jnp.complex128,
) -> jnp.ndarray:
    """
    Extract frequency-dependent features from a model or network into a unified matrix.

    Features are combined column-by-column, resulting in a matrix where rows correspond
    to frequency points and columns to the requested features. This supports extracting
    scalars (e.g., 's11_mag'), complex parameters, or custom functions from both
    simulation models and measured networks.


    Parameters
    ----------
    source : Model, skrf.Network, or NetworkCollection
        The source object. If a list of networks is provided, it is treated as a
        single stacked network with isolated ports. Missing networks in a collection
        should be specified using integers representing the port count at that index.
    frequency : pmrf.Frequency or skrf.Frequency, optional
        The frequency points for extraction.
        
        * **Required** if `source` is a Model.
        * **Optional** if `source` is a Network; defaults to the Network's native
          frequency points. If provided, the Network is interpolated to these points.
    features : str, list, dict, or tuple
        The features to extract. See Notes for formatting syntax.
    sparam_kind : {'transmission', 'reflection', 'all'} or None, optional
        Filters port expansion when generic features are requested (e.g., 's_re'
        without specific indices).
    dtype : data-type, optional
        The desired data-type for the output feature matrix.

    Returns
    -------
    feature_matrix : ndarray
        The extracted feature matrix of shape (M, N), where M is the number of
        frequency points and N is the total number of features.

    Notes
    -----
    **Feature Syntax**

    Features can be specified using string aliases, explicit tuples, or dictionaries
    for sub-components:

    * **String Aliases:** Convenient shorthands for standard parameters.

      * *Example:* ``'s11_db'``, ``'a21_deg'``, ``'s11_mag'``.
      * *Custom:* Any model attribute dependent on frequency can be accessed by name
        (e.g., ``'my_custom_gain'``).

    * **Explicit Tuples:** Defined as ``(path, parameter, index)``.

      * *Example:* ``('', 's_db', (0, 0))`` is equivalent to ``'s11_db'``.
      * The empty string ``''`` denotes the base model.

    * **Dictionaries:** Used to target submodels (via ``getattr``) or specific networks.

      * *Example:* ``{'lna': 's21_db'}`` extracts S21 from the 'lna' submodel.
      * *Nested:* ``{'front_end.lna': 's11'}`` accesses deep submodels.
      * *Measurement:* Maps to the label of a specific network in a collection.

    * **Lists:** A list containing any combination of the above will be flattened
      into columns.

    """
    if isinstance(features, Callable):
        return features(source)
    
    # We canonicalize the features to be flat (and parse them in the process)
    features = _canonicalize_features(features, source=source, sparam_kind=sparam_kind)
    
    # Get the frequency and format the sources
    if isinstance(source, skrf.Network):
        source_cpy = source.copy()
        source_cpy.name = ''
        source = NetworkCollection([source_cpy])
        frequency = frequency or source.common_frequency()
    elif isinstance(source, NetworkCollection):
        # Currently only support a single frequency across networks
        frequency = frequency or source.common_frequency()
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

def _canonicalize_features(features: FeatureSpecT, *, base_label='', source: Model | skrf.Network | NetworkCollection | None = None, sparam_kind: str = 'all') -> list[FeatureT]:
    # For the dict case, we just recursively call format features for each attribute and return early
    if isinstance(features, dict):
        features_out = []
        for attr, attr_features in features.items():
            if isinstance(source, NetworkCollection):
                src_obj = source[attr]
            else:
                src_obj = getattr(source, attr)
            features_out.extend(_canonicalize_features(attr_features, base_label=attr, source=src_obj, sparam_kind=sparam_kind))
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
        if no_ports_passed and all_labels_equal:
            if sparam_kind == 'all':
                port_tuples = source.port_tuples
            elif sparam_kind == 'reflection':
                port_tuples = [pt for pt in source.port_tuples if pt[0] == pt[1]]
            elif sparam_kind == 'transmission':
                port_tuples = [pt for pt in source.port_tuples if pt[0] != pt[1]]
            else:
                raise Exception('Unknown S-parameter type for port expansion')

            label = features_out[0][0]
            features_out = [(label, feature_out[1], ports) for ports in port_tuples for feature_out in features_out]

    return features_out

def _parse_feature_alias(alias: str) -> tuple[FeatureT, tuple[int, int]]:
    """
    Convert a feature alias like 's11_mag' to a feature tuple like ('s_mag', (0, 0)).
    
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
    extracted_cols = []
    
    for feature in features:
        label, prop, (m, n) = feature[0], feature[1], feature[2]
        
        feature_model = model
        if label != '':
            sublabels = label.split('.')
            for sublabel in sublabels:
                feature_model = getattr(feature_model, sublabel)

        if m >= feature_model.nports or n >= feature_model.nports:
            raise Exception(f'Property {prop}{m+1}{n+1} requested but network is a {model.nports}-port')        

        if prop[2:4] == 'mn':
            xfn = getattr(feature_model, prop)
            x = xfn(freq,m,n)
        elif m != -1 and n != -1:
            xfn = getattr(feature_model, prop)
            x = xfn(freq)[:,m,n]
        else:
            xfn = getattr(feature_model, prop)
            x = xfn(freq)
            
        # Append the 1D array to our Python list
        extracted_cols.append(x)
        
    # Let JAX compile a single concatenation operation
    return jnp.column_stack(extracted_cols).astype(dtype)    
    
# def _extract_model_features(model: Model, features: list[FeatureT], freq: Frequency, dtype: jnp.dtype) -> jnp.ndarray:
#     n_frequencies = len(freq)
#     n_features = len(features)

#     X = jnp.zeros((n_frequencies, n_features), dtype=dtype)
#     for d, feature in enumerate(features):
#         label, prop, (m, n) = feature[0], feature[1], feature[2]
        
#         feature_model = model
#         if label != '':
#             sublabels = label.split('.')
#             for sublabel in sublabels:
#                 feature_model = getattr(feature_model, sublabel)

#         if m >= feature_model.nports or n >= feature_model.nports:
#             raise Exception(f'Property {prop}{m+1}{n+1} requested but network is a {model.nports}-port')        

#         if prop[2:4] == 'mn':
#             xfn = getattr(feature_model, prop)
#             x = xfn(freq,m,n)
#         elif m != -1 and n != -1:
#             xfn = getattr(feature_model, prop)
#             x = xfn(freq)[:,m,n]
#         else:
#             xfn = getattr(feature_model, prop)
#             x = xfn(freq)
            
#         X = X.at[:, d].set(x)
        
#     return X

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
            raise Exception(f'Property {prop} for ports {m+1}{n+1} requested but network is a {ntwk.nports}-port')
        
        if prop[2:4] == 'mn':
            i = prop.index('_')
            prop_new = prop[0:i]
            if len(prop) > i + 3:
                prop_new += prop[i+3:]
            prop = prop_new
        
        x = getattr(ntwk, prop)[:,m,n]
        X = X.at[:, d].set(x)
    return X