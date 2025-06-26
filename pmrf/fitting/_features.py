from typing import Sequence

import skrf
import pmrf as prf
import pmrf.numpy as np
from pmrf._model import Model
from pmrf._frequency import Frequency
from pmrf.numpy import USE_JAX

Feature = tuple[str, tuple[int, int]]

def extract_features(source: Model | skrf.Network | Sequence[skrf.Network], features: list[Feature] | list[list[Feature]], freq: prf.Frequency | skrf.Frequency = None) -> np.ndarray:
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
    if isinstance(features[0], list):
        features = [feature for features_inner in features for feature in features_inner]
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

    n_frequencies = len(freq)
    n_features = len(features)
    
    X = np.zeros((n_frequencies, n_features), dtype=np.complex128)
    for d, feature in enumerate(features):
        prop = feature[0]
        m, n = feature[1]
        x = None
        
        if isinstance(source, Model):
            if prop[-2:] == 'mn' and hasattr(source, prop):
                xfn = getattr(source, prop)
                x = xfn(freq,m,n)
            else:
                xfn = getattr(source, prop)
                x = xfn(freq)[:,m,n]
        else: # isinstance(source, list[Network])
            x = None
            offset = 0 # of the full stacked network
            for ntwk in source:
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
            
        if USE_JAX:
            X = X.at[:, d].set(x)
        else:
            X[:, d] = x
    return X