from dataclasses import dataclass
import skrf

from pmrf.frequency import Frequency

from pmrf._math import dB20
from pmrf._numpy import numpy as np
from pmrf._numpy import USE_JAX
from pmrf.model import Model
from pmrf.system import ModelSystem


"""
This file contains functions related to extracting "features" from models e.g. S11 magnitude, S21 complex etc.
"""
@dataclass
class Feature:
    mode: str = 'complex'
    property: str = 's'
    ports: tuple[int, int] = (0, 0)
    scale: str = 'lin'

def features_from_strings(strings: list[str]) -> list[Feature]:
    features = []
    raise ValueError("Not yet implemented")
    for str in strings:
        pass

def extract_features(source: Model | ModelSystem | skrf.Network | list[skrf.Network], features: list[Feature], freq: Frequency = None) -> np.ndarray:
    # We use explicit defaults because cost is quite a common high-level user requirement
    # TODO optimize jax cases
    if freq is None:
        freq = source.frequency

    if isinstance(source, Model) or isinstance(source, skrf.Network):
        n_frequencies = len(freq)
        n_features = len(features)

        X = np.zeros((n_frequencies, n_features), dtype=np.complex128)
        for d, feature in enumerate(features):
            if USE_JAX:
                X = X.at[:, d].set(_extract_feature(source, feature, freq=freq))
            else:
                X[:, d] = _extract_feature(source, feature, freq=freq)
    else:
        X = np.zeros((freq.npoints, len(features)), dtype=np.complex128)
        d = 0
        if isinstance(source, ModelSystem):
            sources = source.models
        else:
            sources = source
        for source in sources:
            for feature in features:
                if USE_JAX:
                    X.at[:, d].set(_extract_feature(source, feature, freq=freq))
                else:
                    X[:, d] = _extract_feature(source, feature, freq=freq)
                d += 1
    
    return X

    
def _extract_feature(source: Model | skrf.Network, feature: Feature, freq: Frequency = None) -> np.ndarray:
    m, n = feature.ports
    if isinstance(source, Model):
        y = getattr(source, source.primary_property)(freq)[:, m, n]
    elif isinstance(source, skrf.Network):
        y = source.s[:, m, n]

    if feature.mode == 'complex':
        pass
    elif feature.mode == 'magnitude':
        y = np.abs(y)
    elif feature.mode == 'real':
        y = np.real(y)
    elif feature.mode == 'imaginary':
        y = np.imag(y)
    elif feature.mode == 'phase':
        y = np.angle(y)
    else:
        raise Exception('Unknown network feature type')

    if feature.scale == 'dB':
        y = dB20(y)

    return y