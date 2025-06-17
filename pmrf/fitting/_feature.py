from dataclasses import dataclass
import skrf
import re

import pmrf.numpy as np
from pmrf.numpy import USE_JAX

from pmrf._frequency import Frequency
from pmrf._math import dB20
from pmrf._model import Model
from pmrf._system import SystemModel

"""
This file contains functions related to extracting "features" from models e.g. S11 magnitude, S21 complex etc.
"""
@dataclass
class Feature:
    property: str = 's'
    ports: tuple[int, int] = (0, 0)
    mode: str = 'complex'
    scale: str = 'lin'

    @classmethod
    def from_string(cls, feature_str: str) -> 'Feature':
        """
        Parses a user-friendly string into a structured Feature object.

        Handles formats like 's11', 's21_db', 'a11_mag', 'y12_deg', etc.

        Args:
            feature_str: The string representation of the feature.

        Returns:
            A Feature dataclass instance.

        Raises:
            ValueError: If the string format is invalid.
        """
        feature_str = feature_str.lower().strip()

        # Breakdown of the regex: ^([syza])(\d)(\d)(?:_([a-z]+))?$
        # ^             - Start of the string
        # ([syza])      - Group 1: Matches and captures one character from the set 's', 'y', 'z', 'a'.
        # (\d)          - Group 2: Matches and captures a single digit (the first port number).
        # (\d)          - Group 3: Matches and captures a single digit (the second port number).
        # (?:           - Start of a non-capturing group for the optional suffix.
        #   _           - Matches the literal underscore separator.
        #   ([a-z]+)    - Group 4: Matches and captures one or more lowercase letters (the scale/mode).
        # )?            - End of the non-capturing group, making the whole suffix optional.
        # $             - End of the string
        pattern = re.compile(r"^([syza])(\d)(\d)(?:_([a-z]+))?$")
        match = pattern.match(feature_str)

        if not match:
            raise ValueError(
                f"Invalid feature string format: '{feature_str}'. "
                f"Expected format like 's11', 's21_db', etc."
            )

        prop, port1_str, port2_str, suffix = match.groups()

        # Convert 1-based port strings to 0-based integer indices
        ports = (int(port1_str) - 1, int(port2_str) - 1)

        # --- Determine mode and scale based on the suffix ---
        mode = 'complex'
        scale = 'lin'

        if suffix:
            # This mapping defines how suffixes translate to mode and scale.
            # It's easy to extend with more options.
            suffix_map = {
                'db':      {'mode': 'mag', 'scale': 'db'},
                'mag':     {'mode': 'mag', 'scale': 'lin'},
                'abs':     {'mode': 'mag', 'scale': 'lin'}, # Alias for mag
                'deg':     {'mode': 'phase', 'scale': 'deg'},
                'rad':     {'mode': 'phase', 'scale': 'rad'},
                're':      {'mode': 'real', 'scale': 'lin'},
                'real':    {'mode': 'real', 'scale': 'lin'}, # Alias for re
                'im':      {'mode': 'imag', 'scale': 'lin'},
                'imag':    {'mode': 'imag', 'scale': 'lin'}, # Alias for im
            }
            if suffix in suffix_map:
                mode = suffix_map[suffix]['mode']
                scale = suffix_map[suffix]['scale']
            else:
                raise ValueError(f"Unknown suffix '_{suffix}' in feature string '{feature_str}'.")

        return Feature(
            property=prop,
            ports=ports,
            mode=mode,
            scale=scale
        )    

def extract_features(source: Model | SystemModel | skrf.Network | list[skrf.Network], features: list[Feature], freq: Frequency = None) -> np.ndarray:
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
                X = X.at[:, d].set(extract_feature(source, feature, freq=freq))
            else:
                X[:, d] = extract_feature(source, feature, freq=freq)
    else:
        X = np.zeros((freq.npoints, len(features)), dtype=np.complex128)
        d = 0
        if isinstance(source, SystemModel):
            sources = source.models
        else:
            sources = source
        for source in sources:
            for feature in features:
                if USE_JAX:
                    X.at[:, d].set(extract_feature(source, feature, freq=freq))
                else:
                    X[:, d] = extract_feature(source, feature, freq=freq)
                d += 1
    
    return X

    
def extract_feature(source: Model | skrf.Network, feature: Feature, freq: Frequency = None) -> np.ndarray:
    m, n = feature.ports
    if isinstance(source, Model):
        y = getattr(source, feature.property)(freq)[:, m, n]
    elif isinstance(source, skrf.Network):
        y = source.s[:, m, n]

    if feature.mode == 'complex':
        pass
    elif feature.mode == 'mag' or feature.mode == 'magnitude':
        y = np.abs(y)
    elif feature.mode == 'real' or feature.mode == 're':
        y = np.real(y)
    elif feature.mode == 'imaginary' or feature.mode == 'imag'or feature.mode == 'im':
        y = np.imag(y)
    elif feature.mode == 'phase':
        if feature.scale == 'deg':            
            y = np.angle(y, deg=True)
        else:
            y = np.angle(y, deg=False)
    else:
        raise Exception('Unknown network feature type')

    if feature.scale == 'dB' or feature.scale == 'db':
        y = dB20(y)

    return y