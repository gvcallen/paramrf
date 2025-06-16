from dataclasses import dataclass
import skrf

from pmrf._math import dB20
from pmrf._numpy import numpy as np
# from pmrf._model import Model


"""
This file contains functions related to extracting "features" from models e.g. S11 magnitude, S21 complex etc.
"""
@dataclass
class FeatureExtractor:
    mode: str = 'complex'
    property: str = 's'
    ports: tuple[int, int] = (0, 0)
    scale: str = 'lin'

    def extract_from_model(self, model, x: np.ndarray) -> np.ndarray:
        """
        x is in units of the model's frequency
        """
        m, n = self.ports
        y = getattr(model, self.property)(x)[:, m, n]
        return self._process_property(y)

    def extract_from_network(self, network: skrf.Network) -> np.ndarray:
        m, n = self.ports
        y = network.s[:, m, n]
        return self._process_property(y)
    
    def _process_property(self, y):
        if self.mode == 'complex':
            pass
        elif self.mode == 'magnitude':
            y = np.abs(y)
        elif self.mode == 'real':
            y = np.real(y)
        elif self.mode == 'imaginary':
            y = np.imag(y)
        elif self.mode == 'phase':
            y = np.angle(y)
        else:
            raise Exception('Unknown network feature type')

        if self.scale == 'dB':
            y = dB20(y)

        return y            

class FeatureExtractorSet:
    """
    A set of features to be extracted.
    """
    def __init__(self, features: list[FeatureExtractor] | list[str] = None):
        self.features = features or [FeatureExtractor()]

    def __call__(self, networks: list[skrf.Network]) -> np.ndarray:
        """
        Returns a feature matrix of a given network with shape (F, D),
        where F is the number of network frequencies, and D is the number of features.
        If a list of networks is provided, D is calculated by the summing the number of features per network,
        and it is assumed that all networks have the same number of frequencies.
        """
        features = self.features

        if type(networks) == list:
            F = networks[0].frequency.npoints
            D = 0
            
            if type(features[0]) == list:
                for network_features in features:
                    D += len(network_features)
                
                x = np.zeros((F, D), dtype=np.complex128)
                d = 0
                for network_features, network in zip(features, networks):
                    for feature in network_features:
                        x[:, d] = feature(network)
                        d += 1
            else:
                D += len(features)
            
                x = np.zeros((F, D), dtype=np.complex128)
                d = 0
                for network in networks:
                    for feature in features:
                        x[:, d] = feature(network)
                        d += 1        
                
            return x
        else:
            network = networks
            F = network.frequency.npoints
            D = len(features)
            x = np.zeros((F, D), dtype=np.complex128)
            for d, feature in enumerate(features):
                x[:, d] = feature(network)

            return x