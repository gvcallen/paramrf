import numpy as jnp
import skrf as rf

from pmrf.legacy.misc.math import dB20


"""
This file contains functions related to extracting "features" from networks e.g. S11 magnitude, S21 complex etc.
"""

class Feature:
    def __init__(self, mode, ports=(0, 0), scale='lin', weight=1.0):
        self.mode = mode
        self.ports = ports
        self.scale = scale
        self.weight = weight

    def __call__(self, network: rf.Network) -> jnp.ndarray:
        """
        Returns a single feature column vector of dimension F,
        where F is the number of network frequencies.
        """
        m, n = self.ports
        y = network.s[:, m, n]
        
        if self.mode == 'complex':
            pass
        elif self.mode == 'magnitude':
            y = jnp.abs(y)
        elif self.mode == 'real':
            y = jnp.real(y)
        elif self.mode == 'imaginary':
            y = jnp.imag(y)
        elif self.mode == 'phase':
            y = jnp.angle(y)
        else:
            raise Exception('Unknown network feature type')

        if self.scale == 'dB':
            y = dB20(y)

        return self.weight * y
    
# def extract_features(networks: rf.Network | list[rf.Network], features: list[Feature] | list[list[Feature]]) -> np.ndarray:
def extract_features(networks, features, ignore_imag=False) -> jnp.ndarray:
    """
    Returns a feature matrix of a given network with shape (F, D),
    where F is the number of network frequencies, and D is the number of features.
    If a list of networks is provided, D is calculated by the summing the number of features per network,
    and it is assumed that all networks have the same number of frequencies.
    """
    if ignore_imag:
        data_type = jnp.float64
    else:
        data_type = jnp.complex128
    
    if type(networks) == list:
        F = networks[0].frequency.npoints
        D = 0
        
        if type(features[0]) == list:
            for network_features in features:
                D += len(network_features)
            
            x = jnp.zeros((F, D), dtype=data_type)
            d = 0
            for network_features, network in zip(features, networks):
                for feature in network_features:
                    x[:, d] = feature(network)
                    d += 1
        else:
            D += len(features)
        
            x = jnp.zeros((F, D), dtype=data_type)
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
        x = jnp.zeros((F, D), dtype=data_type)
        for d, feature in enumerate(features):
            x[:, d] = feature(network)

        return x