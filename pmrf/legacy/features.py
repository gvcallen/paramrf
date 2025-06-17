from dataclasses import dataclass
import skrf

from pmrf.frequency import Frequency

from pmrf._math import dB20
from pmrf._numpy import numpy as np
from pmrf.model import Model


"""
This file contains functions related to extracting "features" from models e.g. S11 magnitude, S21 complex etc.
"""
@dataclass
class Feature:
    mode: str = 'complex'
    property: str = 's'
    ports: tuple[int, int] = (0, 0)
    scale: str = 'lin'

def extract_from_model(model: Model, feature: Feature, freq: Frequency) -> np.ndarray:
    m, n = feature.ports
    y = getattr(model, self.property)(freq)[:, m, n]
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