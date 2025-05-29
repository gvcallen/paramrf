import numpy as np
import skrf as rf
from skrf.media import Media

from pmrf.modeling import ParametricNetwork, ComputableNetwork
from pmrf.misc.inspection import get_args_of

   
class Capacitor(ParametricNetwork):
    def __init__(self, C=1.0, name='capacitor', **kwargs):
        super().__init__(params=get_args_of(float), nports=2, name=name, **kwargs)
    
    def compute(self) -> None:
        smat = self.media.capacitor(self.C).s
        self.s = smat
        
        
class Inductor(ParametricNetwork):
    def __init__(self, L=1.0, name='inductor', **kwargs):
        super().__init__(params=get_args_of(float), nports=2, name=name, **kwargs)
    
    def compute(self) -> None:
        self.s = self.media.inductor(self.L).s
        
        
class Resistor(ParametricNetwork):
    def __init__(self, R=1.0, name='resistor', terminated=False, **kwargs):
        self.terminated = terminated
        if terminated:
            nports = 1
        else:
            nports = 2
        super().__init__(params=get_args_of(float), nports=nports, name=name, **kwargs)
    
    def compute(self) -> None:
        if self.terminated:
            self.s = (self.media.resistor(self.R) ** self.media.short()).s
        else:
            self.s = self.media.resistor(self.R).s
        

class Transformer(ParametricNetwork):
    def __init__(self, N = 1, name = 'transformer', **kwargs):
        super().__init__(params=get_args_of(float), nports=2, name=name, **kwargs)
        
    def compute(self):
        self.s = 0.5 * np.ones((self.frequency.npoints, 4, 4), dtype=complex)
        self.s[:, 0, 3] *= -1
        self.s[:, 1, 2] *= -1
        self.s[:, 2, 1] *= -1
        self.s[:, 3, 0] *= -1
        
        
class SourceConverter(ComputableNetwork):
    def __init__(self, name = 'source_converter', **kwargs):
        super().__init__(params=get_args_of(float), nports=3, name=name, **kwargs)
        
    def compute(self):
        tf = Transformer(media=self.media)
        port1 = rf.Circuit.Port(self.frequency, name='port1', z0=self.z0_port)
        port2 = rf.Circuit.Port(self.frequency, name='port2', z0=self.z0_port)
        port3 = rf.Circuit.Port(self.frequency, name='port3', z0=self.z0_port)
        ground = rf.Circuit.Ground(self.frequency, name='ground', z0=self.z0_port)
    
        cnx = [
            [(tf, 0), (port1, 0)],
            [(tf, 1), (ground, 0)],
            [(tf, 2), (port2, 0)],
            [(tf, 3), (port3, 0)],
        ]

        self.s = rf.Circuit(cnx).network.s
        
def transformer(media: Media, N = 1, **kwargs) -> rf.Network:
    r"""
    An ideal transformer of winding 1 : N, represented as a four-port network.
    Currently only N = 1 is supported. The primary coil is between nodes 1 and 2,
    and the secondary coil is between nodes 3 and 4.

    Parameters
    ----------
    media : the media
    N : number
            the winding ratio 1 : N
    \*\*kwargs : key word arguments
        passed to :func:`match`, which is called initially to create a
        'blank' network.

    Returns
    -------
    transformer : :class:`~skrf.network.Network` object
        ideal transformer

    """
    result = rf.Network(**kwargs)
    result.frequency = media.frequency
    
    result.s = 0.5 * np.ones((media.frequency.npoints, 4, 4), dtype=complex)
    result.s[:, 0, 3] *= -1
    result.s[:, 1, 2] *= -1
    result.s[:, 2, 1] *= -1
    result.s[:, 3, 0] *= -1
    
    result.port_modes = np.array(["S"] * result.nports)
    if media.z0_port is None:
        z0 = media.z0
    else:
        z0 = media.z0_port

    result.z0 = z0
    return result

def source_converter(media: Media, **kwargs) -> rf.Network:
    r"""
    An ideal transformer for converting a single-ended source to a two-terminal source.
    The single end is port 1, and the two-terminal end is ports 2 and 3.

    Parameters
    ----------
    media : the media
    \*\*kwargs : key word arguments
        passed to :func:`match`, which is called initially to create a
        'blank' network.

    Returns
    -------
    source_converter : :class:`~skrf.network.Network` object
        source converter network

    """
    tf = transformer(media, name='transformer')
    port1 = rf.Circuit.Port(media.frequency, name='port1', z0=media.z0)
    port2 = rf.Circuit.Port(media.frequency, name='port2', z0=media.z0)
    port3 = rf.Circuit.Port(media.frequency, name='port3', z0=media.z0)
    ground = rf.Circuit.Ground(media.frequency, name='ground', z0=media.z0)
    
    cnx = [
        [(tf, 0), (port1, 0)],
        [(tf, 1), (ground, 0)],
        [(tf, 2), (port2, 0)],
        [(tf, 3), (port3, 0)],
    ]

    return rf.Circuit(cnx).network