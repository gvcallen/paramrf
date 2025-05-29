import skrf as rf

from paramrf.misc.inspection import get_args_of
from paramrf.modeling import CompositeNetwork
from paramrf.modeling.elements.lumped import Capacitor
from paramrf.modeling.models.lines import MicrostripLine
from paramrf.modeling.models.connectors import SMAEdgeConnector

"""
Misc circuit models for quick testing. Generally it is convenient to all these models to take sub-network arguments to be None in their constructors.
Then, it is trivial to make use of these networks using the CircuitFitter.
"""

class SMAMicrostripOpen(CompositeNetwork):
    """
    A model for an SMA edge connector, followed by an open-circuited microstrip line.
    """
    def __init__(self, sma: SMAEdgeConnector | None = None, mstrip: MicrostripLine | None = None, edge: Capacitor | None = None, **kwargs):
        frequency = kwargs['frequency']
        
        sma = sma or SMAEdgeConnector(frequency=frequency)
        mstrip = mstrip or MicrostripLine(frequency=frequency)
        edge = edge or Capacitor(frequency=frequency, name='edge')

        networks = {
            'sma': sma,
            'mstrip': mstrip,
            'edge': edge,
        }
        
        super().__init__(networks, nports=1, **kwargs)
        
    def compute(self):
        ground = rf.Circuit.Ground(self.frequency, 'ground', z0=self.z0_port)        
        self.s = (self.sma ** self.mstrip ** self.edge ** ground).s