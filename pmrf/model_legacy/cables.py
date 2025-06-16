import skrf as rf

from _inspection_legacy import get_args_of
from pmrf.models.models.connectors import SMAConnector
from pmrf.models.models.lines import DatasheetCoax
from pmrf._core import ParametricNetwork, CompositeNetwork
from pmrf.rf.passive import available_gain

class CoaxialCable(CompositeNetwork):
    def __init__(self, connector1: ParametricNetwork = None, line: ParametricNetwork = None, connector2: ParametricNetwork = None, **kwargs):
        connector1 = connector1 or SMAConnector(name='sma1')
        line = line or DatasheetCoax(name='line')
        connector2 = connector2 or SMAConnector(name='sma2')
        
        networks = {
            'connector1': connector1,
            'line': line,
            'connector2': connector2,
        }
        
        super().__init__(networks, nports=2, **kwargs)

    def compute(self):
        self.s = (self.connector1 ** self.line ** self.connector2).s