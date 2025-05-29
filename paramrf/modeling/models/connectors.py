from skrf.media import DefinedGammaZ0, DefinedAEpTandZ0

from paramrf.modeling import ParametricNetwork
from paramrf.modeling.models.lines import DatasheetCoax
from paramrf.modeling.elements.lumped import Resistor
from paramrf.misc.inspection import get_args_of

class SMAEdgeConnector(ParametricNetwork):
    """
    An SMA edge connector, with port 1 being the part of the SMA that can be connected to, and port 2 connecting to the PCB.
    Current, it is simply modelled as a piece of transmission line followed by an inductor representing the portion of the pin that protrudes.
    """
    def __init__(self, zn=50.0, len=1.0, epr=1.0, L=0.0, name='sma', **kwargs):
        super().__init__(params=get_args_of(float), nports=2, name=name, **kwargs)
        
    def compute(self):
        zn, len, epr, L = self.params['zn'], self.params['len'], self.params['epr'], self.params['L']

        media_wire = DefinedGammaZ0(frequency=self.frequency, z0=self.z0[:,1])
        media_tline = DefinedAEpTandZ0(frequency=self.frequency, z0=zn, z0_port=self.z0[:,0], ep_r=epr)

        # Elements, ordered from the coax to the microstrip
        tline = media_tline.line(len, 'm', name='tline')
        inductor = media_wire.inductor(L, name='inductor')

        self.s = (tline ** inductor).s

class SMAConnector(ParametricNetwork):
    """
    A lossy SMA connector that is soldered on the end of a coaxial line. Currently modeled simply as a coaxial line with only skin effect losses.
    """
    def __init__(self, zn=50.0, len=10.0e-3, epr=1.0, k1=0.0, name='sma', **kwargs):
        super().__init__(params=get_args_of(float), nports=2, name=name, **kwargs)

    def compute(self):
        zn, len, epr, k1 = self.zn, self.len, self.epr, self.k1

        self.s = DatasheetCoax(zn=zn, epr=epr, len=len, k1=k1, frequency=self.frequency).s
        

class TransferSwitch(ParametricNetwork):
    """
    A lossy transfer switch. Currently modeled simply as a coaxial line with dielectric losses ignored.
    """
    def __init__(self, zn=50.0, len=10.0e-3, epr=1.0, k1=0.0, kc=0.0, separate_connector_loss=False, name='switch', **kwargs):
        self.separate_connector_loss = separate_connector_loss
        params = get_args_of(float)
        if not separate_connector_loss:
            params.pop('kc')
            
        super().__init__(params=params, nports=2, name=name, **kwargs)

    def compute(self):
        zn, len, epr, k1 = self.zn, self.len, self.epr, self.k1
        
        switch = DatasheetCoax(zn=zn, epr=epr, len=len, k1=k1, frequency=self.frequency)
        if self.separate_connector_loss:
            switch = DatasheetCoax(zn=zn, epr=epr, len=len, k1=k1, kc=self.kc, separate_connector_loss=self.separate_connector_loss, frequency=self.frequency)
        else:
            switch = DatasheetCoax(zn=zn, epr=epr, len=len, k1=k1, frequency=self.frequency)

        self.s = switch.s