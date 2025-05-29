import numpy as np
import skrf as rf

from pmrf.core import ParametricNetwork
from pmrf.modeling.elements.lumped import Resistor
from pmrf.modeling.elements.topological import PiCLC
from pmrf.misc.inspection import get_args_of

class PhysicalResistor(ParametricNetwork):
    """
    A physical resistor, with parasitics.
    """
    def __init__(self, R = 1.0, C1 = 1.0e-12, L = 1.0e-9, C2 = 1.0e-12, terminated = False, **kwargs):
        self.terminated = terminated
        
        if not self.terminated:
            raise Exception('Non-terminated physical resistor model not yet implemented')
        
        super().__init__(get_args_of(float), nports=1, **kwargs)
        
    def compute(self):
        # Re-writing the maths explicitly in terms of input admittance results in a 25-30% total speedup
        # self.s = (self.parasitics ** self.ideal).s
        C1, L, C2, R = self.C1, self.L, self.C2, self.R
        w = self.frequency.w

        Y1 = 1j * w * C1
        Y2 = 1j * w * C2 + (1 / R)
        Ys = Y2 / ((1j * w * L * Y2) + 1)
        Yin = Y1 + Ys

        Z0 = self.z0_port
        self.s[:,0,0] = ((1 - Yin * Z0) / (1 + Yin * Z0))

    @property
    def ideal(self) -> rf.Network:
        return Resistor(self.R, z0_port=self.z0_port, frequency=self.frequency, terminated=self.terminated)
    
    @property
    def parasitics(self) -> rf.Network:
        """
        The parasitics of the resistor. When terminated, port 1 is the same as port 1 of the full terminated network.
        """
        return PiCLC(self.C1, self.L, self.C2, frequency=self.frequency)