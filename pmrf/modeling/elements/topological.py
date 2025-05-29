import numpy as np
import skrf as rf

from pmrf.modeling import ParametricNetwork
from pmrf.misc.inspection import get_args_of
from pmrf.modeling.elements.lumped import Capacitor, Inductor
           
           
class PiCLC(ParametricNetwork):
    def __init__(self, C1=1.0, L=1.0, C2=1.0, three_port=False, name='pi_clc', **kwargs):
        if three_port:
            nports = 3
        else:
            nports = 2
        
        super().__init__(params=get_args_of(float), nports=nports, name=name, **kwargs)
    
    def compute(self) -> None:
        w = self.frequency.w
        C1, L, C2 = self.params['C1'], self.params['L'], self.params['C2']
        
        if self.L == 0.0:
            # Exception case to avoid div by zero (which pi_Y does not support)
            if self.nports == 3:
                C = C1 + C2
                port1 = rf.Circuit.Port(self.frequency, name='port1', z0=self.z0_port)
                port2 = rf.Circuit.Port(self.frequency, name='port2', z0=self.z0_port)
                port3 = rf.Circuit.Port(self.frequency, name='port3', z0=self.z0_port)

                capacitor = self.media.capacitor(C)
                capacitor.name = 'capacitor'

                cnx = [
                    [(port1, 0), (port2, 0), (capacitor, 1)],
                    [(port3, 0), (capacitor, 0)]
                ]

                self.s = rf.Circuit(cnx).network.s
            else:
                a = np.zeros((self.frequency.npoints, 2, 2), dtype=complex)
            
                C = C1 + C2
                wC = w * C
                Y = 1j * wC
                a[:,0,0] = 1
                a[:,0,1] = 0
                a[:,1,0] = Y
                a[:,1,1] = 1
            
                self.a = a
        else:
            Y1 = 1j * w * C1
            Y2 = 1j * w * C2
            Y3 = 1 / (1j * w * L)
            
            if self.nports == 2:
                a = np.zeros((self.frequency.npoints, 2, 2), dtype=complex)
                a[:,0,0] = 1 + Y2 / Y3
                a[:,0,1] = 1 / Y3
                a[:,1,0] = Y1 + Y2 + Y1 * Y2 / Y3
                a[:,1,1] = 1 + Y1 / Y3

                self.a = a
            else:
                y = np.zeros((self.frequency.npoints, 3, 3), dtype=complex)

                y[:,0,0] = Y1 + Y3
                y[:,0,1] = -Y3
                y[:,0,2] = -Y1
                y[:,1,0] = -Y3
                y[:,1,1] = Y2 + Y3
                y[:,1,2] = -Y2
                y[:,2,0] = -Y1
                y[:,2,1] = -Y2
                y[:,2,2] = Y1 + Y2

                self.y = y                
            
class BoxCLCC(ParametricNetwork):
    def __init__(self, C1=1.0, L=1.0, C2=1.0, C3=1.0, four_port=False, **kwargs):
        if not four_port:
            raise Exception('Three or two-port box not yet derived')
        
        super().__init__(params=get_args_of(float), nports=4, **kwargs)
        
    def compute(self):
        cap1 = Capacitor(self.C1, name='C1', frequency=self.frequency, media=self.media)
        cap2 = Capacitor(self.C2, name='C2', frequency=self.frequency, media=self.media)
        cap3 = Capacitor(self.C3, name='C3', frequency=self.frequency, media=self.media)
        ind1 = Inductor(self.L, name='L', frequency=self.frequency, media=self.media)
        
        port1 = rf.Circuit.Port(self.frequency, 'port1', self.z0_port)
        port2 = rf.Circuit.Port(self.frequency, 'port2', self.z0_port)
        port3 = rf.Circuit.Port(self.frequency, 'port3', self.z0_port)
        port4 = rf.Circuit.Port(self.frequency, 'port4', self.z0_port)
        
        cnx = [
            [(port1, 0), (cap1, 1), (ind1, 1)],
            [(port2, 0), (cap2, 1), (ind1, 0)],
            [(port3, 0), (cap1, 0), (cap3, 1)],
            [(port4, 0), (cap2, 0), (cap3, 0)],
        ]
        
        self.s = rf.Circuit(cnx).network.s
        
        
        # C1, C2, L, C3 = self.params['C1'], self.params['C2'], self.params['L'], self.params['C3']
        
        # w = self.frequency.w
        # y = np.zeros((self.frequency.npoints, 4, 4), dtype=complex)

        # Y1 = 1j*w*C1
        # Y2 = 1j*w*C2
        # Y3 = 1/(1j*w*L)
        # Y4 = 1j*w*C3

        # y[:,0,0] = Y1 + Y3
        # y[:,0,1] = -Y3
        # y[:,0,2] = -Y1
        # y[:,0,3] = 0
        # y[:,1,0] = -Y3
        # y[:,1,1] = Y2 + Y3
        # y[:,1,2] = 0
        # y[:,1,3] = -Y2
        # y[:,2,0] = -Y1
        # y[:,2,1] = 0
        # y[:,2,2] = Y1 + Y4
        # y[:,2,3] = -Y4
        # y[:,3,0] = 0
        # y[:,3,1] = -Y2
        # y[:,3,2] = -Y4
        # y[:,3,3] = Y2 + Y4
        
        # self.y = y
        
        