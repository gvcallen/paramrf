from pmrf._numpy import numpy as np
from pmrf._frequency import Frequency
from pmrf._model import Model

class PiCLC(Model):
    C1: float = 1.0e-12
    L: float = 1.0e-9
    C2: float = 1.0e-12

    def a(self, freq: Frequency) -> np.ndarray:
        C1, C2, L = self.C1, self. C2, self.L
        w = freq.w
        
        a = np.zeros((freq.npoints, 2, 2), dtype=np.complex128)
        if L == 0.0:
            C = C1 + C2
            wC = w * C
            Y = 1j * wC            
            a = a.at[:,0,0].set(1)
            a = a.at[:,0,1].set(0)
            a = a.at[:,1,0].set(Y)
            a = a.at[:,1,1].set(1)
        else:
            Y1 = 1j * w * C1
            Y2 = 1j * w * C2
            Y3 = 1 / (1j * w * L)
            
            a = a.at[:,0,0].set(1 + Y2 / Y3)
            a = a.at[:,0,1].set(1 / Y3)
            a = a.at[:,1,0].set(Y1 + Y2 + Y1 * Y2 / Y3)
            a = a.at[:,1,1].set(1 + Y1 / Y3)

        return a