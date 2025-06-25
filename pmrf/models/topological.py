import pmrf.numpy as np
from pmrf._frequency import Frequency
from pmrf.parameters import Parameter
from pmrf._model import Model

import jax

class PiCLC(Model):
    C1: Parameter = 1.0e-12
    L: Parameter = 1.0e-9
    C2: Parameter = 1.0e-12

    def a(self, freq: Frequency) -> np.ndarray:
        return jax.lax.cond(
            self.L == 0.0, 
            lambda: self.a_zero_inductance(freq),
            lambda: self.a_general(freq),
        )
    
    def a_general(self, freq: Frequency):
        C1, C2, L = self.C1, self.C2, self.L
        w = freq.w
        Y1 = 1j * w * C1
        Y2 = 1j * w * C2
        Y3 = 1 / (1j * w * L)

        return np.array([
            [1 + Y2 / Y3,           1 / Y3      ],
            [Y1 + Y2 + Y1*Y2/Y3,    1 + Y1 / Y3 ],
        ]).transpose(2, 0, 1)        
    
    def a_zero_inductance(self, freq: Frequency):
        C1, C2 = self.C1, self.C2
        w = freq.w
        
        C = C1 + C2
        wC = w * C
        Y = 1j * wC
        return np.array([
            [np.ones_like(Y), np.zeros_like(Y)],
            [Y, np.ones_like(Y)]
        ]).transpose(2, 0, 1)