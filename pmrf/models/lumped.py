from model import Model
from parameter import Parameter
from pmrf._numpy import numpy as np
from frequency import Frequency
  
class Capacitor(Model):
    C: Parameter = 1.0

    def s(self, freq: Frequency) -> np.ndarray:
        w = freq.w
        C = self.C

        z0_0 = z0_1 = self.z0
        denom = 1.0 + 1j * w * C * (z0_0 + z0_1)
        s11 = (1.0 - 1j * w * C * (z0_0.conj() - z0_1) ) / denom
        s22 = (1.0 - 1j * w * C * (z0_1.conj() - z0_0) ) / denom
        s12 = s21 = (2j * w * C * (z0_0.real * z0_1.real)**0.5) / denom

        s = np.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
                
class Inductor(Model):
    L: Parameter = 1.0

    def s(self, freq: Frequency) -> np.ndarray:
        L = self.L
        w = freq.w

        z0_0 = z0_1 = self.z0
        
        denom = (1j * w * L) + (z0_0 + z0_1)
        s11 = (1j * w * L - z0_0.conj() + z0_1) / denom
        s22 = (1j * w * L + z0_0 - z0_1.conj()) / denom
        s12 = s21 = 2 * (z0_0.real * z0_1.real)**0.5 / denom

        s = np.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s         

class Resistor(Model):
    R: Parameter = 1.0

    def s(self, freq: Frequency) -> np.ndarray:
        w = freq.w
        R = self.R
        z0_0 = z0_1 = self.z0
        ones = np.ones(freq.npoints, dtype=np.complex128)

        denom = R + (z0_0 + z0_1)
        s11 = ((R - z0_0.conj() + z0_1) / denom) * ones
        s22 = ((R + z0_0 - z0_1.conj()) / denom) * ones
        s12 = (2 * (z0_0.real * z0_1.real)**0.5 / denom) * ones
        s21 = s12

        s = np.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s
        

class Transformer(Model):
    def s(self, freq: Frequency):
        s = 0.5 * np.ones((freq.npoints, 4, 4), dtype=np.complex128)
        s[:, 0, 3] *= -1
        s[:, 1, 2] *= -1
        s[:, 2, 1] *= -1
        s[:, 3, 0] *= -1

        return s                