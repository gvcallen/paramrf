import pmrf.numpy as np
from pmrf._model import Model
from pmrf._frequency import Frequency

class Load(Model):
    nports: int = 1

    @property
    def gamma(self) -> float | np.ndarray:
        pass

    def s(self, freq: Frequency) -> np.ndarray:
        gamma, nports = self.gamma, self.nports
        s = np.array(gamma).reshape(-1, 1, 1) * \
            np.eye(nports, dtype=np.complex128).reshape((-1, nports, nports)).\
            repeat(freq.npoints, 0)
        return s
    
class Match(Load):
    @property
    def gamma(self) -> float | np.ndarray:
        return 0.0

class Short(Load):
    @property
    def gamma(self) -> float | np.ndarray:
        return -1.0

class Open(Load):
    @property
    def gamma(self) -> float | np.ndarray:
        return 1.0
  
class Capacitor(Model):
    C: float = 1.0

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
    L: float = 1.0

    def s(self, freq: Frequency) -> np.ndarray:
        L = self.L
        w = freq.w

        z0_0 = z0_1 = self.z0
        
        denom = (1j * w * L) + (z0_0 + z0_1)
        s11 = (1j * w * L - np.conj(z0_0) + z0_1) / denom
        s22 = (1j * w * L + z0_0 - np.conj(z0_1)) / denom
        # s11 = (1j * w * L - z0_0 + z0_1) / denom
        # s22 = (1j * w * L + z0_0 - z0_1) / denom
        s12 = s21 = 2 * (z0_0.real * z0_1.real)**0.5 / denom

        s = np.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)

        return s         

class Resistor(Model):
    R: float = 1.0

    def s(self, freq: Frequency) -> np.ndarray:
        R = self.R
        z0_0 = z0_1 = self.z0
        ones = np.ones(freq.npoints, dtype=np.complex128)

        denom = R + (z0_0 + z0_1)
        s11 = ((R - np.conj(z0_0) + z0_1) / denom) * ones
        s22 = ((R + z0_0 - np.conj(z0_1)) / denom) * ones
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