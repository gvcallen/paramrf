import jax.numpy as jnp
import pmrf as prf
from parax import Parameter

# Define a model class. Behaviour is defined by implementing 
# a primary matrix function such as "s" in this case.
class Capacitor(prf.Model):
    C: Parameter = 1.0e-12

    def s(self, freq: prf.Frequency) -> jnp.ndarray:
        w = freq.w
        C = self.C

        z0_0 = z0_1 = self.z0
        denom = 1.0 + 1j * w * C * (z0_0 + z0_1)
        s11 = (1.0 - 1j * w * C * (jnp.conj(z0_0) - z0_1) ) / denom
        s22 = (1.0 - 1j * w * C * (jnp.conj(z0_1) - z0_0) ) / denom
        s12 = s21 = (2j * w * C * (z0_0.real * z0_1.real)**0.5) / denom

        return jnp.array([
            [s11, s12],
            [s21, s22]
        ]).transpose(2, 0, 1)