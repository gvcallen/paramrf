from parax.parameters import Uniform, Fixed
import pmrf as prf
from pmrf.models import Capacitor, Inductor, Circuit, Port, Ground

class PiCLC(prf.Model):
    capacitor1: Capacitor =     Capacitor(C=Fixed(1.0e-12))
    capacitor2: Capacitor =     Capacitor(C=Uniform(0.0, 10.0, value=2.0, scale=1e-12))
    inductor: Inductor =        Inductor(L=Uniform(0.0, 10.0, value=2.0, scale=1e-12))

    def __call__(self) -> prf.Model:
        # Instantiate the ports and grounds
        port1, port2, ground = Port(), Port(), Ground()

        # Create the connections list. This time, capacitor1, capacitor2 and inductor are members.
        connections = [
            [(port1, 0), (self.capacitor1, 1), (self.inductor, 1)], # E0
            [(port2, 0), (self.capacitor2, 1), (self.inductor, 0)], # E1
            [(ground, 0), (self.capacitor1, 0), (self.capacitor2, 0)], # E2
        ]

        # Return the model
        return Circuit(connections)