import pmrf as prf
from pmrf.models import Capacitor, Inductor, Circuit, Port, Ground

# Instantiate the elements, ports and grounds
capacitor1, capacitor2 = Capacitor(C=2e-12), Capacitor(C=1.5e-12)
inductor = Inductor(L=3e-9)
port1, port2 = Port(), Port()
ground = Ground()

# Create the connections list
connections = [
    [(port1, 0), (capacitor1, 1), (inductor, 1)], # E0
    [(port2, 0), (capacitor2, 1), (inductor, 0)], # E1
    [(ground, 0), (capacitor1, 0), (capacitor2, 0)], # E2
]

# Create the model and plot it's S21 parameter
pi_clc = Circuit(connections)
freq = prf.Frequency(1, 1000, 1001, 'MHz')
pi_clc.plot_s_db(freq, m=1, n=0)

# Note that ParamRF already provides a built in, more efficient PiCLC model
from pmrf.models import PiCLC
PiCLC(2e-12, 3e-9, 1.5e-12).plot_s_db(freq, m=1, n=0)