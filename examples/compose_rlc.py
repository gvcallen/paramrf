from pmrf.models import Resistor, Inductor, Capacitor, ShuntCapacitor, SHORT, Load
from pmrf.parameters import Fixed
import skrf as rf
from skrf.media import DefinedGammaZ0
import matplotlib.pyplot as plt

frequency = rf.Frequency(50, 200, 151, 'MHz')

# Params
R = 100.0
L = 20e-9
C = 10e-10

# Instantiate the lumped element models
resistor = Resistor(R=R, name="res")
inductor = Inductor(L=L)
capacitor = ShuntCapacitor(C=C)

# Cascade the models, storing the result, and a terminated version with fixed R
rlc = resistor ** inductor ** capacitor
terminated_rlc = rlc
# terminated_rlc = rlc.terminated(OPEN).with_params(res_R=Fixed(90.0))