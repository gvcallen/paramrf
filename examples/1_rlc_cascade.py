from parax import Parameter, Fixed
import pmrf as prf
from pmrf.models import Resistor, Inductor, ShuntCapacitor, OPEN

# Instantiate the lumped element models
resistor = Resistor(R=100.0)
inductor = Inductor(L=Parameter(2.0, scale=1e-9)) # we can optionally provide a parameter scale
capacitor = ShuntCapacitor(C=1.0e-12, name="cap") # naming makes parameter manipulation later easy

# Cascade the models, storing the result.
# We also create a terminated version with a new, fixed C
rlc = resistor ** inductor ** capacitor
terminated_rlc = rlc.terminated(OPEN).with_params(cap_C=Fixed(0.5e-12))

# Plot the S11 of the terminated model at a specified frequency range
freq = prf.Frequency(1, 1000, 1000, 'MHz')
terminated_rlc.plot_s_db(freq, m=0, n=0)