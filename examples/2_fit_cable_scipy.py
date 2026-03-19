import matplotlib.pyplot as plt
import logging
import skrf as rf

from pmrf.core import CoaxialLine
from parax.parameters import Uniform, RelativeNormal, Fixed
from pmrf.fitting import SciPyMinimizeFitter

logging.basicConfig(level=logging.INFO)

# Load the measured data.
measured = rf.Network('data/10m_cable.s2p', f_unit='MHz')

# Setup the model. Note that any parameters not passed
# are set as free with infinite bounds
model = CoaxialLine(
    din = RelativeNormal(1.12, 0.05, scale=1e-3),
    dout = RelativeNormal(3.2, 0.05, scale=1e-3),
    epr = Fixed(1.384),
    rho = RelativeNormal(1.6, 0.05, scale=1e-8),
    tand = Uniform(0.0, 0.01, value=0.0, scale=0.01),
    length = RelativeNormal(10.0, 0.05),
    mur = Fixed(1.0),
)

# Initialize the fitter
fitter = SciPyMinimizeFitter(model)

# Run the fit and plot some results
fitted_model, fit_results = fitter.run(measured)
fit_results.plot_s_db()
plt.show()