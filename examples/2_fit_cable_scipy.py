import logging
import skrf as rf

from pmrf.models import CoaxialLine
from pmrf.parameters import Uniform, RelativeNormal, Fixed
from pmrf.fitting import SciPyMinimizeFitter

logging.basicConfig(level=logging.INFO)

# Load the measured data and setup the model
measured = rf.Network('../paramrf/examples/data/10m_cable.s2p', f_unit='MHz')
model = CoaxialLine(
    din = RelativeNormal(1.12, 0.05, scale=1e-3),
    dout = RelativeNormal(3.2, 0.05, scale=1e-3),
    epr = Fixed(1.384),
    rho = RelativeNormal(1.6, 0.05, scale=1e-8),
    tand = Uniform(0.0, 0.01, value=0.0, scale=0.01),
    length = RelativeNormal(10.0, 0.05),
)

# Initialize the fitter
fitter = SciPyMinimizeFitter(model)

# Run the fit and plot some results
fitted_model, fit_results = fitter.run(measured, solver='Nelder-Mead')
fit_results.plot_s_db()