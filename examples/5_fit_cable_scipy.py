import logging
import skrf as rf

from parax.parameters import Uniform, RelativeNormal, Fixed
from pmrf.models import CoaxialLine
from pmrf.optimize import fit, ScipyMinimizer

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


# Fit the model
results = fit(model, measured, solver=ScipyMinimizer())
fitted_model = results.model

# Plot some results
fitted_model.to_skrf(measured.frequency).plot_s_db()
measured.plot_s_db()